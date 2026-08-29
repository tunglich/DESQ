import os
import gc
import json
import sys
import time
import warnings
from datetime import datetime, timedelta

"""Attention + FCN + Flooding + Keras Tuner training script.

Pipeline:
1) Load features and labels for a single ticker / factor pair.
2) Clean features, apply scaling, and cut the series into sliding windows.
3) Build an Attention + FCN model via HyperModel and search hyperparameters
   with Bayesian Optimization.
4) Evaluate trial metrics under time-series splits inside a custom Tuner.
5) Iterate over tickers in the main flow and emit the best trial result.

Design goals:
- Prevent time-series data leakage.
- Stay stable on Blackwell GPUs (RTX 50 series) under WSL.
- Preserve observability (key metrics and warnings) and maintainability (clear sections).
"""

# Reduce TensorFlow/XLA log noise and first-compile stalls (must be set before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 'TF_USE_LEGACY_KERAS' not in os.environ:
    os.environ['TF_USE_LEGACY_KERAS'] = '0'
if 'TF_GPU_ALLOCATOR' not in os.environ:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import recall_score, fbeta_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import keras_tuner as kt
try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None
import platform

# On WSL / single-GPU setups, avoid grabbing all VRAM at once to reduce the risk
# of low-level aborts (core dumped)
try:
    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu_device in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu_device, True)
except Exception:
    pass

# from numba import cuda
warnings.filterwarnings('ignore')

# Stay on native Keras optimizer/metrics to sidestep native crashes (double free)
# seen on some TF/WSL/GPU combinations
USE_TFA_OPTIMIZER = (os.getenv('ENABLE_TFA_OPTIMIZER', '0') == '1') and (tfa is not None)
TRAIN_MODE = os.getenv('TRAIN_MODE', 'speed').lower()
if TRAIN_MODE not in {'speed', 'safe', 'custom'}:
    TRAIN_MODE = 'speed'

if TRAIN_MODE == 'safe':
    FORCE_MODEL_BUILD_ON_CPU = True
    USE_EAGER_TRAINING = True
elif TRAIN_MODE == 'custom':
    FORCE_MODEL_BUILD_ON_CPU = os.getenv('FORCE_MODEL_BUILD_ON_CPU', '0') == '1'
    USE_EAGER_TRAINING = os.getenv('USE_EAGER_TRAINING', '0') == '1'
else:
    # speed mode (default): guard against stale shell env vars accidentally enabling slow eager mode
    FORCE_MODEL_BUILD_ON_CPU = False
    USE_EAGER_TRAINING = False

LOOKBACK_WINDOW_CHOICES = [5, 10, 20, 30, 40, 60]
STAGE1_MAX_TRIALS = int(os.getenv('STAGE1_MAX_TRIALS', '12'))
STAGE2_MAX_TRIALS = int(os.getenv('STAGE2_MAX_TRIALS', '24'))
STAGE1_EPOCHS = int(os.getenv('STAGE1_EPOCHS', '80'))
STAGE2_EPOCHS = int(os.getenv('STAGE2_EPOCHS', '120'))
FIT_VERBOSE = int(os.getenv('FIT_VERBOSE', '2'))
TRIAL_TIMEOUT_SECONDS = int(os.getenv('TRIAL_TIMEOUT_SECONDS', '3600'))
ENABLE_TRIAL_WARMUP = os.getenv('ENABLE_TRIAL_WARMUP', '0') == '1'
GPU_MEMORY_LIMIT_MB = int(os.getenv('GPU_MEMORY_LIMIT_MB', '0'))
ISOLATE_STOCK_MODEL_RUNS = os.getenv('ISOLATE_STOCK_MODEL_RUNS', '1') == '1'
ISOLATED_CHILD_RUN = os.getenv('ISOLATED_CHILD_RUN', '0') == '1'
ENABLE_KERAS_PICKLE_HOTFIX = os.getenv('ENABLE_KERAS_PICKLE_HOTFIX', '0') == '1'
ENABLE_XLA = os.getenv('ENABLE_XLA', '1') == '1'
ENABLE_TF32 = os.getenv('ENABLE_TF32', '1') == '1'
ENABLE_MIXED_PRECISION = os.getenv('ENABLE_MIXED_PRECISION', '1') == '1'
_WS_DIR = os.path.dirname(os.path.abspath(__file__))
ATT_HYPERBAYES_DIR = os.getenv('HYPERBAYES_ATT_DIR', os.path.join(_WS_DIR, 'hyper'))
ATT_SCALER_DIR = os.getenv('FEATURE_SCALER_ATT_DIR', os.path.join(_WS_DIR, 'scalar'))
ATT_DATA_DIR = os.getenv('DATA_ROOT', os.path.join(_WS_DIR, 'feature'))

# Walk-Forward Validation options
# VALIDATION_MODE: 'blocking' (default, single block CV — prior behavior),
#                  'walk_forward_rolling'   — fixed-size train window slides forward,
#                  'walk_forward_expanding' — train start fixed, train window grows.


def _normalize_validation_mode(raw_mode: str) -> str:
    """Normalize a validation-mode alias to the internal value."""
    mode = (raw_mode or '').strip().lower()
    alias = {
        'traditional': 'blocking',
        'block': 'blocking',
        'blocking': 'blocking',
        'rolling': 'walk_forward_rolling',
        'walk_forward_rolling': 'walk_forward_rolling',
        'walk-forward-rolling': 'walk_forward_rolling',
        'walkforwardrolling': 'walk_forward_rolling',
        'expanding': 'walk_forward_expanding',
        'walk_forward_expanding': 'walk_forward_expanding',
        'walk-forward-expanding': 'walk_forward_expanding',
        'walkforwardexpanding': 'walk_forward_expanding',
    }
    return alias.get(mode, mode)


def _resolve_validation_mode() -> str:
    """Decide the validation mode.

    Resolution order:
      1) `VALIDATION_MODE` env var (accepts traditional/blocking/rolling/expanding aliases).
      2) If unset and running under an interactive TTY (not sandboxed subprocess):
         prompt the user (default rolling).
      3) Otherwise fall back to `blocking` (legacy behaviour).
    """
    raw = os.getenv('VALIDATION_MODE')
    if raw is not None:
        normalized = _normalize_validation_mode(raw)
        if normalized in {'blocking', 'walk_forward_rolling', 'walk_forward_expanding'}:
            return normalized
        print(f"[WARN] Unknown VALIDATION_MODE={raw!r}, fallback to 'blocking'.")
        return 'blocking'

    if ISOLATED_CHILD_RUN or not sys.stdin.isatty():
        return 'blocking'

    print("Please select validation mode:")
    print("  1) traditional (blocking)")
    print("  2) walk-forward expanding")
    print("  3) walk-forward rolling [default]")
    try:
        answer = input("Enter 1/2/3 (default 3): ").strip().lower()
    except EOFError:
        return 'walk_forward_rolling'

    if answer in {'', '3', 'rolling', 'walk_forward_rolling', 'walk-forward-rolling'}:
        return 'walk_forward_rolling'
    if answer in {'2', 'expanding', 'walk_forward_expanding', 'walk-forward-expanding'}:
        return 'walk_forward_expanding'
    if answer in {'1', 'traditional', 'blocking', 'block'}:
        return 'blocking'

    print(f"[WARN] Unknown selection={answer!r}, fallback to default 'walk_forward_rolling'.")
    return 'walk_forward_rolling'


VALIDATION_MODE = _resolve_validation_mode()
WF_N_SPLITS = int(os.getenv('WF_N_SPLITS', '5'))
WF_VAL_RATIO = float(os.getenv('WF_VAL_RATIO', '0.2'))
WF_GAP = int(os.getenv('WF_GAP', '50'))
WF_VAL_YEARS = float(os.getenv('WF_VAL_YEARS', '0'))
WF_TRADING_DAYS_PER_YEAR = int(os.getenv('WF_TRADING_DAYS_PER_YEAR', '252'))
WF_VAL_SAMPLES = int(round(WF_VAL_YEARS * WF_TRADING_DAYS_PER_YEAR)) if WF_VAL_YEARS > 0 else 0


def _resolve_feature_preprocess():
    """Decide whether to run feature preprocessing (corr filter + sanitize + scaler).

    Resolution order:
      1) `FEATURE_PREPROCESS` env var (0/no/false = off; 1/yes/true = on).
      2) If unset and running under an interactive TTY (not sandboxed subprocess): prompt the user.
      3) Otherwise on by default.
    """
    raw = os.getenv('FEATURE_PREPROCESS')
    if raw is not None:
        return raw.strip().lower() not in {'0', 'no', 'false', 'off', ''}
    if ISOLATED_CHILD_RUN or not sys.stdin.isatty():
        return True
    try:
        answer = input("Run feature preprocessing (corr filter + sanitize + Yeo-Johnson/Robust scaler)? [Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer not in {'n', 'no', '0', 'false'}

FAST_DEBUG = os.getenv('FAST_DEBUG', '0') == '1'
if FAST_DEBUG:
    STAGE1_MAX_TRIALS = int(os.getenv('STAGE1_MAX_TRIALS', '2'))
    STAGE2_MAX_TRIALS = int(os.getenv('STAGE2_MAX_TRIALS', '4'))
    STAGE1_EPOCHS = int(os.getenv('STAGE1_EPOCHS', '12'))
    STAGE2_EPOCHS = int(os.getenv('STAGE2_EPOCHS', '24'))
    print(
        f"[DEBUG] FAST_DEBUG enabled: "
        f"stage1_trials={STAGE1_MAX_TRIALS}, stage2_trials={STAGE2_MAX_TRIALS}, "
        f"stage1_epochs={STAGE1_EPOCHS}, stage2_epochs={STAGE2_EPOCHS}, fit_verbose={FIT_VERBOSE}"
    )

# In walk-forward mode search can be scaled down automatically to shorten wall-time.
# Default: no downscaling (matches the TW full-budget setup). Set
# AUTO_REDUCE_SEARCH_FOR_WF=1 to opt in.
_stage_budget_overridden = any(
    os.getenv(k) is not None
    for k in ('STAGE1_MAX_TRIALS', 'STAGE2_MAX_TRIALS', 'STAGE1_EPOCHS', 'STAGE2_EPOCHS')
)
if (
    (not FAST_DEBUG)
    and (not _stage_budget_overridden)
    and (VALIDATION_MODE in ('walk_forward_rolling', 'walk_forward_expanding'))
    and (os.getenv('AUTO_REDUCE_SEARCH_FOR_WF', '0') == '1')
):
    STAGE1_MAX_TRIALS = min(STAGE1_MAX_TRIALS, 8)
    STAGE2_MAX_TRIALS = min(STAGE2_MAX_TRIALS, 12)
    STAGE1_EPOCHS = min(STAGE1_EPOCHS, 60)
    STAGE2_EPOCHS = min(STAGE2_EPOCHS, 90)
    print(
        "[SPEED] walk-forward detected, auto-reduced search budget: "
        f"stage1_trials={STAGE1_MAX_TRIALS}, stage2_trials={STAGE2_MAX_TRIALS}, "
        f"stage1_epochs={STAGE1_EPOCHS}, stage2_epochs={STAGE2_EPOCHS}"
    )

print(
    "[BUDGET] "
    f"stage1_trials={STAGE1_MAX_TRIALS}, stage2_trials={STAGE2_MAX_TRIALS}, "
    f"stage1_epochs={STAGE1_EPOCHS}, stage2_epochs={STAGE2_EPOCHS}"
)

tf.config.run_functions_eagerly(USE_EAGER_TRAINING)
try:
    tf.config.optimizer.set_jit(ENABLE_XLA)
except Exception:
    pass
try:
    tf.config.experimental.enable_tensor_float_32_execution(ENABLE_TF32)
except Exception:
    pass

if ENABLE_MIXED_PRECISION:
    try:
        mixed_precision.set_global_policy('mixed_float16')
    except Exception as mixed_precision_error:
        print(f"[WARN] mixed precision setup failed: {mixed_precision_error}")

print(
    f"[RUNTIME] train_mode={TRAIN_MODE}, eager_training={USE_EAGER_TRAINING}, "
    f"force_build_on_cpu={FORCE_MODEL_BUILD_ON_CPU}, trial_timeout={TRIAL_TIMEOUT_SECONDS}s, "
    f"fit_verbose={FIT_VERBOSE}, enable_trial_warmup={ENABLE_TRIAL_WARMUP}, "
    f"gpu_allocator={os.environ.get('TF_GPU_ALLOCATOR')}, "
    f"gpu_memory_limit_mb={GPU_MEMORY_LIMIT_MB}, "
    f"enable_xla={ENABLE_XLA}, enable_tf32={ENABLE_TF32}, "
    f"enable_mixed_precision={ENABLE_MIXED_PRECISION}, use_tfa_optimizer={USE_TFA_OPTIMIZER}, "
    f"isolate_stock_model_runs={ISOLATE_STOCK_MODEL_RUNS}, isolated_child_run={ISOLATED_CHILD_RUN}, "
    f"enable_keras_pickle_hotfix={ENABLE_KERAS_PICKLE_HOTFIX}"
)

TRIAL_PREP_CACHE = {}

def make_dataset_cache_key(prefix, x, y):
    """Build a data-specific cache key so distinct datasets do not collide."""
    x_ptr = int(x.__array_interface__['data'][0]) if hasattr(x, '__array_interface__') else id(x)
    y_ptr = int(y.__array_interface__['data'][0]) if hasattr(y, '__array_interface__') else id(y)
    return (
        prefix,
        x_ptr,
        y_ptr,
        tuple(x.shape),
        tuple(y.shape),
    )

def platform_path(win_path: str) -> str:
    """Convert a Windows-style path into one usable on the current system.

    Args:
        win_path: path string, possibly in Windows form (e.g. `D:/...`).

    Returns:
        Original path on Windows; `/mnt/<drive>/...` on Linux / WSL.

    Side effects:
        None.
    """
    if platform.system() != 'Windows' and len(win_path) >= 2 and win_path[1] == ':':
        drive = win_path[0].lower()
        return '/mnt/' + drive + win_path[2:].replace('\\', '/')
    return win_path


SCALER_ROOT = platform_path(ATT_SCALER_DIR)
os.makedirs(SCALER_ROOT, exist_ok=True)

class SinusoidalPositionalEncoding(layers.Layer):
    """Fixed sinusoidal / cosine positional-encoding layer.

    Purpose:
        Replace third-party PositionEmbedding with a parameter-free positional signal.
        Compatible with mixed precision (auto-casts the encoding dtype).

    Args:
        None (dimensions are inferred from the input tensor shape).

    Attributes:
        No persistent trainable parameters.

    Side effects:
        None.
    """
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        d_model_float = tf.cast(d_model, tf.float32)
        positions = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
        dims = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
        angles = positions / tf.pow(10000.0, 2.0 * tf.math.floor(dims / 2.0) / d_model_float)
        sin_mask = tf.math.mod(tf.range(d_model), 2)
        cos_mask = 1 - sin_mask
        encoding = (tf.sin(angles) * tf.cast(cos_mask, tf.float32)
                    + tf.cos(angles) * tf.cast(sin_mask, tf.float32))
        encoding = encoding[tf.newaxis, :, :]
        # Force-cast the encoding to match the input dtype (supports mixed-precision float16)
        encoding = tf.cast(encoding, x.dtype)
        return x + encoding
    
    def get_config(self):
        return super().get_config()

# ==============================================================
# Purpose of this file:
# 1) Load features (X) and labels (y) for a single ticker.
# 2) Clean / normalize features and cut into time-series windows.
# 3) Classify with a variable-depth Attention + FCN model.
# 4) Search hyperparameters via Keras Tuner (Bayesian Optimization).
# 5) Use time-series cross-validated metrics as the trial objective.
# ==============================================================

# Time-series cross-validation uses the paper's 50-anchor effective gap by default.
class BlockingTimeSeriesSplit:
    """Time-series splitter (Blocking CV).

    Purpose:
        Split data chronologically and insert a gap between train and val to prevent leakage.

    Args:
        n_splits: number of splits.
        val_ratio: validation fraction per block.
        gap: samples between train and val.

    Attributes:
        n_splits, val_ratio, gap。

    Side effects:Side effects:
        None.
    """
    def __init__(self, n_splits, val_ratio=0.25, gap=50):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.gap = gap
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    # Split chronologically; leave a gap of `gap` days between train and val to avoid leakage
    def split(self, X, y=None, groups=None):
        """Yield (train_idx, val_idx) for each fold.

        Args
        ----
        X : array-like
            time-sorted sample sequence.
        y, groups : kept for sklearn-interface compatibility.
        """
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            start = i * k_fold_size
            end = min(start + k_fold_size, n_samples)
            val_size = int(k_fold_size * self.val_ratio)
            train_end = max(start, end - val_size - self.gap)
            val_start = min(train_end + self.gap, end)

            train_index = np.arange(start, train_end)
            val_index = np.arange(val_start, end)
            if len(train_index) == 0 or len(val_index) == 0:
                continue
            yield train_index, val_index


class WalkForwardSplit:
    """Walk-Forward Validation splitter (supports rolling / expanding).

    Purpose:
        Simulate a rolling "train on past, validate on the immediate future" scheme
        to avoid look-ahead bias and to catch regime shifts.

    Args:
        n_splits: number of folds, one validation window per fold.
        val_ratio: validation window as a fraction of the full series (default 0.2).
        val_samples: validation window size (>0 overrides val_ratio; useful for fixed year windows).
        gap: samples between train and val to avoid label leakage.
        mode: 'rolling' or 'expanding'.

    Attributes:
        n_splits, val_ratio, gap, mode。

    Side effects:Side effects:
        None.
    """

    def __init__(self, n_splits=5, val_ratio=0.2, gap=50, mode='rolling', val_samples=0):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if val_samples <= 0 and not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be in (0, 1)")
        if val_samples < 0:
            raise ValueError("val_samples must be >= 0")
        if mode not in ('rolling', 'expanding'):
            raise ValueError(f"Unknown WalkForward mode: {mode}")
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.val_samples = int(val_samples)
        self.gap = gap
        self.mode = mode

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yield (train_idx, val_idx) pairs in chronological order across n_splits rolling windows.

        Validation windows are equal-length and non-overlapping. In rolling mode
        train length is fixed; in expanding mode train start is fixed and train
        length grows with the window.
        """
        n_samples = int(len(X))
        if n_samples <= self.gap + 2:
            return

        if self.val_samples > 0:
            val_size = max(1, int(self.val_samples))
        else:
            val_size = max(1, int(n_samples * self.val_ratio / self.n_splits))
        # Reserve n_splits validation windows plus at least one train window + gap.
        total_val = val_size * self.n_splits
        if total_val + self.gap + 1 >= n_samples:
            # Shrink val_size until it fits
            val_size = max(1, (n_samples - self.gap - 2) // (self.n_splits + 1))
            total_val = val_size * self.n_splits

        # rolling train_len: all folds share the same train length, sized from the first fold's available room.
        first_val_start = n_samples - total_val
        first_train_end = first_val_start - self.gap
        rolling_train_len = max(1, first_train_end)

        for i in range(self.n_splits):
            val_start = n_samples - total_val + i * val_size
            val_end = val_start + val_size
            train_end = val_start - self.gap
            if train_end <= 0:
                continue
            if self.mode == 'rolling':
                train_start = max(0, train_end - rolling_train_len)
            else:  # expanding
                train_start = 0
            train_index = np.arange(train_start, train_end)
            val_index = np.arange(val_start, min(val_end, n_samples))
            if len(train_index) == 0 or len(val_index) == 0:
                continue
            yield train_index, val_index


def build_validation_splitter():
    """Pick the time-series splitter specified by the VALIDATION_MODE env var."""
    if VALIDATION_MODE == 'walk_forward_rolling':
        return WalkForwardSplit(
            n_splits=WF_N_SPLITS,
            val_ratio=WF_VAL_RATIO,
            gap=WF_GAP,
            mode='rolling',
            val_samples=WF_VAL_SAMPLES,
        )
    if VALIDATION_MODE == 'walk_forward_expanding':
        return WalkForwardSplit(
            n_splits=WF_N_SPLITS,
            val_ratio=WF_VAL_RATIO,
            gap=WF_GAP,
            mode='expanding',
            val_samples=WF_VAL_SAMPLES,
        )
    if VALIDATION_MODE != 'blocking':
        print(f"[WARN] Unknown VALIDATION_MODE={VALIDATION_MODE!r}, fallback to 'blocking'.")
    return BlockingTimeSeriesSplit(n_splits=1, gap=WF_GAP)

def val_windows(data, ref_day=60, period=20): 
    """Convert a time-series DataFrame into supervised windows.

    Args:
        data: DataFrame with feature and label columns; the last 4 columns are treated as labels / reserved.
        ref_day: number of lookback days per sample (window length).
        period: label column suffix corresponding to `y_{period}`.

    Returns:
        (X_val, y_val)
        - X_val: shape=(n_samples, ref_day, n_features)
        - y_val: shape=(n_samples,)

    Side effects:Side effects:
        None.
    """
    n_features = data.shape[1] - 4
    feat_arr = np.ascontiguousarray(data.iloc[:, :-4].to_numpy(dtype=np.float64))
    n_windows = feat_arr.shape[0] - ref_day + 1
    shape = (n_windows, ref_day, n_features)
    strides = (feat_arr.strides[0],) + feat_arr.strides
    X_val = np.lib.stride_tricks.as_strided(feat_arr, shape=shape, strides=strides).copy()
    y_val = data[f"y_{period}"].to_numpy()[ref_day - 1:]
    return X_val, y_val
    
# Maximize usable data: PE / PB require at least 3 years of history, so training starts at least 3*250 trading days in.
# cast_to_floatx converts numpy arrays to the Keras floating-point dtype.
def make_datasets(X, y, idx, start, end, ref_day):
    """Slice train / test by date and apply a minimum history-length guard.

    Args:
        X: windowed feature array.
        y: label array aligned with X.
        idx: original data date index (DatetimeIndex).
        start: start-date string (YYYY-MM-DD).
        end: end-date string (YYYY-MM-DD).
        ref_day: window length, used to back out the index.

    Returns:
        (X_slice, y_slice), both in Keras floatx dtype.

    Side effects:
        None.
    """
    start_idx = idx.get_loc(idx.to_series()[start:].iloc[0])
    end_idx = idx.get_loc(idx.to_series()[:end].iloc[-1])
    return K.cast_to_floatx(X[max(1*250-1, start_idx-ref_day+1):end_idx-ref_day+2]), K.cast_to_floatx(y[max(1*250-1, start_idx-ref_day+1):end_idx-ref_day+2])

def fit_sanitize_statistics(df, max_abs=1e6, q=0.001):
    """Estimate sanitize statistics on the training slice to avoid look-ahead leakage."""
    numeric_df = df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce').astype(np.float64)

    fill_values = numeric_df.median().fillna(0.0)
    filled = numeric_df.fillna(fill_values)
    lower = filled.quantile(q)
    upper = filled.quantile(1 - q)
    return {
        'fill_values': fill_values,
        'lower': lower,
        'upper': upper,
        'max_abs': float(max_abs)
    }


def apply_sanitize_statistics(df, stats):
    """Apply the fixed sanitize statistics so training and inference stay consistent."""
    max_abs = float(stats.get('max_abs', 1e6))
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    sanitized = sanitized.apply(pd.to_numeric, errors='coerce').astype(np.float64)

    fill_values = stats['fill_values'].reindex(sanitized.columns).fillna(0.0)
    lower = stats['lower'].reindex(sanitized.columns).fillna(-max_abs)
    upper = stats['upper'].reindex(sanitized.columns).fillna(max_abs)

    sanitized = sanitized.fillna(fill_values)
    sanitized = sanitized.clip(lower=lower, upper=upper, axis=1)
    sanitized = sanitized.clip(lower=-max_abs, upper=max_abs)
    sanitized = np.sign(sanitized) * np.log1p(np.abs(sanitized))

    values = np.nan_to_num(
        sanitized.to_numpy(dtype=np.float64, copy=False),
        nan=0.0,
        posinf=np.log1p(max_abs),
        neginf=-np.log1p(max_abs)
    )
    return pd.DataFrame(values, index=sanitized.index, columns=sanitized.columns)


def sanitize_feature_values(feature_df, fit_start, fit_end):
    """Backwards-compat: fit sanitize statistics on the fit slice, then apply to all data."""
    fit_slice = feature_df.loc[fit_start:fit_end]
    if fit_slice.empty:
        fit_slice = feature_df
    sanitize_stats = fit_sanitize_statistics(fit_slice)
    return apply_sanitize_statistics(feature_df, sanitize_stats)


tf.get_logger().setLevel('ERROR')

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.utils import to_categorical

def configure_gpu():
    """Enable GPU memory growth to avoid a one-shot VRAM grab.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Mutates the TensorFlow runtime GPU memory growth setting.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if GPU_MEMORY_LIMIT_MB > 0:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MB)]
                    )
                print(f"[GPU] set logical memory limit to {GPU_MEMORY_LIMIT_MB} MB per GPU")
            else:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

configure_gpu()

def warmup_one_batch(model, x_train, y_train, batch_size):
    """Run 1 warm-up batch to move first-time kernel compilation earlier.

    Args:
        model: compiled Keras model.
        x_train: training feature array.
        y_train: training label array.
        batch_size: max warm-up batch size.

    Returns:
        Warm-up duration in seconds.

    Side effects:
        Triggers a forward-inference / eval graph compile; does not update weights.
    """
    warmup_size = min(int(batch_size), int(x_train.shape[0]))
    if warmup_size <= 0:
        return 0.0
    if USE_EAGER_TRAINING:
        return 0.0
    x_warm = x_train[:warmup_size]
    y_warm = y_train[:warmup_size]
    started_at = time.time()
    try:
        model.test_on_batch(x_warm, y_warm, return_dict=False)
    except Exception as warmup_error:
        print(f"[WARN] warmup skipped: {warmup_error}")
        return 0.0
    return time.time() - started_at

def get_trial_prep_cache(cache_key, x, y):
    """Cache split indices and weights per window so per-trial computation is cheaper.

    Return shape: ``{'folds': [{train_indices, val_indices, class_weights, val_weights}, ...],
    'mode': VALIDATION_MODE}``. Blocking mode yields 1 fold; walk-forward yields
    ``WF_N_SPLITS`` folds (rolling or expanding).
    """
    cache = TRIAL_PREP_CACHE.get(cache_key)
    if cache is not None:
        return cache

    cv = build_validation_splitter()
    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(cv.split(x), start=1):
        if len(train_indices) == 0 or len(val_indices) == 0:
            continue
        # leakage sanity check
        if int(train_indices.max()) + WF_GAP > int(val_indices.min()):
            print(
                f"[WARN] fold={fold_idx} train_max+gap={int(train_indices.max())+WF_GAP} > "
                f"val_min={int(val_indices.min())} (possible leakage)"
            )

        y_train = y[train_indices]
        y_train_labels = np.argmax(y_train, axis=1)
        unique, counts = np.unique(y_train_labels, return_counts=True)
        counts = (1 / counts) * len(y_train_labels)
        class_weights = dict(zip(unique.tolist(), counts.tolist()))

        val_weights = np.ones(len(val_indices))
        index1 = (val_weights.shape[0] // 3)
        index2 = index1 * 2
        val_weights[:index1] = 0.2
        val_weights[index1:index2] = 0.3
        val_weights[index2:] = 0.5

        folds.append({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'class_weights': class_weights,
            'val_weights': val_weights,
        })
        print(
            f"[CV] {VALIDATION_MODE} fold={fold_idx}/{cv.get_n_splits(x) if hasattr(cv,'get_n_splits') else len(folds)} "
            f"train=[{int(train_indices.min())}:{int(train_indices.max())+1}] "
            f"val=[{int(val_indices.min())}:{int(val_indices.max())+1}] "
            f"(train_n={len(train_indices)}, val_n={len(val_indices)})"
        )

    if not folds:
        raise ValueError(f"No valid split for cache_key={cache_key}")

    cache = {
        'folds': folds,
        'mode': VALIDATION_MODE,
        # back-compat fields (first fold) for any external consumer
        'train_indices': folds[0]['train_indices'],
        'val_indices': folds[0]['val_indices'],
        'class_weights': folds[0]['class_weights'],
        'val_weights': folds[0]['val_weights'],
    }
    TRIAL_PREP_CACHE[cache_key] = cache
    return cache

if ENABLE_KERAS_PICKLE_HOTFIX:
    from tensorflow.python.keras.layers import deserialize, serialize
    from tensorflow.python.keras.saving import saving_utils

    def unpack(model, training_config, weights):
        """Rehydrate serialized data into a usable Keras model."""
        restored_model = deserialize(model, custom_objects={
            'TCN': TCN,
        })
        if training_config is not None:
            restored_model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config
                )
            )
        restored_model.set_weights(weights)
        return restored_model

    def make_keras_picklable():
        """Inject `__reduce__` into Keras Model so it becomes pickle-serializable."""

        def __reduce__(self):
            model_metadata = saving_utils.model_metadata(self)
            training_config = model_metadata.get("training_config", None)
            model = serialize(self)
            weights = self.get_weights()
            return (unpack, (model, training_config, weights))

        cls = Model
        cls.__reduce__ = __reduce__

    make_keras_picklable()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
if USE_TFA_OPTIMIZER and tfa is not None:
    from tensorflow_addons.metrics import FBetaScore

class FloodingModel(keras.Model):
    """Custom Keras model that applies the Flooding training strategy.

    Purpose:
        Override `train_step` to map the raw loss to flooding loss and curb overfitting.

    Args:
        Same constructor arguments as `keras.Model` (built via the Functional API).

    Attributes:
        flooding_b: flooding level (float, default 0.10).

    Side effects:
        Changes the per-batch training loss computation.
    """
    flooding_b = 0.10       
    def train_step(self, data):
        """Run a single training step and apply flooding loss.

        Purpose:
            Replace the default `Model.train_step` and rewrite loss computation per batch.

        Args:
            data: Keras batch input, either `(x, y)` or `(x, y, sample_weight)`.

        Returns:
            dict mapping metric name -> current metric value.

        Side effects:
            Updates model weights and metric state.
        """
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses
            )
            if self.flooding_b > 0:
                loss = (tf.math.abs(loss - self.flooding_b) + self.flooding_b)      
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return { m.name: m.result() for m in self.metrics }


class DynamicFloodingCallback(tf.keras.callbacks.Callback):
    """Callback that dynamically adjusts `flooding_b` based on validation performance.

    Purpose:
        After each epoch, raise or lower the flooding level depending on whether
        the monitored metric improved.

    Args:
        monitor: name of the monitored metric (default `val_recall`).
        min_b / max_b: lower and upper bounds for flooding_b.
        step_up / step_down: adjustment step size.
        patience: epochs of no improvement tolerated.
        min_delta: minimum improvement magnitude to count as progress.
        verbose: whether to print adjustment info.

    Attributes:
        best, wait, and the control state described above.

    Side effects:
        Mutates `self.model.flooding_b` directly.
    """
    def __init__(
        self,
        monitor='val_recall',
        min_b=0.02,
        max_b=0.20,
        step_up=0.01,
        step_down=0.005,
        patience=4,
        min_delta=1e-4,
        verbose=1
    ):
        super().__init__()
        self.monitor = monitor
        self.min_b = float(min_b)
        self.max_b = float(max_b)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.verbose = int(verbose)
        self.best = -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        current_b = float(getattr(self.model, 'flooding_b', 0.10))
        improved = current > (self.best + self.min_delta)

        if improved:
            self.best = current
            self.wait = 0
            new_b = max(self.min_b, current_b - self.step_down)
        else:
            self.wait += 1
            new_b = current_b
            if self.wait >= self.patience:
                new_b = min(self.max_b, current_b + self.step_up)
                self.wait = 0

        setattr(self.model, 'flooding_b', float(new_b))
        logs['flooding_b'] = float(new_b)

        if self.verbose:
            print(f"[DynamicFlooding] epoch={epoch + 1}, {self.monitor}={current:.6f}, flooding_b={new_b:.4f}")


class TrialTimeoutCallback(tf.keras.callbacks.Callback):
    """Callback that enforces a per-trial timeout."""
    def __init__(self, trial_id, timeout_seconds, trial_started_at, verbose=1):
        super().__init__()
        self.trial_id = trial_id
        self.timeout_seconds = int(timeout_seconds)
        self.trial_started_at = float(trial_started_at)
        self.verbose = int(verbose)
        self.timed_out = False

    def _check_timeout(self):
        if self.timeout_seconds <= 0 or self.timed_out:
            return
        elapsed = time.time() - self.trial_started_at
        if elapsed >= self.timeout_seconds:
            self.timed_out = True
            self.model.stop_training = True
            if self.verbose:
                print(
                    f"[TIMEOUT] trial={self.trial_id} reached {elapsed:.1f}s "
                    f"(limit={self.timeout_seconds}s). Stop and skip this trial."
                )

    def on_train_batch_end(self, batch, logs=None):
        self._check_timeout()

    def on_epoch_end(self, epoch, logs=None):
        self._check_timeout()

class CausalMask(layers.Layer):
    """Produce a lower-triangular causal mask (past only).

    Purpose:
        Feed as `attention_mask` to MultiHeadAttention so it cannot see future timesteps.

    Args:
        None.

    Attributes:
        None trainable.

    Side effects:
        None.
    """
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return tf.linalg.band_part(
            tf.ones((1, seq_len, seq_len), dtype=tf.float32), -1, 0
        )
    
    def get_config(self):
        return super().get_config()

def add_causal_mha_block(x, num_heads, key_dim, dropout_rate, name_prefix):
    """Build a causal multi-head attention block.

    Args:
        x: input tensor with shape (batch, time, features).
        num_heads: number of attention heads.
        key_dim: key dimensionality per head.
        dropout_rate: dropout applied to the attention output.
        name_prefix: layer name prefix.

    Returns:
        Output tensor after MHA + residual + LayerNorm.

    Side effects:
        None.
    """
    causal_mask = CausalMask(name=f"{name_prefix}_causal_mask")(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=f"{name_prefix}_mha"
    )(x, x, attention_mask=causal_mask)
    attn = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(attn)
    x = layers.Add(name=f"{name_prefix}_add")([x, attn])
    x = LayerNormalization(name=f"{name_prefix}_ln")(x)
    return x

# Define model with hp (hyperparameter) turnable
class HyperTCN(kt.HyperModel):
    """HyperModel used by Keras Tuner.

    Purpose:
        Define the searchable Attention + FCN network plus its compile settings.

    Args:
        name: HyperModel name.
        tunable: whether the tuner may adjust it.
        input_shape: model input shape.
        strategy: distribution strategy (kept for compatibility).

    Attributes:
        input_shape, strategy。

    Side effects:
        `build()` builds and compiles a new model from hp.
    """
    def __init__(self, name=None, tunable=True, input_shape=None, strategy=None):
        self.input_shape = input_shape
        self.strategy = strategy
        super().__init__(name, tunable)

    def build(self, hp):     
        """Build and compile a model from a hyperparameter combination."""
        inputs = keras.Input(shape=self.input_shape, name="inputs")

        # Positional encoding: parameter-free sinusoidal encoding in place of PositionEmbedding
        x = SinusoidalPositionalEncoding(name='pos_enc')(inputs)

        attn_layers = hp.Int("attn_layers", min_value=1, max_value=3, step=1)

        attn_heads_1 = hp.Int("attn_heads_1", min_value=2, max_value=3, step=1)
        attn_key_dim_1 = hp.Int("attn_key_dim_1", min_value=16, max_value=32, step=16)
        attn_dropout_1 = hp.Float("attn_dropout_1", min_value=0.0, max_value=0.2, step=0.1)

        attn_heads_2 = hp.Int("attn_heads_2", min_value=2, max_value=3, step=1)
        attn_key_dim_2 = hp.Int("attn_key_dim_2", min_value=16, max_value=32, step=16)
        attn_dropout_2 = hp.Float("attn_dropout_2", min_value=0.0, max_value=0.2, step=0.1)

        attn_heads_3 = hp.Int("attn_heads_3", min_value=2, max_value=3, step=1)
        attn_key_dim_3 = hp.Int("attn_key_dim_3", min_value=16, max_value=32, step=16)
        attn_dropout_3 = hp.Float("attn_dropout_3", min_value=0.0, max_value=0.2, step=0.1)

        attn_configs = [
            (attn_heads_1, attn_key_dim_1, attn_dropout_1),
            (attn_heads_2, attn_key_dim_2, attn_dropout_2),
            (attn_heads_3, attn_key_dim_3, attn_dropout_3),
        ]
        for layer_index in range(attn_layers):
            num_heads, key_dim, dropout_rate = attn_configs[layer_index]
            x = add_causal_mha_block(
                x,
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                name_prefix=f"attn{layer_index + 1}"
            )

        x = layers.GlobalAveragePooling1D(name="attn_pool")(x)
        x = LayerNormalization(name="attn_pool_ln")(x)

        x = Dense(hp.Int("Dense_units_1", min_value=64, max_value=192, step=32), 
            kernel_initializer=hp.Choice("Dense_kernel_1", ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"], default='glorot_normal'), 
            #activation="relu"
            activation=hp.Choice("activation_2", ["relu", "elu", "selu", "tanh", "swish"], default="relu")
        )(x)
        x = LayerNormalization()(x)

        logits = Dense(
            2,
            kernel_initializer=hp.Choice("Dense_kernel_2", ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"], default='glorot_normal')
        )(x)

        # Temperature: controls softmax smoothness
        logits = layers.Lambda(
            lambda t: t / hp.Float("temp", min_value=0.3, max_value=1, step=0.1),
            name="temperature"
        )(logits)
        outputs = layers.Activation("softmax", name="softmax")(logits)
        
        if USE_TFA_OPTIMIZER:
            radam = tfa.optimizers.RectifiedAdam(
                learning_rate=0.001,
                beta_1=0.95,
                total_steps=1200,
                epsilon=1e-07,
                amsgrad=False,
                sma_threshold=5.0,
                warmup_proportion=0.10,
                min_lr=1e-5,
                name='RectifiedAdam'
            )
            opt = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        if USE_TFA_OPTIMIZER and tfa is not None:
            fbeta_metric = FBetaScore(num_classes=2, beta=1.0, threshold=0.5, average='weighted')
        else:
            fbeta_metric = tf.keras.metrics.F1Score(average='weighted', name='f_beta_score')

        # Wrap in the custom FloodingModel so our train_step takes effect
        model = FloodingModel(inputs, outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            run_eagerly=USE_EAGER_TRAINING,
            metrics=[
                tf.keras.metrics.Recall(class_id=1, name='recall'),
                fbeta_metric,
                tf.keras.metrics.AUC(curve='ROC', name='auc'),
                tf.keras.metrics.AUC(curve='PR', name='pr_auc')
            ]
        )
        return model

class TunerCV(kt.engine.tuner.Tuner):
    """Custom Tuner that scores trials with time-series splits.

    Purpose:
        Override `run_trial` to run blocking CV inside each trial and report the average metric.

    Args:
        Compatible with `kt.engine.tuner.Tuner`.

    Attributes:
        Inherits from the Keras Tuner Tuner.

    Side effects:
        Runs training, updates oracle trial metrics, and clears the Keras session.
    """
    def run_trial(self, trial, x=None, y=None, batch_size=40, epochs=1, callbacks=[], windowed_data=None, *args, **kwargs):
        """Run one trial and report the time-series validation result.

        Purpose:
            Perform blocking CV inside the trial, aggregate validation metrics,
            and update the oracle.

        Args:
            trial: current Keras Tuner trial object.
            x: training feature array (used when `windowed_data` is not supplied).
            y: training label array (used when `windowed_data` is not supplied).
            batch_size: training batch size.
            epochs: number of training epochs.
            callbacks: extra callback list (kept for compatibility).
            windowed_data: optional dict mapping lookback_window -> (X_train, y_train_onehot).
            *args, **kwargs: reserved for the parent class / caller.

        Returns:
            None (results are reported via `self.oracle.update_trial(...)`).

        Side effects:
            Trains the model, runs inference, updates trial metrics, and clears the Keras session.
        """
        if windowed_data is not None:
            lookback_window = trial.hyperparameters.Choice(
                'lookback_window',
                values=LOOKBACK_WINDOW_CHOICES
            )
            if lookback_window not in windowed_data:
                available_windows = sorted(windowed_data.keys())
                lookback_window = available_windows[0]
                print(f"[WARN] lookback_window not found in prepared data. Fallback to {lookback_window}.")
            x, y = windowed_data[lookback_window]
            print(f"[Trial {trial.trial_id}] lookback_window={lookback_window}, train_samples={x.shape[0]}")

        if x is None or y is None:
            raise ValueError("run_trial requires either (x, y) or windowed_data.")

        trial_started_at = time.time()
        trial_timed_out = False
        print(
            f"[Trial {trial.trial_id}] start: samples={x.shape[0]}, timesteps={x.shape[1]}, "
            f"features={x.shape[2]}, epochs={epochs}, batch_size={batch_size}, "
            f"timeout={TRIAL_TIMEOUT_SECONDS}s"
        )

        val_losses = []
        val_recall_metrics = []
        val_fbeta_metrics = []
        val_auc_metrics = []
        val_pr_auc_metrics = []
        val_fbeta_scores = []
        val_recall_scores = []

        if windowed_data is not None:
            cache_prefix = f"lb_{lookback_window}"
        else:
            cache_prefix = "direct"
        cache_key = make_dataset_cache_key(cache_prefix, x, y)
        prep_cache = get_trial_prep_cache(cache_key, x, y)

        # Split training data into 3 blocks; the average classification score across their validation sets is the trial objective.
        folds = prep_cache['folds']
        n_folds = len(folds)
        for fold_idx, fold in enumerate(folds, start=1):
            train_indices = fold['train_indices']
            val_indices = fold['val_indices']
            elapsed_before_fold = time.time() - trial_started_at
            if TRIAL_TIMEOUT_SECONDS > 0 and elapsed_before_fold >= TRIAL_TIMEOUT_SECONDS:
                trial_timed_out = True
                print(
                    f"[TIMEOUT] trial={trial.trial_id} skip fold={fold_idx} "
                    f"(elapsed={elapsed_before_fold:.1f}s, limit={TRIAL_TIMEOUT_SECONDS}s)"
                )
                break

            x_train, x_val = x[train_indices], x[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            print(
                f"[Trial {trial.trial_id}] fold={fold_idx}/{n_folds} "
                f"train={x_train.shape[0]} val={x_val.shape[0]} mode={prep_cache.get('mode','blocking')}"
            )
            self.hypermodel.input_shape = (x_train.shape[1], x_train.shape[2])
            if FORCE_MODEL_BUILD_ON_CPU:
                with tf.device('/CPU:0'):
                    model = self.hypermodel.build(trial.hyperparameters)
            else:
                model = self.hypermodel.build(trial.hyperparameters)

            flooding_candidates = [0.05, 0.10, 0.20]
            try:
                trial_idx = int(trial.trial_id)
            except (TypeError, ValueError):
                trial_idx = 0
            model.flooding_b = flooding_candidates[trial_idx % len(flooding_candidates)]
            print(f"[Trial {trial.trial_id}] init flooding_b={model.flooding_b:.2f}")

            class_weights = fold['class_weights']
            val_weights = fold['val_weights']

            warmup_elapsed = 0.0
            if ENABLE_TRIAL_WARMUP:
                warmup_elapsed = warmup_one_batch(model, x_train, y_train, batch_size)
            print(f"[Trial {trial.trial_id}] prefit warmup={warmup_elapsed:.3f}s")

            timeout_callback = TrialTimeoutCallback(
                trial_id=trial.trial_id,
                timeout_seconds=TRIAL_TIMEOUT_SECONDS,
                trial_started_at=trial_started_at,
                verbose=1
            )

            model.fit(
                x_train, 
                y_train,
                batch_size=batch_size, 
                epochs=epochs, 
                verbose=FIT_VERBOSE,
                shuffle=False,  # never shuffle time-series data to avoid look-ahead leakage
                validation_data=(x_val, y_val, val_weights), 
                class_weight=class_weights,
                callbacks=[
                    timeout_callback,
                    DynamicFloodingCallback(
                        monitor='val_recall',
                        min_b=0.02,
                        max_b=0.20,
                        step_up=0.01,
                        step_down=0.005,
                        patience=4,
                        min_delta=1e-4,
                        verbose=1
                    ),
                    # Reduce learning rate when training plateaus
                    #ReduceLROnPlateau(monitor='val_fbeta_score', mode='max', factor=0.2, patience=10, min_delta=1E-7, verbose=1)
                    #,EarlyStopping(monitor='val_fbeta_score', mode='max', patience= 10, verbose=1)
                    ReduceLROnPlateau(monitor='val_recall', mode='max', factor=0.5, patience=25, min_delta=1e-7, verbose=1)
                    ,EarlyStopping(monitor='val_recall', mode='max', patience=65, min_delta=1e-7, restore_best_weights=True, verbose=1)
                ]
            )

            if timeout_callback.timed_out:
                trial_timed_out = True
                K.clear_session()
                gc.collect()
                break

            eval_results = model.evaluate(x_val, y_val, verbose=0)
            # metrics_names and evaluate() return in the same order, so we can zip them into a dict for convenience
            eval_by_name = dict(zip(model.metrics_names, eval_results))
            val_loss = eval_by_name.get('loss', np.nan)
            val_recall_metric = eval_by_name.get('recall', np.nan)
            val_fbeta_metric = eval_by_name.get('f_beta_score', np.nan)
            val_auc_metric = eval_by_name.get('auc', np.nan)
            val_pr_auc_metric = eval_by_name.get('pr_auc', np.nan)

            y_pred = model.predict(x_val, batch_size=batch_size, verbose=0)
            y_pred_labels = (y_pred[:, 1] >= 0.5).astype(int)
            val_fbeta = fbeta_score(y_val[:, 1], y_pred_labels, zero_division=1, beta=1.0)
            val_recall = recall_score(y_val[:, 1], y_pred_labels)
            val_losses.append(val_loss)
            val_recall_metrics.append(val_recall_metric)
            val_fbeta_metrics.append(val_fbeta_metric)
            val_auc_metrics.append(val_auc_metric)
            val_pr_auc_metrics.append(val_pr_auc_metric)
            val_fbeta_scores.append(val_fbeta)
            val_recall_scores.append(val_recall)
            print(f"F-beta score: {val_fbeta:.2f}")
            print(f"Recall score: {val_recall:.2f}")
            K.clear_session()
            gc.collect()

        print(f"[Trial {trial.trial_id}] done in {(time.time() - trial_started_at):.1f}s")

        if trial_timed_out or len(val_recall_scores) == 0:
            self.oracle.update_trial(
                trial.trial_id,
                {
                    'val_loss': 1e9,
                    'val_recall_metric': 0.0,
                    'val_fbeta_metric': 0.0,
                    'val_auc_metric': 0.0,
                    'val_pr_auc_metric': 0.0,
                    'val_fbeta_score': 0.0,
                    'val_recall_score': 0.0
                }
            )
            print(f"[Trial {trial.trial_id}] marked as timeout/invalid and skipped.")
            return

        self.oracle.update_trial(
            trial.trial_id,
            {
                # Report the aggregated trial score to Bayesian Optimization
                'val_loss': np.mean(val_losses),
                'val_recall_metric': np.mean(val_recall_metrics),
                'val_fbeta_metric': np.mean(val_fbeta_metrics),
                'val_auc_metric': np.mean(val_auc_metrics),
                'val_pr_auc_metric': np.mean(val_pr_auc_metrics),
                'val_fbeta_score': np.mean(val_fbeta_scores),
                'val_recall_score': np.mean(val_recall_scores)
            }
        )
        # fname = os.path.join(self.get_trial_dir(trial.trial_id), "model.pickle")
        # with tf.io.gfile.GFile(fname, "wb") as f:
        #     pickle.dump(model, f)
        # oracle.update_trial - used by worker to report the status of a trial, keras oracle.up_trial(trial_id, metrics, step=0)
import traceback, sys, subprocess
from glob import glob
from sklearn.preprocessing import RobustScaler, PowerTransformer


# Trade-CSV columns with heavy tails (pct_change / abs(net / MA20) etc. can explode);
# apply signed log1p before sanitize so Yeo-Johnson / quantile transforms are not
# biased by extreme tails.
TRADE_HEAVY_TAIL_COLS = (
    'foreign_cap_ratio', 'invst_cap_ratio', 'ins_nbd', 'Force_nbd', 'smr'
)


def _prepare_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply signed log1p to heavy-tailed columns of the trade CSV; leave others untouched."""
    out = df.copy()
    for c in TRADE_HEAVY_TAIL_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors='coerce').astype(np.float64)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s))
    return out


def _detect_non_zero_date(df: pd.DataFrame, ratio_threshold: float = 0.0):
    """Return the first date whose fraction of non-zero columns reaches ratio_threshold.
    ratio_threshold=0 preserves the legacy behaviour (any column non-zero)."""
    feat = df.iloc[:, :-4]
    if ratio_threshold <= 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    else:
        idx = feat.index[(feat != 0).mean(axis=1) >= ratio_threshold]
    if len(idx) == 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    return idx[0] if len(idx) else feat.index[0]


def _expand_sentiment_features(df: pd.DataFrame, stock_id) -> pd.DataFrame:
    """Expand a sentiment CSV (three 0-100 score columns: US / TW / ticker) into derived features.

    Input df: index=date; columns include 'US_sentiment_score',
    'TW_sentiment_score', str(stock_id); the last 4 columns are the y_10..y_60 labels.
    Returns concat(derived features, original 4 label columns). The shared
    downstream preprocessing (corr filter / scaler / sanitize) runs on this
    function's output columns.
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    us = 'US_sentiment_score'
    tw = 'TW_sentiment_score'
    stock_col = str(stock_id)
    if stock_col not in df.columns:
        candidates = [c for c in df.columns[:-4] if c not in (us, tw)]
        if not candidates:
            return df
        stock_col = candidates[0]
    if us not in df.columns or tw not in df.columns:
        return df

    us_s = pd.to_numeric(df[us], errors='coerce')
    tw_s = pd.to_numeric(df[tw], errors='coerce')
    st_s = pd.to_numeric(df[stock_col], errors='coerce')

    out = pd.DataFrame(index=df.index)
    out['us_score']    = us_s / 100.0
    out['tw_score']    = tw_s / 100.0
    out['stock_score'] = st_s / 100.0
    for name, src in (('us', us_s), ('tw', tw_s), ('stock', st_s)):
        out[f'{name}_d5']  = src.diff(5)  / 100.0
        out[f'{name}_d20'] = src.diff(20) / 100.0
    out['stock_ewm10']    = st_s.ewm(span=10, adjust=False).mean() / 100.0
    out['tw_ewm20']       = tw_s.ewm(span=20, adjust=False).mean() / 100.0
    out['stock_minus_tw'] = (st_s - tw_s) / 100.0
    out['tw_minus_us']    = (tw_s - us_s) / 100.0
    out['stock_gt_tw']    = (st_s > tw_s).astype(np.float32)
    for name, src, win in (('stock', st_s, 60), ('tw', tw_s, 60), ('us', us_s, 120)):
        m = src.rolling(win, min_periods=max(win // 2, 1)).mean()
        s = src.rolling(win, min_periods=max(win // 2, 1)).std().replace(0, np.nan)
        out[f'{name}_z{win}'] = (src - m) / s

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return pd.concat([out, labels], axis=1)


# macro column groups:
#   - Stationary / bounded: keep level.
#   - Non-stationary prices: convert to log-return / rolling z-score.
#   - Option volumes and futures net: first ffill(0->NaN), then apply the matching transform.
MACRO_STATIONARY_COLS = ('Price_rate_3m', 'Price_rate_10y', 'Price_FX', 'Price_VIX')
MACRO_PRICE_LEVEL_COLS = (
    'Price_oil', 'Price_gold', 'Price_copper',
    'Price_S&P500', 'Price_Nasdaq', 'Price_SOX',
    'Price_CRB', 'Price_BDI',
)
MACRO_VOL_COLS = ('Price_TX03C', 'Price_TX03P')
MACRO_SIGNED_LEVEL_COLS = ('Price_TX03F',)


def _expand_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the macro CSV to near-stationary features: non-stationary prices ->
    log-return / z-score; bounded columns keep their level; option volume ->
    log1p + diff20; futures net -> signed log1p.

    In the macro CSV, 0 means missing (holiday / gap), so we do replace(0->NaN).ffill()
    before transforming.
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 1) Stationary / bounded columns: keep the level
    for c in MACRO_STATIONARY_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            # 0 in rate / FX / VIX is usually also missing (especially rates) -> ffill
            if c in ('Price_rate_3m', 'Price_rate_10y', 'Price_FX'):
                s = s.replace(0, np.nan).ffill()
            out[c] = s

    # 2) term spread (if both rate_3m and rate_10y are present)
    if 'Price_rate_3m' in out.columns and 'Price_rate_10y' in out.columns:
        out['term_spread'] = out['Price_rate_10y'] - out['Price_rate_3m']

    # 3) VIX also adds log(vix) and its z-score
    if 'Price_VIX' in out.columns:
        vix = out['Price_VIX'].replace(0, np.nan).ffill()
        out['log_vix'] = np.log(vix)
        m = vix.rolling(60, min_periods=30).mean()
        sd = vix.rolling(60, min_periods=30).std().replace(0, np.nan)
        out['vix_z60'] = (vix - m) / sd

    # 4) Non-stationary prices -> log-return + rolling z-score
    for c in MACRO_PRICE_LEVEL_COLS:
        if c not in df.columns:
            continue
        p = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        out[f'{c}_logret1']  = np.log(p / p.shift(1))
        out[f'{c}_logret20'] = np.log(p / p.shift(20))
        m = p.rolling(60, min_periods=30).mean()
        sd = p.rolling(60, min_periods=30).std().replace(0, np.nan)
        out[f'{c}_z60'] = (p - m) / sd

    # 5) TAIEX option volumes -> log1p + diff20; also add put/call log-ratio
    vol_logs = {}
    for c in MACRO_VOL_COLS:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        lv = np.log1p(v)
        vol_logs[c] = lv
        out[f'{c}_log1p']  = lv
        out[f'{c}_diff20'] = lv - lv.shift(20)
    if 'Price_TX03P' in vol_logs and 'Price_TX03C' in vol_logs:
        out['tx03_pc_logratio'] = vol_logs['Price_TX03P'] - vol_logs['Price_TX03C']

    # 6) Futures net (can be positive or negative) -> signed log1p + diff
    for c in MACRO_SIGNED_LEVEL_COLS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        sl = np.sign(s) * np.log1p(np.abs(s))
        out[f'{c}_slog1p']  = sl
        out[f'{c}_diff5']   = sl - sl.shift(5)
        out[f'{c}_diff20']  = sl - sl.shift(20)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


# ===================================================================
# tech_trend / fundamental / moment expanders (pass-through friendly version).
# Design principles:
#   1. All outputs are deterministic and hard-clipped; they do not depend on
#      training-window quantiles, so FEATURE_PREPROCESS=off can feed them
#      straight into the model.
#   2. Only obviously non-stationary or heavy-tailed columns (raw OHLCV,
#      growth rates, PEG, cci, acc_*, vpt) are transformed. Columns that are
#      already stationary / bounded (sma / hullma / bias / alpha / RSI / K / D
#      etc.) are left untouched to minimize old-vs-new model divergence.
#   3. Output values sit roughly within +/-3 thanks to signed log1p / fixed
#      clips that damp extreme tails.
# ===================================================================


def _expand_tech_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """tech_trend CSV: convert open/high/low/close/volume into scale-free features;
    osc becomes a ratio (osc / close); the remaining columns
    (sma_*/hullma_*/mmi_*/aroon_osc/bb/bias/alpha) are left as-is.
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]

    out = pd.DataFrame(index=df.index)

    # 1) Columns that are already stationary / bounded: keep as-is
    passthrough_cols = [
        'sma_5', 'sma_10', 'sma_20', 'sma_60', 'sma_120',
        'hullma_20', 'hullma_60', 'hullma_120',
        'mmi_5', 'mmi_10', 'mmi_20',
        'aroon_osc', 'bb', 'bias', 'alpha',
    ]
    for c in passthrough_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors='coerce')

    # 2) osc: price-scaled -> convert to a close-scaled ratio
    if 'osc' in df.columns and 'close' in df.columns:
        close_raw = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        osc_raw = pd.to_numeric(df['osc'], errors='coerce')
        out['osc_pct'] = (osc_raw / close_raw).clip(-0.3, 0.3)

    # 3) close -> log-return 1/5/20 (drop absolute price)
    if 'close' in df.columns:
        close = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        log_close = np.log(close)
        out['ret_1']  = (log_close - log_close.shift(1)).clip(-0.3, 0.3)
        out['ret_5']  = (log_close - log_close.shift(5)).clip(-0.5, 0.5)
        out['ret_20'] = (log_close - log_close.shift(20)).clip(-0.8, 0.8)

    # 4) (high - low) / close: intraday volatility; gap = log(open / prev_close)
    if {'high', 'low', 'close'}.issubset(df.columns):
        h = pd.to_numeric(df['high'], errors='coerce')
        l = pd.to_numeric(df['low'], errors='coerce')
        c = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        out['hl_range'] = ((h - l) / c).clip(0.0, 0.2)
    if {'open', 'close'}.issubset(df.columns):
        o = pd.to_numeric(df['open'], errors='coerce').replace(0, np.nan).ffill()
        c = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        out['gap'] = (np.log(o) - np.log(c.shift(1))).clip(-0.15, 0.15)

    # 5) volume: convert to a ratio against the 20-day mean
    if 'volume' in df.columns:
        v = pd.to_numeric(df['volume'], errors='coerce').clip(lower=0)
        v_ma = v.rolling(20, min_periods=5).mean().replace(0, np.nan)
        out['vol_ratio20'] = ((v / v_ma) - 1.0).clip(-5.0, 5.0)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


# fundamental column groups (see Feature_Cmoney_update.py):
#   - PE_trailing / PBR: rolling 3Y river-level percentile, already in 0..1.
#   - DY: annualized dividend yield %, typical range 0..15.
#   - Gross: gross margin %, typical range 0..100.
#   - PEG: PE / growth; can be positive or negative and blow up.
#   - R_* / E_* / Op_* / Gross_qoq / EPS_qoq: growth rate %, spikes on small base periods.
FUNDAMENTAL_GROWTH_COLS = (
    'R_mom', 'R_yoy', 'R_acc_yoy',
    'E_qoq', 'E_yoy', 'E_acc_yoy',
    'Op_qoq', 'Op_yoy', 'Op_acc_yoy',
    'Gross_qoq', 'EPS_qoq',
)


def _expand_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """fundamental CSV: recenter bounded columns near +/-1; apply clip +
    signed log1p to PEG and growth-rate columns to damp small-base blowups.
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 1) river-level percentile (0..1) -> recenter to +/-1
    for c in ('PE_trailing', 'PBR'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 1.0)
            out[c] = (s - 0.5) * 2.0

    # 2) DY: dividend yield %, clipped to [0, 20] then /10 -> roughly [0, 2]
    if 'DY' in df.columns:
        s = pd.to_numeric(df['DY'], errors='coerce').clip(0.0, 20.0)
        out['DY'] = s / 10.0

    # 3) Gross: gross margin %, clipped to [-20, 100] then /100 -> roughly [-0.2, 1.0]
    if 'Gross' in df.columns:
        s = pd.to_numeric(df['Gross'], errors='coerce').clip(-20.0, 100.0)
        out['Gross'] = s / 100.0

    # 4) Growth rate (%): hard-clip to +/-300, then sign * log1p(|x|/100)
    #    x=100% -> 0.69; x=300% -> 1.39; sign preserved
    for c in FUNDAMENTAL_GROWTH_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(-300.0, 300.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s) / 100.0)

    # 5) PEG: ratio (can be negative or large); clip to +/-10 then sign * log1p(|x|)
    if 'PEG' in df.columns:
        s = pd.to_numeric(df['PEG'], errors='coerce').clip(-10.0, 10.0)
        out['PEG'] = np.sign(s) * np.log1p(np.abs(s))

    # 6) CMDTY (if present): commodity-index level -> 20-day log-return
    if 'CMDTY' in df.columns:
        p = pd.to_numeric(df['CMDTY'], errors='coerce').replace(0, np.nan).ffill()
        out['CMDTY_logret20'] = (np.log(p) - np.log(p.shift(20))).clip(-0.5, 0.5)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


def _expand_moment_features(df: pd.DataFrame) -> pd.DataFrame:
    """moment CSV: recenter RSI/K/D/ADX/WR to +/-1; CCI clipped to +/-300 then /100;
    acc_* clipped then log; vpt replaced with the rolling z-score of its diff;
    beta clipped to +/-3.
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 1) 0..100 bounded → (x-50)/50 → [-1, 1]
    for c in ('rsi', 'k', 'd', 'adx'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 100.0)
            out[c] = (s - 50.0) / 50.0

    # 2) wr is originally in [-100, 0] -> (x+50)/50 -> [-1, 1]
    if 'wr' in df.columns:
        s = pd.to_numeric(df['wr'], errors='coerce').clip(-100.0, 0.0)
        out['wr'] = (s + 50.0) / 50.0

    # 3) cci: heavy-tailed around +/-200 to +/-400; clip to +/-300 then /100 -> [-3, 3]
    if 'cci' in df.columns:
        s = pd.to_numeric(df['cci'], errors='coerce').clip(-300.0, 300.0)
        out['cci'] = s / 100.0

    # 4) acc_*: price ratios centered near 1.0; clip to [0.5, 2.0] then log -> [-0.69, 0.69]
    for c in ('acc_5', 'acc_10', 'acc_20', 'acc_60', 'acc_120'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.5, 2.0)
            out[c] = np.log(s)

    # 5) vpt: cumulative volume, non-stationary -> diff then 60-day rolling z-score
    if 'vpt' in df.columns:
        s = pd.to_numeric(df['vpt'], errors='coerce')
        d = s.diff()
        m = d.rolling(60, min_periods=20).mean()
        sd = d.rolling(60, min_periods=20).std().replace(0, np.nan)
        out['vpt_z60'] = ((d - m) / sd).clip(-3.0, 3.0)

    # 6) beta: rolling OLS slope; clip extreme values
    if 'beta' in df.columns:
        s = pd.to_numeric(df['beta'], errors='coerce').clip(-3.0, 3.0)
        out['beta'] = s

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


def run_isolated_stock_model_jobs(stock_ids, model_types):
    """Run each (stock_id, model_type) in its own subprocess to avoid long-run memory build-up."""
    script_path = os.path.abspath(__file__)
    total_jobs = len(stock_ids) * len(model_types)
    failed_jobs = []
    current = 0

    for stock_id in stock_ids:
        for model_type in model_types:
            current += 1
            child_env = os.environ.copy()
            child_env['STOCK_IDS'] = stock_id
            child_env['MODEL_TYPES'] = model_type
            child_env['ISOLATED_CHILD_RUN'] = '1'
            print(f"[ISOLATE] ({current}/{total_jobs}) start child job: stock={stock_id}, model={model_type}")
            result = subprocess.run([sys.executable, script_path], env=child_env)
            if result.returncode != 0:
                print(
                    f"[ISOLATE] child failed (attempt 1): "
                    f"stock={stock_id}, model={model_type}, rc={result.returncode}. "
                    "Retry with safe fallback settings."
                )

                safe_env = child_env.copy()
                safe_env['TRAIN_MODE'] = 'safe'
                safe_env['ENABLE_MIXED_PRECISION'] = '0'
                safe_env['ENABLE_XLA'] = '0'
                safe_env['ENABLE_TF32'] = '1'
                safe_env['FIT_VERBOSE'] = safe_env.get('FIT_VERBOSE', '2')

                retry_result = subprocess.run([sys.executable, script_path], env=safe_env)
                if retry_result.returncode != 0:
                    failed_jobs.append((stock_id, model_type, retry_result.returncode))
                    print(
                        f"[ISOLATE] child failed (attempt 2): "
                        f"stock={stock_id}, model={model_type}, rc={retry_result.returncode}"
                    )
                else:
                    print(
                        f"[ISOLATE] child recovered on attempt 2 (safe mode): "
                        f"stock={stock_id}, model={model_type}"
                    )
            else:
                print(f"[ISOLATE] child done: stock={stock_id}, model={model_type}")

    if failed_jobs:
        raise RuntimeError(f"Isolated jobs failed: {failed_jobs}")

# Track (stock_id, model_type) pairs that failed during this batch
error_list = []
#train_progress = symbols.reindex(sorted([os.path.basename(i)[6:-4] for i in glob('features/macro_*.csv')])).dropna().index.to_list() # ticker sorted

# Main training entry: extend the stock_id / model_type list as needed
stock_ids = [x.strip() for x in os.getenv('STOCK_IDS', 'AXP,AMGN,AMZN,AAPL,BA,CAT,CSCO,CVX,GS,HD,HON,IBM,JNJ,KO,JPM,MCD,MMM,MRK,MSFT,NVDA,NKE,PG,TRV,UNH,CRM,VZ,V,WMT,DIS,SHW').split(',') if x.strip()]
model_types = [
    x.strip() for x in os.getenv(
        'MODEL_TYPES',
        'fundamental,moment,tech_trend,macro'
    ).split(',') if x.strip()
]

if ISOLATE_STOCK_MODEL_RUNS and (not ISOLATED_CHILD_RUN) and (len(stock_ids) * len(model_types) > 1):
    # Feature-preprocessing master switch: parent asks (or reads env) once,
    # writes back into os.environ so every child inherits the choice
    DO_FEATURE_PREPROCESS = _resolve_feature_preprocess()
    os.environ['FEATURE_PREPROCESS'] = '1' if DO_FEATURE_PREPROCESS else '0'
    print(f"[PREPROCESS] feature preprocessing = {'ON' if DO_FEATURE_PREPROCESS else 'OFF (pass-through)'} (will propagate to child jobs)")
    print(
        f"[CV] VALIDATION_MODE={VALIDATION_MODE} WF_N_SPLITS={WF_N_SPLITS} "
        f"WF_VAL_RATIO={WF_VAL_RATIO} WF_VAL_YEARS={WF_VAL_YEARS} "
        f"WF_VAL_SAMPLES={WF_VAL_SAMPLES} WF_GAP={WF_GAP}"
    )
    run_isolated_stock_model_jobs(stock_ids, model_types)
    print("[ISOLATE] all child jobs finished")
    sys.exit(0)

# Single-process mode (or already-a-child): decide right here
DO_FEATURE_PREPROCESS = _resolve_feature_preprocess()
os.environ['FEATURE_PREPROCESS'] = '1' if DO_FEATURE_PREPROCESS else '0'
print(f"[PREPROCESS] feature preprocessing = {'ON' if DO_FEATURE_PREPROCESS else 'OFF (pass-through)'}")
print(
    f"[CV] VALIDATION_MODE={VALIDATION_MODE} WF_N_SPLITS={WF_N_SPLITS} "
    f"WF_VAL_RATIO={WF_VAL_RATIO} WF_VAL_YEARS={WF_VAL_YEARS} "
    f"WF_VAL_SAMPLES={WF_VAL_SAMPLES} WF_GAP={WF_GAP}"
)

for stock_id in stock_ids: # symbols.index.tolist():,,'2317','2308'
    for model_type in model_types:  #,'tech_trend' ,'macro' 'fundamental','tech_trend','moment','sentiment'
        try:
            TRIAL_PREP_CACHE.clear()
            print(f"[CACHE] reset trial prep cache for {stock_id} {model_type}")
            # ===================== 1) Load data =====================
            print(f"[TRAIN] {stock_id} {model_type}")
            X_y_all = pd.read_csv(platform_path(f"{ATT_DATA_DIR}/{model_type}_{stock_id}.csv"), index_col=0, parse_dates=True)                             

            # sentiment aspect: raw CSV has only 3 columns (US / TW / per-stock
            # sentiment score); expand into momentum / relative-strength /
            # rolling-z features so the model does not see overly sparse or
            # constant inputs
            if model_type == 'sentiment':
                X_y_all = _expand_sentiment_features(X_y_all, stock_id)

            # trade aspect: dampen heavy-tailed columns first (signed log1p) so the downstream sanitize / scaler stays stable
            if model_type == 'trade':
                X_y_all = _prepare_trade_features(X_y_all)

            # macro aspect: price levels are highly non-stationary (recent
            # values keep exceeding the training window max); convert to
            # near-stationary log-return / rolling z / log1p features; keep rate
            # / FX / VIX levels as-is
            if model_type == 'macro':
                X_y_all = _expand_macro_features(X_y_all)

            # tech_trend aspect: raw OHLCV is non-stationary (recent prices
            # dwarf the training window); convert to scale-free features
            # log-return / (hl)/close / vol_ratio; keep sma / hullma / bias /
            # alpha as-is
            if model_type == 'tech_trend':
                X_y_all = _expand_tech_trend_features(X_y_all)

            # fundamental aspect: bounded rescale for PE / PBR / DY / Gross;
            # clip + signed log1p for growth rate & PEG to guard against
            # small-base blowups
            if model_type == 'fundamental':
                X_y_all = _expand_fundamental_features(X_y_all)

            # moment aspect: recenter RSI / K / D / ADX / WR to +/-1; hard-clip
            # CCI; log(acc_*); replace vpt with the rolling z-score of its diff;
            # clip beta
            if model_type == 'moment':
                X_y_all = _expand_moment_features(X_y_all)

            #Find the ealist vaild start date where all figures are positive
            if model_type == 'sentiment':
                # sentiment CSV has been valid since 2015-01-05; use as-is
                train_start = '2015-01-06'
            elif model_type == 'trade':
                # In the early trade window (1999-2007) many columns are 0; require ">=50% non-zero columns" instead
                non_zero_date = _detect_non_zero_date(X_y_all, ratio_threshold=0.5)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            elif model_type == 'macro':
                # In the early macro window (1994-2007) most columns are 0; require ">=90% non-zero" instead
                non_zero_date = _detect_non_zero_date(X_y_all, ratio_threshold=0.9)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            elif model_type in ('tech_trend', 'moment'):
                # tech_trend: sma_120 needs 120 days; moment: acc_120 / vpt_z60 need 60-120 days
                non_zero_date = _detect_non_zero_date(X_y_all, ratio_threshold=0.5)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            else:
                temp = X_y_all.iloc[:,:-4]
                non_zero_date = temp.index[~(temp==0).all(axis=1)][0]
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')   
            #train_start = '1994-01-05'
            n_timesteps = max(LOOKBACK_WINDOW_CHOICES)
            #train_start = '2005-01-01'
            #train_start = '2002-12-17'
            val_end = '2025-12-31'
            train_end = '2025-12-31'
            test_start = '2026-01-01'
            test_end = X_y_all.index.max().strftime('%Y-%m-%d')
            
            # ===================== 2) Feature preprocessing =====================
            if not DO_FEATURE_PREPROCESS:
                # pass-through: skip correlation filter, sanitize, and scaler
                # Still align to lookback_start and save a "transparent" bundle so the inference side auto-skips the transform
                lookback_start = X_y_all.loc[:train_start].iloc[-n_timesteps+1:].index.min()
                feature_cols = X_y_all.columns[:-4]

                # Minimal NaN / inf cleanup only to avoid training-time crashes (no scaling)
                raw_features = X_y_all.loc[:, feature_cols].apply(pd.to_numeric, errors='coerce').astype(np.float64)
                raw_features = raw_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                preprocess_bundle = {
                    'scaler': None,
                    'sanitize_stats': None,
                    'feature_columns': feature_cols.tolist(),
                    'version': 'v3_passthrough'
                }
                joblib.dump(preprocess_bundle, f"{SCALER_ROOT}/scaler_{model_type}_{stock_id}.pkl")
                X_y_all = pd.concat([raw_features, X_y_all.iloc[:, -4:]], axis=1).loc[lookback_start:]
                print(f"[PREPROCESS] skipped for {stock_id}/{model_type} (pass-through bundle saved)")
            else:
                # (a) Drop highly correlated features to reduce collinearity (estimated on train slice only)
                if model_type in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment'):
                    # Derived features intentionally keep complementary signals
                    # (short/mid/long return, multi-period growth rates, etc.); skip corr filter
                    selected_feature_cols = X_y_all.columns[:-4].tolist()
                else:
                    train_feature_slice = X_y_all.loc[:train_end].iloc[:, :-4]
                    cor_matrix = train_feature_slice.corr()
                    cor_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
                    selected_feature_cols = cor_matrix.drop(
                        [var for var in cor_matrix.columns if any(cor_matrix[var] > 0.85)],
                        axis=1
                    ).columns.tolist()
                X_y_all = X_y_all[selected_feature_cols + X_y_all.iloc[:, -4:].columns.tolist()]

                # (b) Post-expander features are mostly bounded / z-score /
                #     log-return / signed log1p, so use RobustScaler; keep
                #     PowerTransformer for any future aspects.
                if model_type in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment'):
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))
                else:
                    scaler = PowerTransformer(method='yeo-johnson')

                lookback_start = X_y_all.loc[:train_start].iloc[-n_timesteps+1:].index.min()
                feature_cols = X_y_all.columns[:-4]

                # (c) First cast raw features to finite values that scale safely
                sanitize_stats = fit_sanitize_statistics(X_y_all.loc[lookback_start:train_end, feature_cols])
                clean_features = apply_sanitize_statistics(X_y_all.loc[:, feature_cols], sanitize_stats)
                fit_features = clean_features.loc[lookback_start:train_end]

                # (d) Convert to numpy and guard against NaN / inf again
                fit_array = np.nan_to_num(
                    fit_features.to_numpy(dtype=np.float64, copy=True),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )
                all_array = np.nan_to_num(
                    clean_features.to_numpy(dtype=np.float64, copy=True),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                # (e) Fall back to RobustScaler if PowerTransformer fails due to unstable input
                try:
                    scaler.fit(fit_array)
                    X_scaled = scaler.transform(all_array)
                except ValueError as scale_error:
                    print(f"[WARN] PowerTransformer failed, fallback to RobustScaler: {scale_error}")
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))
                    scaler.fit(fit_array)
                    X_scaled = scaler.transform(all_array)

                preprocess_bundle = {
                    'scaler': scaler,
                    'sanitize_stats': sanitize_stats,
                    'feature_columns': feature_cols.tolist(),
                    'version': 'v2_train_only_sanitize'
                }
                joblib.dump(preprocess_bundle, f"{SCALER_ROOT}/scaler_{model_type}_{stock_id}.pkl")
                X_y_all = pd.concat([pd.DataFrame(X_scaled, index=X_y_all.index, columns=feature_cols), X_y_all.iloc[:, -4:]], axis=1).loc[lookback_start:]
            # ============================================='''
            
            # ===================== 3) Build supervised data ====================
            # Pre-build training data for each lookback_window so AutoML trials
            # can pick between them.
            windowed_train_data = {}
            for lookback_window in LOOKBACK_WINDOW_CHOICES:
                X_lb, y_lb = val_windows(X_y_all, ref_day=lookback_window, period=20)
                X_train_lb, y_train_lb = make_datasets(
                    X_lb,
                    y_lb,
                    X_y_all.index,
                    train_start,
                    train_end,
                    lookback_window
                )
                if X_train_lb.shape[0] == 0:
                    print(f"[WARN] lookback_window={lookback_window} has no train samples, skip.")
                    continue
                windowed_train_data[lookback_window] = (X_train_lb, to_categorical(y_train_lb))

            if len(windowed_train_data) == 0:
                raise ValueError("No valid training data for any lookback_window choice.")
            init_lookback = sorted(windowed_train_data.keys())[0]
            init_x, _ = windowed_train_data[init_lookback]
            init_input_shape = (init_x.shape[1], init_x.shape[2])
           
            # ===================== 4) Build distribution strategy ===============
            # Prefer OneDeviceStrategy on single GPU (more stable); MirroredStrategy on multi-GPU.
            gpu_count = len(tf.config.list_physical_devices('GPU'))
            if gpu_count > 1:
                strategy = tf.distribute.MirroredStrategy()
            elif gpu_count == 1:
                strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
            else:
                strategy = tf.distribute.OneDeviceStrategy(device='/CPU:0')

            # ===================== 5) Two-stage hyperparameter search ===========
            # Stage 1: broad search (fewer epochs; explore quickly).
            project_name = f'ATT_{model_type}_{stock_id}'
            tuner_stage1 = TunerCV(
                hypermodel=HyperTCN(input_shape=init_input_shape, strategy=strategy),  # provide a valid shape now; run_trial overrides it per lookback
                oracle=kt.oracles.BayesianOptimizationOracle(
                    objective=kt.Objective('val_recall_score', 'max'),
                    max_trials=STAGE1_MAX_TRIALS
                ),
                directory=platform_path(ATT_HYPERBAYES_DIR),
                project_name=project_name,
                overwrite=True,
                distribution_strategy=strategy
            )
            tuner_stage1.search(
                windowed_data=windowed_train_data,
                batch_size=60,
                epochs=STAGE1_EPOCHS,
            )

            # Stage 2: fine-grained search (continue same project; more trials, more epochs).
            try:
                tuner_stage2 = TunerCV(
                    hypermodel=HyperTCN(input_shape=init_input_shape, strategy=strategy),
                    oracle=kt.oracles.BayesianOptimizationOracle(
                        objective=kt.Objective('val_recall_score', 'max'),
                        max_trials=STAGE2_MAX_TRIALS
                    ),
                    directory=platform_path(ATT_HYPERBAYES_DIR),
                    project_name=project_name,
                    overwrite=False,
                    distribution_strategy=strategy
                )
                tuner_stage2.search(
                    windowed_data=windowed_train_data,
                    batch_size=60,
                    epochs=STAGE2_EPOCHS,
                )
            except KeyError as stage2_error:
                print(f"[WARN] Stage2 tuner state corrupted ({stage2_error}). Rebuild Stage2 with overwrite=True.")
                tuner_stage2 = TunerCV(
                    hypermodel=HyperTCN(input_shape=init_input_shape, strategy=strategy),
                    oracle=kt.oracles.BayesianOptimizationOracle(
                        objective=kt.Objective('val_recall_score', 'max'),
                        max_trials=STAGE2_MAX_TRIALS
                    ),
                    directory=platform_path(ATT_HYPERBAYES_DIR),
                    project_name=project_name,
                    overwrite=True,
                    distribution_strategy=strategy
                )
                tuner_stage2.search(
                    windowed_data=windowed_train_data,
                    batch_size=60,
                    epochs=STAGE2_EPOCHS,
                )

            # Write the best-trial summary (incl. lookback_window) to disk for the fixed-params trainer to consume.
            best_trials = tuner_stage2.oracle.get_best_trials(num_trials=1)
            if len(best_trials) > 0:
                best_trial = best_trials[0]
                best_hp_values = dict(best_trial.hyperparameters.values)
                summary_path = f"{platform_path(ATT_HYPERBAYES_DIR)}/ATT_{model_type}_{stock_id}/best_trial_summary.json"
                with open(summary_path, 'w', encoding='utf-8') as summary_file:
                    json.dump(
                        {
                            'trial_id': best_trial.trial_id,
                            'score': best_trial.score,
                            'lookback_window': best_hp_values.get('lookback_window'),
                            'hyperparameters': best_hp_values
                        },
                        summary_file,
                        indent=2,
                        ensure_ascii=False
                    )
                print(f"[BEST] trial={best_trial.trial_id}, lookback_window={best_hp_values.get('lookback_window')}, score={best_trial.score}")
                print(f"[BEST] summary saved: {summary_path}")
            #device = cuda.get_current_device()
            #device.reset()
            
        except Exception as e :
            # On exception, keep a minimal traceback and record the failed combo so the whole batch does not abort.
            traceback.print_exc(limit=1, file=sys.stdout)
            error_list.append([stock_id, model_type])
            continue
        # Clean up the graph and Python memory after each ticker finishes
        K.clear_session()    
        gc.collect()

