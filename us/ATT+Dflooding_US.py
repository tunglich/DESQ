import os, json, gc, sys, re

"""ATT fixed-hyperparameter training script (Dynamic Flooding + causal attention).

Differences vs. the Keras Tuner version:
- No trial search; loads previously discovered best hyperparameters and retrains repeatedly.
- Focuses on stable reruns, saving multiple experiment models, and writing per-run predictions.
- Intended as a production / batch update workflow.
"""

# Reduce TensorFlow/XLA log noise and first-compile stalls (must be set before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 'TF_XLA_FLAGS' not in os.environ:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
if 'XLA_FLAGS' not in os.environ:
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
if 'TF_GPU_ALLOCATOR' not in os.environ:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
if 'TF_USE_LEGACY_KERAS' not in os.environ:
    os.environ['TF_USE_LEGACY_KERAS'] = '0'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import PowerTransformer, RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, joblib
try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None
#from numba import cuda0


# n_steps = lookback window
# to make dataset into windows

tf.get_logger().setLevel('ERROR')

# Fall back to native Adam if tensorflow-addons is unavailable (e.g. newer TF/Keras combinations)
USE_TFA_OPTIMIZER = (os.getenv('ENABLE_TFA_OPTIMIZER', '0') == '1') and (tfa is not None)
FORCE_MODEL_BUILD_ON_CPU = os.getenv('FORCE_MODEL_BUILD_ON_CPU', '0') == '1'
ENABLE_XLA = os.getenv('ENABLE_XLA', '0') == '1'
ENABLE_TF32 = os.getenv('ENABLE_TF32', '1') == '1'
USE_MIXED_PRECISION = os.getenv('ENABLE_MIXED_PRECISION', '1') == '1'
GPU_MEMORY_LIMIT_MB = int(os.getenv('GPU_MEMORY_LIMIT_MB', '0'))
FIT_VERBOSE = int(os.getenv('FIT_VERBOSE', '1'))

def platform_path(path_str):
    """Normalize a path string for the current runtime.

    Args:
        path_str: original path (may be a Windows `D:/...` string).

    Returns:
        Original value on Windows; `/mnt/<drive>/...` on Linux / WSL.

    Side effects:
        None.
    """
    if os.name != 'nt' and len(path_str) >= 2 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path_str

DATA_ROOT = os.getenv('DATA_ROOT', platform_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature')))
HYPER_ROOT = os.getenv('HYPERBAYES_ATT_DIR', platform_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyper')))
FEATURE_SELECTION_ROOT = os.getenv('FEATURE_SELECTION_ATT_DIR', platform_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selection')))
SCALER_ROOT = os.getenv('FEATURE_SCALER_ATT_DIR', platform_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scalar')))
EXPERIMENT_ROOT = os.getenv('EXPERIMENTS_ATT_DIR', platform_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment')))

# Walk-Forward Validation options
# VALIDATION_MODE: 'blocking' (default, single block CV — prior behavior),
#                  'walk_forward_rolling'   — fixed-size train window slides forward,
#                  'walk_forward_expanding' — train start fixed, train window grows.
# Dflooding is the "final training" stage: it picks the last (most recent)
# walk-forward fold as the train/val split, matching the distribution seen at
# live inference time as closely as possible.


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
      2) If unset and running under an interactive TTY: prompt the user (default rolling).
      3) Otherwise fall back to `blocking` (legacy behaviour).
    """
    raw = os.getenv('VALIDATION_MODE')
    if raw is not None:
        normalized = _normalize_validation_mode(raw)
        if normalized in {'blocking', 'walk_forward_rolling', 'walk_forward_expanding'}:
            return normalized
        print(f"[WARN] Unknown VALIDATION_MODE={raw!r}, fallback to 'blocking'.")
        return 'blocking'

    if not sys.stdin.isatty():
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
WF_GAP = int(os.getenv('WF_GAP', '20'))
WF_VAL_YEARS = float(os.getenv('WF_VAL_YEARS', '0'))
WF_TRADING_DAYS_PER_YEAR = int(os.getenv('WF_TRADING_DAYS_PER_YEAR', '252'))
WF_VAL_SAMPLES = int(round(WF_VAL_YEARS * WF_TRADING_DAYS_PER_YEAR)) if WF_VAL_YEARS > 0 else 0

Path(FEATURE_SELECTION_ROOT).mkdir(parents=True, exist_ok=True)
Path(SCALER_ROOT).mkdir(parents=True, exist_ok=True)
Path(EXPERIMENT_ROOT).mkdir(parents=True, exist_ok=True)


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
    downstream preprocessing (corr filter / scaler / sanitize) runs on top of
    this function's output.
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


# macro column groups (see _expand_macro_features in ATT+Flood.py for details)
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
    before transforming. Must stay in exact lockstep with the version in ATT+Flood.py
    / prediction_update_tony_2026.py.
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    for c in MACRO_STATIONARY_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if c in ('Price_rate_3m', 'Price_rate_10y', 'Price_FX'):
                s = s.replace(0, np.nan).ffill()
            out[c] = s

    if 'Price_rate_3m' in out.columns and 'Price_rate_10y' in out.columns:
        out['term_spread'] = out['Price_rate_10y'] - out['Price_rate_3m']

    if 'Price_VIX' in out.columns:
        vix = out['Price_VIX'].replace(0, np.nan).ffill()
        out['log_vix'] = np.log(vix)
        m = vix.rolling(60, min_periods=30).mean()
        sd = vix.rolling(60, min_periods=30).std().replace(0, np.nan)
        out['vix_z60'] = (vix - m) / sd

    for c in MACRO_PRICE_LEVEL_COLS:
        if c not in df.columns:
            continue
        p = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        out[f'{c}_logret1']  = np.log(p / p.shift(1))
        out[f'{c}_logret20'] = np.log(p / p.shift(20))
        m = p.rolling(60, min_periods=30).mean()
        sd = p.rolling(60, min_periods=30).std().replace(0, np.nan)
        out[f'{c}_z60'] = (p - m) / sd

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

    for c in MACRO_SIGNED_LEVEL_COLS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        sl = np.sign(s) * np.log1p(np.abs(s))
        out[f'{c}_slog1p']  = sl
        out[f'{c}_diff5']   = sl - sl.shift(5)
        out[f'{c}_diff20'] = sl - sl.shift(20)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


# ===================================================================
# tech_trend / fundamental / moment expanders (see the same-name functions in
# ATT+Flood.py for details). Must stay in exact lockstep with ATT+Flood.py /
# prediction_update_tony_2026.py implementations.
# ===================================================================


def _expand_tech_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    passthrough_cols = [
        'sma_5', 'sma_10', 'sma_20', 'sma_60', 'sma_120',
        'hullma_20', 'hullma_60', 'hullma_120',
        'mmi_5', 'mmi_10', 'mmi_20',
        'aroon_osc', 'bb', 'bias', 'alpha',
    ]
    for c in passthrough_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors='coerce')

    if 'osc' in df.columns and 'close' in df.columns:
        close_raw = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        osc_raw = pd.to_numeric(df['osc'], errors='coerce')
        out['osc_pct'] = (osc_raw / close_raw).clip(-0.3, 0.3)

    if 'close' in df.columns:
        close = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        log_close = np.log(close)
        out['ret_1']  = (log_close - log_close.shift(1)).clip(-0.3, 0.3)
        out['ret_5']  = (log_close - log_close.shift(5)).clip(-0.5, 0.5)
        out['ret_20'] = (log_close - log_close.shift(20)).clip(-0.8, 0.8)

    if {'high', 'low', 'close'}.issubset(df.columns):
        h = pd.to_numeric(df['high'], errors='coerce')
        l = pd.to_numeric(df['low'], errors='coerce')
        c = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        out['hl_range'] = ((h - l) / c).clip(0.0, 0.2)
    if {'open', 'close'}.issubset(df.columns):
        o = pd.to_numeric(df['open'], errors='coerce').replace(0, np.nan).ffill()
        c = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        out['gap'] = (np.log(o) - np.log(c.shift(1))).clip(-0.15, 0.15)

    if 'volume' in df.columns:
        v = pd.to_numeric(df['volume'], errors='coerce').clip(lower=0)
        v_ma = v.rolling(20, min_periods=5).mean().replace(0, np.nan)
        out['vol_ratio20'] = ((v / v_ma) - 1.0).clip(-5.0, 5.0)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


FUNDAMENTAL_GROWTH_COLS = (
    'R_mom', 'R_yoy', 'R_acc_yoy',
    'E_qoq', 'E_yoy', 'E_acc_yoy',
    'Op_qoq', 'Op_yoy', 'Op_acc_yoy',
    'Gross_qoq', 'EPS_qoq',
)


def _expand_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    for c in ('PE_trailing', 'PBR'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 1.0)
            out[c] = (s - 0.5) * 2.0

    if 'DY' in df.columns:
        s = pd.to_numeric(df['DY'], errors='coerce').clip(0.0, 20.0)
        out['DY'] = s / 10.0

    if 'Gross' in df.columns:
        s = pd.to_numeric(df['Gross'], errors='coerce').clip(-20.0, 100.0)
        out['Gross'] = s / 100.0

    for c in FUNDAMENTAL_GROWTH_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(-300.0, 300.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s) / 100.0)

    if 'PEG' in df.columns:
        s = pd.to_numeric(df['PEG'], errors='coerce').clip(-10.0, 10.0)
        out['PEG'] = np.sign(s) * np.log1p(np.abs(s))

    if 'CMDTY' in df.columns:
        p = pd.to_numeric(df['CMDTY'], errors='coerce').replace(0, np.nan).ffill()
        out['CMDTY_logret20'] = (np.log(p) - np.log(p.shift(20))).clip(-0.5, 0.5)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


def _expand_moment_features(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    for c in ('rsi', 'k', 'd', 'adx'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 100.0)
            out[c] = (s - 50.0) / 50.0

    if 'wr' in df.columns:
        s = pd.to_numeric(df['wr'], errors='coerce').clip(-100.0, 0.0)
        out['wr'] = (s + 50.0) / 50.0

    if 'cci' in df.columns:
        s = pd.to_numeric(df['cci'], errors='coerce').clip(-300.0, 300.0)
        out['cci'] = s / 100.0

    for c in ('acc_5', 'acc_10', 'acc_20', 'acc_60', 'acc_120'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.5, 2.0)
            out[c] = np.log(s)

    if 'vpt' in df.columns:
        s = pd.to_numeric(df['vpt'], errors='coerce')
        d = s.diff()
        m = d.rolling(60, min_periods=20).mean()
        sd = d.rolling(60, min_periods=20).std().replace(0, np.nan)
        out['vpt_z60'] = ((d - m) / sd).clip(-3.0, 3.0)

    if 'beta' in df.columns:
        s = pd.to_numeric(df['beta'], errors='coerce').clip(-3.0, 3.0)
        out['beta'] = s

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


# XLA JIT is on by default to speed up training; set DISABLE_XLA=1 to disable if unstable
tf.config.optimizer.set_jit(ENABLE_XLA)
DEFAULT_LOOKBACK_WINDOW = 20

# Mixed precision is on by default; set DISABLE_MIXED_PRECISION=1 if you see NaN / Inf during training.
if USE_MIXED_PRECISION:
    try:
        mixed_precision.set_global_policy('mixed_float16')
    except Exception as e:
        print(f"[WARN] mixed precision setup failed: {e}")

def add_sinusoidal_positional_encoding(x, seq_len, feature_dim):
    """Build a fixed sin/cos positional encoding and add it to the input tensor.

    Args:
        x: input tensor.
        seq_len: sequence length.
        feature_dim: feature dimensionality.

    Returns:
        Input tensor with positional encoding added.

    Side effects:
        None.
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    idx = np.arange(feature_dim)[np.newaxis, :]
    denom = np.power(10000.0, (2 * (idx // 2)) / max(feature_dim, 1))
    angle_rads = pos / denom

    pe = np.zeros((seq_len, feature_dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pe = tf.constant(pe, dtype=tf.float32)

    return layers.Lambda(
        lambda t: t + tf.cast(pe, t.dtype),
        name="pos_emb"
    )(x)

try:
    tf.config.experimental.enable_tensor_float_32_execution(ENABLE_TF32)
except Exception:
    pass

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
            # Enable memory growth to avoid grabbing all VRAM at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(
    f"[RUNTIME] force_build_on_cpu={FORCE_MODEL_BUILD_ON_CPU}, "
    f"enable_xla={ENABLE_XLA}, enable_tf32={ENABLE_TF32}, "
    f"enable_mixed_precision={USE_MIXED_PRECISION}, use_tfa_optimizer={USE_TFA_OPTIMIZER}, "
    f"gpu_allocator={os.environ.get('TF_GPU_ALLOCATOR')}, gpu_memory_limit_mb={GPU_MEMORY_LIMIT_MB}"
)

def warmup_one_batch_from_dataset(model, train_dataset):
    """Warm up on 1 dataset batch to move the first-compile latency earlier.

    Args:
        model: compiled Keras model.
        train_dataset: training `tf.data.Dataset` pipeline.

    Returns:
        None.

    Side effects:
        Triggers a one-time eval / inference graph compile; does not update weights.
    """
    try:
        warmup_batch = next(iter(train_dataset.take(1)))
    except StopIteration:
        return
    except Exception as warmup_error:
        print(f"[WARN] warmup batch fetch failed: {warmup_error}")
        return

    try:
        if isinstance(warmup_batch, (tuple, list)) and len(warmup_batch) >= 2:
            x_warm, y_warm = warmup_batch[0], warmup_batch[1]
            model.test_on_batch(x_warm, y_warm, return_dict=False)
            model.predict_on_batch(x_warm)
    except Exception as warmup_error:
        print(f"[WARN] warmup skipped: {warmup_error}")

def sequence_to_windows(seq, y, n_steps):
    """Cut a continuous sequence into fixed-length sliding windows.

    Args:
        seq: feature-sequence DataFrame.
        y: label sequence (Series / array).
        n_steps: window length.

    Returns:
        (_X, _y)
        - _X: shape=(n_samples, n_steps, n_features)
        - _y: shape=(n_samples,)

    Side effects:Side effects:
        None.
    """
    seq_arr = np.asarray(seq)
    y_arr = np.asarray(y)
    n_samples = seq_arr.shape[0] - n_steps + 1
    strides = (seq_arr.strides[0],) + seq_arr.strides
    _X = np.lib.stride_tricks.as_strided(
        seq_arr, shape=(n_samples, n_steps, seq_arr.shape[1]), strides=strides
    ).copy()
    _y = y_arr[n_steps - 1:].copy()
    return _X, _y

def get_windows(X, y, slice, steps):
    """Slice windowed data by index and enforce a minimum history-length guard.

    Args:
        X: windowed feature array.
        y: windowed label array.
        slice: date-index slice result (with start / stop).
        steps: window length.

    Returns:
        (X_slice, y_slice) subset.

    Side effects:
        None.
    """
    min_idx = 1 * 250 - 1
    start_idx, end_idx, _ = slice.start, slice.stop, slice.step
    start_idx = max(min_idx, start_idx - steps + 1)
    end_idx = end_idx - steps + 1
    return X[start_idx:end_idx], y[start_idx:end_idx]

def val_windows(data, ref_day=60, period=20):
    """Convert data into supervised windows; the last 4 columns are treated as labels / reserved.

    Args:
        data: source DataFrame.
        ref_day: lookback window length.
        period: label column suffix; reads `y_{period}`.

    Returns:
        (X_val, y_val) windowed result.

    Side effects:
        None.
    """
    n_features = data.shape[1] - 4
    X_val, y_val = [], []
    for i in range(data.shape[0] - ref_day + 1):
        X_val.append(np.array(data.iloc[i:i+ref_day, :-4]))
        y_val.append(np.array(data.iloc[i+ref_day-1][f"y_{period}"]))
    X_val = np.array(X_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], n_features))
    y_val = np.array(y_val)
    return X_val, y_val

def make_datasets(X, y, idx, start, end, ref_day):
    """Slice data by date while preserving a minimum history segment for feature stability.

    Args:
        X: windowed feature array.
        y: windowed label array.
        idx: original date index.
        start: start-date string.
        end: end-date string.
        ref_day: window length.

    Returns:
        (X_slice, y_slice) cast to Keras floatx.

    Side effects:
        None.
    """
    start_idx = idx.get_loc(idx.to_series()[start:].iloc[0])
    end_idx = idx.get_loc(idx.to_series()[:end].iloc[-1])
    st = max(1*250-1, start_idx-ref_day+1)
    ed = end_idx-ref_day+2
    return K.cast_to_floatx(X[st:ed]), K.cast_to_floatx(y[st:ed])

class BlockingTimeSeriesSplit:
    """Time-series splitter (Blocking CV).

    Purpose:
        Split data chronologically and insert a gap between train and val to reduce leakage.

    Args:
        n_splits: number of splits.
        val_ratio: validation fraction per block.
        gap: samples between train and val.

    Attributes:
        n_splits, val_ratio, gap。

    Side effects:Side effects:
        None.
    """
    def __init__(self, n_splits, val_ratio=0.25, gap=10):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.gap = gap
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    # Split chronologically; leave a gap of `gap` days between train and val to avoid leakage
    def split(self, X, y=None, groups=None):
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

    def __init__(self, n_splits=5, val_ratio=0.2, gap=10, mode='rolling', val_samples=0):
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
        n_samples = int(len(X))
        if n_samples <= self.gap + 2:
            return
        if self.val_samples > 0:
            val_size = max(1, int(self.val_samples))
        else:
            val_size = max(1, int(n_samples * self.val_ratio / self.n_splits))
        total_val = val_size * self.n_splits
        if total_val + self.gap + 1 >= n_samples:
            val_size = max(1, (n_samples - self.gap - 2) // (self.n_splits + 1))
            total_val = val_size * self.n_splits
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
            else:
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
    return BlockingTimeSeriesSplit(n_splits=1)

class FloodingModel(keras.Model):
    """Custom Keras model that applies the Flooding training strategy.

    Purpose:
        Override `train_step` to convert the batch loss into flooding loss and
        curb overfitting.

    Args:
        Same constructor arguments as `keras.Model` (built via the Functional API).

    Attributes:
        flooding_b: flooding level (float, default 0.10).

    Side effects:
        Changes the batch-loss computation.
    """
    # Fixed flooding level; adjust as needed
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
            loss = self.compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )
            if self.flooding_b > 0:
                loss = tf.math.abs(loss - self.flooding_b) + self.flooding_b

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # FBetaScore / F1Score require one-hot 2D y_true; convert integer labels first
        y_flat = tf.cast(tf.reshape(y, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_onehot = tf.one_hot(y_flat, depth=num_classes)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                try:
                    metric.update_state(y_onehot, y_pred, sample_weight=sample_weight)
                except Exception:
                    metric.update_state(y_flat, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        y_pred = self(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)

        y_flat = tf.cast(tf.reshape(y, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_onehot = tf.one_hot(y_flat, depth=num_classes)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                try:
                    metric.update_state(y_onehot, y_pred, sample_weight=sample_weight)
                except Exception:
                    metric.update_state(y_flat, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}


class DynamicFloodingCallback(tf.keras.callbacks.Callback):
    """Callback that dynamically adjusts `flooding_b` based on a validation metric.

    Purpose:
        At each epoch, raise or lower the flooding level depending on whether
        the monitored metric improved.

    Args:
        monitor: name of the monitored metric.
        min_b / max_b: lower and upper bounds for flooding_b.
        step_up / step_down: adjustment step size.
        patience: epochs of no improvement tolerated.
        min_delta: minimum improvement magnitude to count as progress.
        verbose: whether to print adjustment info.

    Attributes:
        best, wait, and related adjustment control state.

    Side effects:
        Mutates `self.model.flooding_b` directly.
    """
    # Dynamically adjust flooding_b based on val_recall
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


def select_uncorrelated_features(feature_df, cutoff=0.85):
    """Keep low-collinearity features using a correlation-coefficient threshold."""
    corr_matrix = feature_df.corr()
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [var for var in corr_matrix.columns if any(corr_matrix[var] > cutoff)]
    return corr_matrix.drop(drop_cols, axis=1).columns.tolist()


def fit_sanitize_statistics(df, max_abs=1e6, q=0.001):
    """Estimate sanitize statistics from the training slice only, avoiding future-data leakage."""
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


def sanitize_for_power_transform(df, max_abs=1e6, q=0.001):
    """Backwards-compat: single-shot sanitize (statistics estimated from the same batch)."""
    stats = fit_sanitize_statistics(df, max_abs=max_abs, q=q)
    return apply_sanitize_statistics(df, stats)

def plot_history(history, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(5, 5))
    ax[0].plot(history['accuracy'])
    ax[0].plot(history['val_accuracy'])
    ax[0].set_title('Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')
    ax[1].plot(history['loss'])
    ax[1].plot(history['val_loss'])
    ax[1].set_title('Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)

def plot_prediction(symbol, prediction, out_path):
    import pyodbc
    conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER=data.autoquant.ai,3333;DATABASE=AutoQuant;UID=aq;PWD=2020@autoquant;MARS_Connection=Yes")
    actual_price = pd.read_sql(f"SELECT date, close FROM sysdbase WHERE ticker = '{symbol}' AND date BETWEEN '{prediction.index.min().strftime('%Y-%m-%d')}' AND '{prediction.index.max().strftime('%Y-%m-%d')}' ORDER BY date ASC", conn, index_col='date', parse_dates=True).iloc[:, 0]
    prediction = prediction.reindex(actual_price.index)

    fig, ax = plt.subplots(2, figsize=(12, 5))
    x = np.arange(actual_price.shape[0])
    ticks = [d.strftime('%Y/%m/%d') for d in actual_price.index]
    space = max(int(len(ticks) / 10), 1)
    ticks = [tick_ if i % space == 0 or i == len(ticks) - 1 else '' for i, tick_ in enumerate(ticks)]
    ax[0].plot(x, actual_price.values, color='k')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ticks)
    ax[1].bar(x, prediction.values, color=['red' if i > 0.5 else 'green' for i in prediction.values])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(ticks)
    ax[1].axhline(0.5, linestyle='--', c='k')
    ax[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)

def add_causal_mha_block(x, num_heads, key_dim, dropout_rate, name_prefix, seq_len):
    """Build a causal multi-head attention block (dynamic-length mask version).

    Args:
        x: input tensor.
        num_heads: number of attention heads.
        key_dim: key dimensionality.
        dropout_rate: dropout ratio.
        name_prefix: layer name prefix.
        seq_len: sequence length (used to build the initial mask).

    Returns:
        Block output tensor.

    Side effects:
        None.
    """
    # Build the causal mask dynamically to avoid graph-mode compatibility issues with fixed tensors
    class DynamicCausalMask(layers.Layer):
        def call(self, x):
            seq_len = tf.shape(x)[1]
            return tf.linalg.band_part(tf.ones((1, seq_len, seq_len), dtype=tf.float32), -1, 0)
        def get_config(self):
            return super().get_config()
    
    causal_mask = DynamicCausalMask(name=f"{name_prefix}_causal_mask")(x)
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

def normalize_hyperparameters(hp):
    """Normalize an externally loaded hp dict into the format the model expects.

    Args:
        hp: hyperparameter dict that may contain alias keys or string values.

    Returns:
        Fully defaulted, typed hyperparameter dict.

    Side effects:
        None.
    """
    defaults = {
        'attn_layers': 2,
        'attn_heads_1': 2,
        'attn_key_dim_1': 16,
        'attn_dropout_1': 0.1,
        'attn_heads_2': 2,
        'attn_key_dim_2': 16,
        'attn_dropout_2': 0.1,
        'attn_heads_3': 2,
        'attn_key_dim_3': 16,
        'attn_dropout_3': 0.1,
        'Dense_units_1': 128,
        'Dense_kernel_1': 'glorot_normal',
        'activation_2': 'relu',
        'Dense_kernel_2': 'glorot_normal',
        'temp': 1.0,
    }
    aliases = {
        'attn_layers': ['num_attn_layers'],
        'attn_heads_1': ['attn_heads', 'num_heads_1', 'heads_1'],
        'attn_key_dim_1': ['attn_key_dim', 'key_dim_1'],
        'attn_dropout_1': ['dropout_1', 'attn_dropout'],
        'attn_heads_2': ['num_heads_2', 'heads_2'],
        'attn_key_dim_2': ['key_dim_2'],
        'attn_dropout_2': ['dropout_2'],
        'attn_heads_3': ['num_heads_3', 'heads_3'],
        'attn_key_dim_3': ['key_dim_3'],
        'attn_dropout_3': ['dropout_3'],
        'Dense_units_1': ['Dense_units', 'dense_units', 'units_1'],
        'Dense_kernel_1': ['Dense_kernel', 'kernel_initializer_2'],
        'activation_2': ['activation_dense'],
        'Dense_kernel_2': ['Dense_kernel_out', 'kernel_initializer_out'],
        'temp': ['temperature'],
    }

    resolved = {}
    for key, default_value in defaults.items():
        value = hp.get(key, None)
        if value is None:
            for alias in aliases.get(key, []):
                if alias in hp and hp[alias] is not None:
                    value = hp[alias]
                    break
        if value is None:
            value = default_value
        resolved[key] = value

    resolved['attn_layers'] = max(1, min(3, int(resolved['attn_layers'])))
    resolved['attn_heads_1'] = int(resolved['attn_heads_1'])
    resolved['attn_key_dim_1'] = int(resolved['attn_key_dim_1'])
    resolved['attn_dropout_1'] = float(resolved['attn_dropout_1'])
    resolved['attn_heads_2'] = int(resolved['attn_heads_2'])
    resolved['attn_key_dim_2'] = int(resolved['attn_key_dim_2'])
    resolved['attn_dropout_2'] = float(resolved['attn_dropout_2'])
    resolved['attn_heads_3'] = int(resolved['attn_heads_3'])
    resolved['attn_key_dim_3'] = int(resolved['attn_key_dim_3'])
    resolved['attn_dropout_3'] = float(resolved['attn_dropout_3'])
    resolved['Dense_units_1'] = int(resolved['Dense_units_1'])
    resolved['temp'] = max(float(resolved['temp']), 1e-3)
    return resolved

def build_model(hp, input_shape):
    """Build a FloodingModel from normalized hyperparameters.

    Args:
        hp: hyperparameter dict (may include raw / alias keys).
        input_shape: input shape (time, features).

    Returns:
        A `FloodingModel` instance (not yet compiled).

    Side effects:
        None.
    """
    # Same architecture as the ATT+Flood AutoML search: PositionEmbedding + Nx Attention + FC
    hp = normalize_hyperparameters(hp)
    inputs = keras.Input(shape=input_shape, name="inputs")

    # Use a weightless sinusoidal encoding: the third-party PositionEmbedding triggers CUDA cast errors on this env
    x = add_sinusoidal_positional_encoding(inputs, seq_len=input_shape[0], feature_dim=input_shape[1])

    attn_configs = [
        (hp['attn_heads_1'], hp['attn_key_dim_1'], hp['attn_dropout_1']),
        (hp['attn_heads_2'], hp['attn_key_dim_2'], hp['attn_dropout_2']),
        (hp['attn_heads_3'], hp['attn_key_dim_3'], hp['attn_dropout_3']),
    ]
    for layer_index in range(hp['attn_layers']):
        num_heads, key_dim, dropout_rate = attn_configs[layer_index]
        x = add_causal_mha_block(
            x,
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            name_prefix=f"attn{layer_index + 1}",
            seq_len=input_shape[0]
        )

    x = layers.GlobalAveragePooling1D(name="attn_pool")(x)
    x = LayerNormalization(name="attn_pool_ln")(x)

    # Dense layers fuse features
    x = Dense(
        hp['Dense_units_1'],
        kernel_initializer=hp['Dense_kernel_1'],
        activation=hp['activation_2']
    )(x)
    x = LayerNormalization()(x)

    logits = Dense(
        2,
        kernel_initializer=hp['Dense_kernel_2']
    )(x)

    logits = layers.Lambda(lambda t: t / hp['temp'], name="temperature")(logits)
    outputs = layers.Activation("softmax", dtype='float32', name="softmax")(logits)

    return FloodingModel(inputs, outputs)

def load_best_att_hyperparameters(root_dir, train_model, symbol):
    """Load the ATT+Flood best hyperparameters (prefer best_trial_summary, fall back to trial.json)."""
    project_dir = f'{root_dir}/ATT_{train_model}_{symbol}'
    summary_path = f'{project_dir}/best_trial_summary.json'

    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as summary_file:
            summary_json = json.load(summary_file)
        hp_values = summary_json.get('hyperparameters', {}) or {}
        if len(hp_values) > 0:
            return hp_values

    trials = []
    if not os.path.isdir(project_dir):
        return None

    for trial_dir in [x for x in os.listdir(project_dir) if x.startswith('trial')]:
        trial_json_path = f'{project_dir}/{trial_dir}/trial.json'
        if not os.path.isfile(trial_json_path):
            continue
        try:
            with open(trial_json_path, 'r', encoding='utf-8') as trial_file:
                trial_json = json.load(trial_file)
            score = trial_json.get('score', None)
            if score is not None:
                trials.append((score, trial_json))
        except Exception:
            continue

    if len(trials) == 0:
        return None

    _, best_trial_json = sorted(trials, key=lambda item: item[0], reverse=True)[0]
    hp_values = best_trial_json.get('hyperparameters', {}).get('values', {})
    return hp_values if len(hp_values) > 0 else None
def prepare_dataset(m, n):
    """Prepare train / test data for a given model type and ticker.

    Args:
        m: model type (e.g. macro, fundamental).
        n: ticker.

    Returns:
        Preprocessing and splitting mostly happen in-place; the return value follows the original implementation.

    Side effects:
        Reads files, applies scaling, and transforms the data.
    """
    # Prepare train / test data by ticker and model type
    X_y_all = pd.read_csv(f"features/{m}_{n}.csv", index_col=0, parse_dates=True)
    
    # Concate trade with sentiment
    '''if m == 'trade':
        y_all = X_all.iloc[:, -12:]
        X_sent = pd.read_csv(f"features/sentiment_{n}.csv", index_col=0, parse_dates=True).iloc[:, :-12]
        X_all = pd.concat([X_all.iloc[:, :-12], X_sent * 100, y_all], axis=1)
        del y_all, X_sent'''
    
        
    # Find the ealist vaild start date where one figure is positive 
    
    temp = X_y_all.iloc[:,:-4]
    non_zero_date = sorted([x for x in temp.index if (temp.loc[x,:]!=0).any()])[0]
    
    train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')               
    n_timesteps = 1
    #train_start = '2006-01-01'
    #train_start = '2010-01-01'
    #train_start = '2002-12-17'
    val_end = '2025-12-31'
    train_end = '2025-12-31'
    test_start = '2026-01-01'
    test_end = X_y_all.index.max().strftime('%Y-%m-%d')

    # Feature selection and normalization =======================================================================================================================
    train_feature_slice = X_y_all.loc[:train_end].iloc[:, :-4]
    selected_feature_cols = select_uncorrelated_features(train_feature_slice, cutoff=0.85)
    X_y_all = X_y_all[selected_feature_cols + X_y_all.iloc[:, -4:].columns.tolist()]

    # Use PowerTransformer to nudge feature distributions closer to normal
    scaler = PowerTransformer(method='yeo-johnson')
    lookback_start = X_y_all.loc[:train_start].iloc[-n_timesteps+1:].index.min()
    fit_raw_features = X_y_all.loc[lookback_start:train_end].iloc[:, :-4]
    sanitize_stats = fit_sanitize_statistics(fit_raw_features)
    fit_features = apply_sanitize_statistics(fit_raw_features, sanitize_stats)
    all_features = apply_sanitize_statistics(X_y_all.iloc[:, :-4], sanitize_stats)
    try:
        scaler.fit(fit_features)
        transformed = scaler.transform(all_features)
    except ValueError as scale_error:
        print(f"[WARN] PowerTransformer failed, fallback to RobustScaler: {scale_error}")
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        scaler.fit(fit_features)
        transformed = scaler.transform(all_features)

    X_y_all = pd.concat(
        [
            pd.DataFrame(
                transformed,
                index=X_y_all.index,
                columns=X_y_all.columns[:-4]
            ),
            X_y_all.iloc[:, -4:]
        ],
        axis=1
    ).loc[lookback_start:]

    # Time-series window length
    
    X, y = val_windows(X_y_all, ref_day=DEFAULT_LOOKBACK_WINDOW, period=20)

    X_train, y_train = make_datasets(X, y, X_y_all.index, train_start, train_end, DEFAULT_LOOKBACK_WINDOW)                      
    X_test, y_test = make_datasets(X, y, X_y_all.index, test_start, test_end, DEFAULT_LOOKBACK_WINDOW)

    return X_train, X_test, y_train, y_test, X_y_all.index

test_mode = False           # <=================== TEST MODE
root_dir = HYPER_ROOT
des_dir = 'dynamicFlooding'
DOW_30_TICKER = ['AXP','AMGN','AMZN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NVDA','NKE','PG','TRV','UNH','CRM','VZ','V','WMT','DIS','SHW']
symbols = [x.strip() for x in os.getenv('STOCK_IDS', ','.join(DOW_30_TICKER)).split(',') if x.strip()]
debug = True

# Flooding-level candidates (tune as needed)
flooding_b_candidates = [0.05, 0.10, 0.20]
repeats_per_flooding = int(os.getenv('DFLOOD_REPEATS_PER_FLOODING', '2'))
total_repeats = int(os.getenv('DFLOOD_TOTAL_REPEATS', '18'))

model_types_list = [x.strip() for x in os.getenv('MODEL_TYPES', 'fundamental,moment,tech_trend,macro').split(',') if x.strip()]


def _resolve_feature_preprocess():
    """Decide whether to run feature preprocessing (corr filter + sanitize + scaler).

    Resolution order:
      1) `FEATURE_PREPROCESS` env var (0/no/false = off; 1/yes/true = on).
      2) Interactive TTY: prompt the user.
      3) Otherwise on by default.
    """
    raw = os.getenv('FEATURE_PREPROCESS')
    if raw is not None:
        return raw.strip().lower() not in {'0', 'no', 'false', 'off', ''}
    if not sys.stdin.isatty():
        return True
    try:
        answer = input("Run feature preprocessing (corr filter + sanitize + Yeo-Johnson/Robust scaler)? [Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer not in {'n', 'no', '0', 'false'}


def _latest_dflood_log_path():
    logs_dir = Path('logs')
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob('dflood_first15_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_resume_scores(train_model, symbol, repeat_count):
    """Restore completed repeat scores by scanning all dflood logs.

    Prefers the log file whose count of ``[{train_model}_{symbol}]`` Average
    val_recall lines matches ``repeat_count`` exactly. Falls back to the most
    recent log that contains any matching lines.
    """
    if repeat_count <= 0:
        return []
    logs_dir = Path('logs')
    if not logs_dir.exists():
        return []
    pattern = re.compile(
        rf'^\[{re.escape(train_model)}_{re.escape(symbol)}\]\s+flooding_b=.*?Average val_recall:\s*([0-9.]+)'
    )
    # Match any dflood_*.log variant (dflood_first15_*, dflood_sp100_first15_*, etc.)
    candidates = sorted(
        logs_dir.glob('dflood_*.log'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    exact_match = None
    fallback = None
    # Also accumulate scores chronologically across logs as a third option
    # (handles cells whose 12 repeats are split across multiple resume logs).
    chronological = sorted(candidates, key=lambda p: p.stat().st_mtime)
    aggregated = []
    for log_path in chronological:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    match = pattern.search(line)
                    if match:
                        aggregated.append(float(match.group(1)))
        except Exception as exc:
            print(f'[WARN] resume score aggregate parse failed for {train_model}_{symbol} in {log_path.name}: {exc}')
            continue
    for log_path in candidates:
        scores = []
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    match = pattern.search(line)
                    if match:
                        scores.append(float(match.group(1)))
        except Exception as exc:
            print(f'[WARN] resume score parse failed for {train_model}_{symbol} in {log_path.name}: {exc}')
            continue
        if not scores:
            continue
        if len(scores) == repeat_count and exact_match is None:
            exact_match = (log_path, scores)
            break
        if fallback is None:
            fallback = (log_path, scores)
    if exact_match is not None:
        log_path, scores = exact_match
        print(f'[{train_model}_{symbol}] recovered {len(scores)} resume scores from {log_path.name}')
        return scores
    # If aggregating across all dflood logs chronologically yields the required
    # count, use the most recent `repeat_count` entries — they correspond to
    # the current run's repeats 1..repeat_count when training was split across
    # multiple resume logs.
    if len(aggregated) >= repeat_count:
        recovered = aggregated[-repeat_count:]
        print(
            f'[{train_model}_{symbol}] recovered {len(recovered)} resume scores '
            f'by aggregating across {len(chronological)} dflood logs'
        )
        return recovered
    if fallback is not None:
        log_path, scores = fallback
        print(
            f'[WARN] no exact-match log for {train_model}_{symbol} (need {repeat_count}); '
            f'using best available ({len(scores)} scores) from {log_path.name}'
        )
        return scores[:repeat_count]
    return []


DO_FEATURE_PREPROCESS = _resolve_feature_preprocess()
os.environ['FEATURE_PREPROCESS'] = '1' if DO_FEATURE_PREPROCESS else '0'
print(f"[PREPROCESS] feature preprocessing = {'ON' if DO_FEATURE_PREPROCESS else 'OFF (pass-through)'}")
print(
    f"[CV] VALIDATION_MODE={VALIDATION_MODE} WF_N_SPLITS={WF_N_SPLITS} "
    f"WF_VAL_RATIO={WF_VAL_RATIO} WF_VAL_YEARS={WF_VAL_YEARS} "
    f"WF_VAL_SAMPLES={WF_VAL_SAMPLES} WF_GAP={WF_GAP}"
)


for symbol in symbols:
    # Train per (ticker, model_type) pair
    for train_model in model_types_list:  # 'tech_trend','moment','sentiment', 'trade' --- IGNORE ---

        experiment_dir = Path(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}')
        if experiment_dir.exists():
            existing_idx = sorted(
                int(p.stem.split('_')[-1])
                for p in experiment_dir.glob('experiment_result_*.csv')
            )
        else:
            existing_idx = []
        n_existing = len(existing_idx)
        is_contiguous_from_1 = existing_idx == list(range(1, n_existing + 1))

        if n_existing == 0:
            completed_repeats = 0
        elif not is_contiguous_from_1:
            # Non-contiguous indices ⇒ already pruned to top-k from a prior
            # completed run. Nothing to do.
            print(
                f'[{train_model}_{symbol}] already completed '
                f'(pruned indices {existing_idx}), skipping'
            )
            continue
        elif n_existing >= total_repeats:
            # All repeats present but pruning never ran. Skip training; we'll
            # rely on resume-score recovery to prune at the end.
            print(
                f'[{train_model}_{symbol}] {n_existing} repeats present and contiguous '
                f'but unpruned; will attempt pruning only'
            )
            completed_repeats = total_repeats
        else:
            completed_repeats = n_existing
 
        # Only train when the ATT AutoML trial exists and no experiment file has been produced yet
        best_hp_values = load_best_att_hyperparameters(root_dir, train_model, symbol)
        if best_hp_values is not None and not os.path.exists(f'experiments_test/{train_model}{"_test" if test_mode else ""}'):
            best_lookback_window = int(best_hp_values.get('lookback_window', DEFAULT_LOOKBACK_WINDOW))
            print(f"[{train_model}_{symbol}] best lookback_window from AutoML: {best_lookback_window}")

            # ===================== 1) Load data and configure date range =====================
            # Read the feature data
            data = pd.read_csv(f"{DATA_ROOT}/{train_model}_{symbol}.csv", index_col=0, parse_dates=True)

            # sentiment aspect: expand the US/TW/ticker triple into momentum /
            # relative-strength / rolling z-score features, matching the ATT+Flood.py flow
            if train_model == 'sentiment':
                data = _expand_sentiment_features(data, symbol)

            # trade aspect: signed log1p on heavy-tailed columns first
            if train_model == 'trade':
                data = _prepare_trade_features(data)

            # macro aspect: price levels are highly non-stationary; convert to near-stationary
            # log-return / z-score / log1p features
            if train_model == 'macro':
                data = _expand_macro_features(data)

            # tech_trend aspect: raw OHLCV -> scale-free features; sma / hullma / bias / alpha kept
            if train_model == 'tech_trend':
                data = _expand_tech_trend_features(data)

            # fundamental aspect: PE / PBR / DY / Gross bounded rescale; growth rate & PEG clip + signed log1p
            if train_model == 'fundamental':
                data = _expand_fundamental_features(data)

            # moment aspect: recenter RSI/K/D/ADX/WR to +/-1; CCI clip; acc_* log; vpt diff z; beta clip
            if train_model == 'moment':
                data = _expand_moment_features(data)

            # sequence window length
            n_steps = best_lookback_window
            print(f"[{train_model}_{symbol}] final lookback_window in use = {n_steps}")
            forecast_days = 20
            
            if train_model == 'sentiment':
                train_start = '2015-01-06'
            elif train_model == 'trade':
                non_zero_date = _detect_non_zero_date(data, ratio_threshold=0.5)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            elif train_model == 'macro':
                non_zero_date = _detect_non_zero_date(data, ratio_threshold=0.9)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            elif train_model in ('tech_trend', 'moment'):
                # tech_trend: sma_120 needs 120 days; moment: acc_120 / vpt_z60 need 60-120 days
                non_zero_date = _detect_non_zero_date(data, ratio_threshold=0.5)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            else:
                temp = data.iloc[:,:-4]
                non_zero_date = sorted([x for x in temp.index if (temp.loc[x,:]!=0).any(axis=None)])[0]
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')     
            
            train_end = '2025-12-31'
            test_start, test_end = '2026-01-01', data.index.max().strftime('%Y-%m-%d')

            # ===================== 2) Feature cleaning and scaling =====================
            if not DO_FEATURE_PREPROCESS:
                # pass-through: skip correlation filter, sanitize, and scaler
                label_cols = data.columns[-4:]
                data = data.astype({col: np.float64 for col in label_cols})
                data = data[~data.index.duplicated(keep='last')]

                # Minimal NaN / inf cleanup on non-label columns only
                feat_df = data.iloc[:, :-4].apply(pd.to_numeric, errors='coerce').astype(np.float64)
                feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                data = pd.concat([feat_df, data.iloc[:, -4:]], axis=1)

                # Record the feature columns used this run (matches the preprocessed flow)
                if not os.path.exists(f"{FEATURE_SELECTION_ROOT}/{symbol}.json"):
                    with open(f"{FEATURE_SELECTION_ROOT}/{symbol}.json", "w+") as fout:
                        json.dump({train_model: data.columns[:-4].tolist()}, fout, indent=2)
                else:
                    with open(f"{FEATURE_SELECTION_ROOT}/{symbol}.json") as fin:
                        json_obj = json.load(fin)
                    json_obj[train_model] = data.columns[:-4].tolist()
                    with open(f"{FEATURE_SELECTION_ROOT}/{symbol}.json", "w+") as fout:
                        json.dump(json_obj, fout, indent=2)

                preprocess_bundle = {
                    'scaler': None,
                    'sanitize_stats': None,
                    'feature_columns': data.columns[:-4].tolist(),
                    'version': 'v3_passthrough'
                }
                joblib.dump(preprocess_bundle, f"{SCALER_ROOT}/scaler_{train_model}_{symbol}.pkl")
                print(f"[PREPROCESS] skipped for {symbol}/{train_model} (pass-through bundle saved)")
            else:
                # Corr_matrix is to remove dependence between features (fit on train period only)
                if train_model in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment'):
                    # Derived features intentionally keep complementary signals
                    # (short/mid/long return, multi-period growth rates, etc.); skip corr filter
                    selected_feature_cols = data.columns[:-4].tolist()
                else:
                    train_feature_slice = data.loc[:train_end].iloc[:, :-4]
                    selected_feature_cols = select_uncorrelated_features(train_feature_slice, cutoff=0.85)
                data = data[selected_feature_cols + data.columns[-4:].tolist()]
                if not os.path.exists(f"{FEATURE_SELECTION_ROOT}/{symbol}.json"):
                    with open(f"{FEATURE_SELECTION_ROOT}/{symbol}.json", "w+") as fout:
                        json.dump({
                            train_model: data.columns[:-4].tolist()}
                        , fout, indent=2)
                else:
                    with open (f"{FEATURE_SELECTION_ROOT}/{symbol}.json") as fin:
                        json_obj = json.load(fin)
                    json_obj[train_model] = data.columns[:-4].tolist()
                    with open(f"{FEATURE_SELECTION_ROOT}/{symbol}.json", "w+") as fout:
                        json.dump(json_obj, fout, indent=2)

                # Post-expander features are mostly bounded / z-score / log-return /
                # signed log1p, so use RobustScaler; keep PowerTransformer (Yeo-Johnson)
                # for everything else
                if train_model in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment'):
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))
                else:
                    scaler = PowerTransformer(method='yeo-johnson')
                label_cols = data.columns[-4:]
                data = data.astype({col: np.float64 for col in label_cols})
                # Remove duplicate index entries (keep last) so get_indexer works
                data = data[~data.index.duplicated(keep='last')]
                train_pos = data.index.get_indexer([pd.Timestamp(train_start)], method='nearest')[0]
                to_fit = data.iloc[max(1*250 - 1, train_pos - n_steps + 2):, :-4].loc[:train_end]
                sanitize_stats = fit_sanitize_statistics(to_fit)
                fit_features = apply_sanitize_statistics(to_fit, sanitize_stats)
                all_features = apply_sanitize_statistics(data.iloc[:, :-4], sanitize_stats)

                try:
                    scaler.fit(fit_features)
                    transformed = scaler.transform(all_features)
                except ValueError as scale_error:
                    print(f"[WARN] PowerTransformer failed, fallback to RobustScaler: {scale_error}")
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))
                    scaler.fit(fit_features)
                    transformed = scaler.transform(all_features)

                data = pd.concat([pd.DataFrame(
                    transformed,
                                 index=data.index, columns=data.columns[:-4]), data.iloc[:, -4:]], axis=1)
                preprocess_bundle = {
                    'scaler': scaler,
                    'sanitize_stats': sanitize_stats,
                    'feature_columns': data.columns[:-4].tolist(),
                    'version': 'v2_train_only_sanitize'
                }
                joblib.dump(preprocess_bundle, f"{SCALER_ROOT}/scaler_{train_model}_{symbol}.pkl")                 
            
            # ===================== 3) Build supervised windowed data =====================
            # prepare X, y windows================================================================================================================
            X, y = sequence_to_windows(seq=data.iloc[:, :-4], y=data[f"y_{forecast_days}"], n_steps=n_steps)
            train_slice = data.index.slice_indexer(start=train_start, end=train_end)
            test_slice = data.index.slice_indexer(start=test_start, end=test_end)
            X_train, y_train = get_windows(X, y, train_slice, n_steps)

            # Random validation ===================================================================================================================
            cv = build_validation_splitter()
            wf_folds = list(cv.split(X_train))
            if wf_folds:
                if VALIDATION_MODE in ('walk_forward_rolling', 'walk_forward_expanding'):
                    print(
                        f"[CV] {VALIDATION_MODE}: {len(wf_folds)} folds available, "
                        f"using last (most recent) fold for final train/val"
                    )
                train_indices, val_indices = wf_folds[-1]
                X_train, X_val = X_train[train_indices], X_train[val_indices]
                y_train, y_val = y_train[train_indices], y_train[val_indices]
                split_found = True
            else:
                split_found = False

            if not split_found:
                # Fallback: if blocking split produced no validation set, use a tail time-based split
                if X_train.shape[0] < 2:
                    raise ValueError("Not enough training samples to create validation split.")
                fallback_val_size = max(1, int(X_train.shape[0] * 0.2))
                fallback_val_size = min(fallback_val_size, X_train.shape[0] - 1)
                X_train, X_val = X_train[:-fallback_val_size], X_train[-fallback_val_size:]
                y_train, y_val = y_train[:-fallback_val_size], y_train[-fallback_val_size:]


            # ====================================================================== Sequential validation =================
            # X_train, y_train = get_windows(X, y, train_slice, n_steps)
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
            # =============================================================================================================''


            # Keep integer labels (matches SparseCategoricalCrossentropy)
            y_train_int = y_train.astype('int32')
            y_val_int = y_val.astype('int32')
            X_test, y_test = get_windows(X, y, test_slice, n_steps)

            batch_size = 120
            nb_epoch = 32
            repeats = total_repeats

            # Build train / val tf.data pipelines
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_int))
            ds_options = tf.data.Options()
            ds_options.experimental_deterministic = False
            train_dataset = train_dataset.with_options(ds_options).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_int))
            val_dataset = val_dataset.with_options(ds_options).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)



            # ===================== 4) Repeat training with the best params ======
            # Run n experiments using the best hyperparameters and save each
            # model plus its predictions
            experiment_dir.mkdir(parents=True, exist_ok=True)

            if completed_repeats > 0:
                print(f'[{train_model}_{symbol}] resume from repeat {completed_repeats + 1}/{repeats}')

            top_acc = _load_resume_scores(train_model, symbol, completed_repeats)
            if len(top_acc) != completed_repeats:
                print(
                    f'[{train_model}_{symbol}] resume score recovery incomplete: '
                    f'{len(top_acc)}/{completed_repeats}; pruning will be skipped if mismatch persists.'
                )
            max_acc = max(top_acc) if top_acc else 0
            skip_prune = len(top_acc) != completed_repeats and completed_repeats > 0


            # ============================================================= RAdam ==============================================================
                                         #<================================== tunable ==========
            '''learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 0.001, 
                decay_steps = total_steps * .7, 
                name = 'CosineDecay')'''
            # RAdam + Lookahead
            # Repeat training multiple times and average the results
            for r in range(completed_repeats, repeats):

                flooding_b = flooding_b_candidates[r % len(flooding_b_candidates)]

                print(f'{train_model}_{symbol} => Repeat {r+1}, flooding_b={flooding_b:.2f}')

                # Build the model with the AutoML best hyperparameters
                build_device = '/CPU:0' if FORCE_MODEL_BUILD_ON_CPU else '/GPU:0'
                try:
                    with tf.device(build_device):
                        model = build_model(
                            hp=best_hp_values,
                            input_shape=(X_train.shape[1], X_train.shape[2])
                        )
                except tf.errors.InternalError as build_error:
                    if 'CUDA_ERROR_INVALID_HANDLE' in str(build_error):
                        print('[WARN] GPU model build failed with CUDA_ERROR_INVALID_HANDLE, fallback to CPU build.')
                        with tf.device('/CPU:0'):
                            model = build_model(
                                hp=best_hp_values,
                                input_shape=(X_train.shape[1], X_train.shape[2])
                            )
                    else:
                        raise
                model.flooding_b = flooding_b

                # Fresh optimizer per repeat to avoid slot-variable conflicts
                if USE_TFA_OPTIMIZER:
                    radam = tfa.optimizers.RectifiedAdam(
                        learning_rate=0.001,
                        beta_1=0.95,
                        total_steps=1200,
                        epsilon=1e-07,
                        #weight_decay = 0.01,
                        amsgrad=False,
                        sma_threshold=5.0,
                        warmup_proportion=0.10,
                        min_lr=1e-5,
                        name='RectifiedAdam'
                    )
                    optimizer = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
                else:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                max_epochs = int(os.getenv('MAX_EPOCHS', '120'))

                metrics_list = [
                    tf.keras.metrics.Recall(class_id=1, name='recall'),
                    tf.keras.metrics.AUC(curve='ROC', name='auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc')
                ]
                if tfa is not None:
                    metrics_list.insert(
                        1,
                        tfa.metrics.FBetaScore(num_classes=2, beta=1.0, threshold=0.5, average='weighted')
                    )
                else:
                    metrics_list.insert(1, tf.keras.metrics.F1Score(name='f1', average='weighted', threshold=0.5))

                model.compile(
                    optimizer=optimizer,
                    loss=SparseCategoricalCrossentropy(from_logits=False),
                    metrics=metrics_list,
                )

                warmup_one_batch_from_dataset(model, train_dataset)

                history_obj = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=max_epochs,
                    verbose=FIT_VERBOSE,
                    callbacks=[
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
                        ReduceLROnPlateau(
                            monitor='val_recall',
                            mode='max',
                            factor=0.5,
                            patience=25,
                            min_delta=1e-7,
                            verbose=1
                        ),
                        EarlyStopping(
                            monitor='val_recall',
                            mode='max',
                            patience=65,
                            min_delta=1e-7,
                            restore_best_weights=True,
                            verbose=1
                        )
                    ]
                )
                history = history_obj.history
                #plot_history(history, out_path="history.png")

                mean_val_recall = np.mean(np.array(history.get('val_recall', [0.0])))
                top_acc.append(mean_val_recall)

                if mean_val_recall >= max_acc:
                    max_acc = mean_val_recall

                print(f'[{train_model}_{symbol}] flooding_b={flooding_b:.2f} Average val_recall: ', round(mean_val_recall, 4))
                print(f'[{train_model}_{symbol}] Best val_recall so far: ', round(max_acc, 4))

                y_pred = pd.Series(model.predict(X_test, batch_size=64, verbose=0)[:, 1], index=data[test_slice].index)
                #plot_prediction(symbol, y_pred, out_path="prediction.png")



                # Save the experimental models

                model.save(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_{r+1}.keras')
                y_pred = pd.Series(
                    model.predict(np.concatenate([X_train, X_test]), batch_size=64, verbose=0)[:, 1],
                    index=data.index[-(len(X_train) + len(X_test)):]
                )
                y_pred.to_csv(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_result_{r+1}.csv', header=False)

                tf.keras.backend.clear_session()
                #device = cuda.get_current_device()
                #device.reset()

        else: 
            # print(f'{train_model[-4:]} {train_model[4:-5]} exists')

            continue
        # To keep top 3 models and predictions, remove the rest
        if not skip_prune:
            sort_index = np.argsort(np.array(top_acc))
            for index in sort_index[:-3]:
                os.remove(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_{index+1}.keras')
                os.remove(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_result_{index+1}.csv')
        else:
            print(f'[{train_model}_{symbol}] skip pruning because resume score recovery was incomplete')
        tf.keras.backend.clear_session()
        gc.collect()
    
  