"""tw50_flood.py

Stage 1 of the TW-50 pipeline.

Bayesian hyperparameter search for the Attention (ATT) classifier with a
static Flooding regularizer. For each (stock_id, aspect) pair we:

    1. Load the aspect CSV from ./features/<aspect>_<stock_id>.csv.
    2. Apply the aspect-specific feature expander (macro / fundamental /
       tech_trend / trade / moment) that translates raw columns into
       stationary, scale-invariant features.
    3. Fit a sanitization / robust scaler on the training window only
       (no test leakage).
    4. Slide fixed-length lookback windows and split with the Walk-Forward
       rolling scheme (5 folds, 4:1 train:val, 10-day gap).
    5. Search the ATT hyperparameter space + flooding_b grid using
       Bayesian Optimization; report per-fold mean recall as the objective.
    6. Save the best trial hyperparameters (including flooding_b) to
       ./artifacts/flood/hyperbayes/ATT_<aspect>_<stock_id>/best_trial_summary.json.

Data window: training samples up to 2023-12-31; test is 2024-01-01 .. 2026-03-31.

Usage
-----
    python tw50_flood.py --stock-ids 2330 --aspect fundamental
    python tw50_flood.py --top50 --aspect all
    python tw50_flood.py --stock-ids 2330,2454 --aspect trade,tech_trend \\
        --trials 12 --epochs 80

Environment
-----------
    GPU is used when available; mixed precision is enabled by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

# Reduce TF/XLA log noise before importing tensorflow.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_USE_LEGACY_KERAS', '0')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from sklearn.metrics import fbeta_score, recall_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
import joblib

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# =============================================================================
# Constants
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = Path(os.environ.get('FEATURE_ROOT', REPO_ROOT / 'features'))
ARTIFACT_ROOT = Path(os.environ.get('MODEL_ROOT', REPO_ROOT / 'artifacts' / 'flood'))
HYPERBAYES_DIR = ARTIFACT_ROOT / 'hyperbayes'
SCALER_DIR = ARTIFACT_ROOT / 'feature_scaler'
FEAT_SEL_DIR = ARTIFACT_ROOT / 'feature_selection'
EXPERIMENT_DIR = ARTIFACT_ROOT / 'experiments'
for _p in (HYPERBAYES_DIR, SCALER_DIR, FEAT_SEL_DIR, EXPERIMENT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Fixed data window (see README).
TRAIN_START = '2010-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-03-31'

# Feature aspects (sentiment intentionally excluded).
# Feature aspect name mapping (see README "Feature aspects" table):
#   Fundamental -> 'fundamental'   Float       -> 'trade'
#   Price-Trend -> 'tech_trend'    Momentum    -> 'moment'
#   Macro       -> 'macro'
ASPECTS: tuple[str, ...] = ('fundamental', 'trade', 'tech_trend', 'moment', 'macro')

# Reference walk-forward: 20-day label isolation followed by a 30-day purge.
WF_N_SPLITS = int(os.getenv('WF_N_SPLITS', '5'))
WF_VAL_RATIO = float(os.getenv('WF_VAL_RATIO', '0.20'))
LABEL_HORIZON = 20
PURGE_GAP = 30
WF_GAP = int(os.getenv('WF_GAP', str(LABEL_HORIZON + PURGE_GAP)))

# Dynamic-flooding candidate grid: b in [0.0, 0.4], step 0.05 (9 values).
FLOODING_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(9))  # 0.00 .. 0.40

# Lookback windows the tuner may pick from.
LOOKBACK_CHOICES = (5, 10, 20, 30, 40, 60)


# =============================================================================
# Global seed
# =============================================================================

DEFAULT_SEED = int(os.getenv('DESQ_SEED', '42'))


def _set_global_seed(seed: int) -> None:
    """Set PYTHONHASHSEED, python `random`, NumPy, TensorFlow and enable op determinism."""
    import random as _random
    os.environ['PYTHONHASHSEED'] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception as err:
        print(f'[SEED] tf.keras.utils.set_random_seed failed: {err}')
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as err:
        print(f'[SEED] enable_op_determinism failed: {err}')
    print(f'[SEED] global seed set to {seed}')


# =============================================================================
# GPU setup
# =============================================================================


def configure_gpu() -> None:
    """Enable memory growth and mixed precision when a GPU is present."""
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as err:
            print(f'[GPU] set_memory_growth failed: {err}')
    try:
        mixed_precision.set_global_policy('mixed_float16')
    except Exception as err:
        print(f'[MP] mixed precision setup failed: {err}')
    print(f'[GPU] physical={len(gpus)}, mixed_precision=on')


# =============================================================================
# Feature expanders (translated verbatim from the finlab source)
# =============================================================================

TRADE_HEAVY_TAIL_COLS = (
    'foreign_cap_ratio', 'invst_cap_ratio', 'ins_nbd', 'Force_nbd', 'smr'
)

MACRO_STATIONARY_COLS = ('Price_rate_3m', 'Price_rate_10y', 'Price_FX', 'Price_VIX')
MACRO_PRICE_LEVEL_COLS = (
    'Price_oil', 'Price_gold', 'Price_copper',
    'Price_S&P500', 'Price_Nasdaq', 'Price_SOX',
    'Price_CRB', 'Price_BDI',
)
MACRO_VOL_COLS = ('Price_TX03C', 'Price_TX03P')
MACRO_SIGNED_LEVEL_COLS = ('Price_TX03F',)

FUNDAMENTAL_GROWTH_COLS = (
    'R_mom', 'R_yoy', 'R_acc_yoy',
    'E_qoq', 'E_yoy', 'E_acc_yoy',
    'Op_qoq', 'Op_yoy', 'Op_acc_yoy',
    'Gross_qoq', 'EPS_qoq',
)


def _prepare_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply signed log1p to heavy-tailed trade columns."""
    out = df.copy()
    for c in TRADE_HEAVY_TAIL_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors='coerce').astype(np.float64)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s))
    return out


def _detect_non_zero_date(df: pd.DataFrame, ratio_threshold: float = 0.0):
    """First index whose feature-non-zero ratio >= threshold."""
    feat = df.iloc[:, :-4]
    if ratio_threshold <= 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    else:
        idx = feat.index[(feat != 0).mean(axis=1) >= ratio_threshold]
    if len(idx) == 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    return idx[0] if len(idx) else feat.index[0]


def _expand_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert macro columns to near-stationary features (log-returns, z-scores)."""
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
        out[f'{c}_logret1'] = np.log(p / p.shift(1))
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
        out[f'{c}_log1p'] = lv
        out[f'{c}_diff20'] = lv - lv.shift(20)
    if 'Price_TX03P' in vol_logs and 'Price_TX03C' in vol_logs:
        out['tx03_pc_logratio'] = vol_logs['Price_TX03P'] - vol_logs['Price_TX03C']

    for c in MACRO_SIGNED_LEVEL_COLS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        sl = np.sign(s) * np.log1p(np.abs(s))
        out[f'{c}_slog1p'] = sl
        out[f'{c}_diff5'] = sl - sl.shift(5)
        out[f'{c}_diff20'] = sl - sl.shift(20)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


def _expand_tech_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw OHLCV into scale-invariant returns / ratios."""
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
        out['ret_1'] = (log_close - log_close.shift(1)).clip(-0.3, 0.3)
        out['ret_5'] = (log_close - log_close.shift(5)).clip(-0.5, 0.5)
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


def _expand_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bounded columns re-centered to ~[-1, 1]; growth-rate columns log-compressed."""
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
    """Center oscillators to [-1, 1]; log-compress heavy-tailed accumulators."""
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 0..100 oscillators -> [-1, 1].
    for c in ('rsi', 'k', 'd', 'adx'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 100.0)
            out[c] = (s - 50.0) / 50.0
    # Williams %R in [-100, 0] -> [-1, 1].
    if 'wr' in df.columns:
        s = pd.to_numeric(df['wr'], errors='coerce').clip(-100.0, 0.0)
        out['wr'] = (s + 50.0) / 50.0

    if 'cci' in df.columns:
        s = pd.to_numeric(df['cci'], errors='coerce').clip(-300.0, 300.0)
        out['cci'] = s / 100.0

    # Accumulators: signed log1p to compress magnitude.
    for c in ('acc_5', 'acc_10', 'acc_20', 'vpt'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s))

    if 'beta' in df.columns:
        s = pd.to_numeric(df['beta'], errors='coerce').clip(-3.0, 3.0)
        out['beta'] = s

    # Pass-through of remaining scale-safe columns not covered above.
    handled = set(out.columns)
    for c in df.columns[:-4]:
        if c in handled:
            continue
        out[c] = pd.to_numeric(df[c], errors='coerce')

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


EXPANDERS = {
    'macro': _expand_macro_features,
    'tech_trend': _expand_tech_trend_features,
    'fundamental': _expand_fundamental_features,
    'moment': _expand_moment_features,
    'trade': _prepare_trade_features,  # trade only needs heavy-tail damping.
}


# =============================================================================
# Loading / preprocessing
# =============================================================================


def load_aspect_csv(aspect: str, stock_id: str) -> pd.DataFrame:
    """Read one <aspect>_<stock_id>.csv into a datetime-indexed frame."""
    fp = FEATURE_ROOT / f'{aspect}_{stock_id}.csv'
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def fit_sanitize_statistics(train_df: pd.DataFrame, max_abs: float = 1e6, q: float = 0.001) -> dict:
    """Estimate sanitize stats on train-only data (no future leakage)."""
    numeric = train_df.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.apply(pd.to_numeric, errors='coerce').astype(np.float64)
    fill_values = numeric.median().fillna(0.0)
    filled = numeric.fillna(fill_values)
    lower = filled.quantile(q)
    upper = filled.quantile(1 - q)
    return {
        'fill_values': fill_values,
        'lower': lower,
        'upper': upper,
        'max_abs': float(max_abs),
    }


def apply_sanitize_statistics(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Fill NaN, clip to fitted quantiles, then signed log1p compress."""
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
    return pd.DataFrame(
        np.nan_to_num(
            sanitized.to_numpy(dtype=np.float64, copy=False),
            nan=0.0,
            posinf=np.log1p(max_abs),
            neginf=-np.log1p(max_abs),
        ),
        index=sanitized.index,
        columns=sanitized.columns,
    )


def preprocess_features(aspect: str, stock_id: str, save_scaler: bool = True) -> pd.DataFrame:
    """Expand + sanitize + robust scale, using training slice statistics only.

    Returns a frame whose first N columns are features and last 4 are labels
    (y_10, y_20, y_40, y_60).
    """
    raw = load_aspect_csv(aspect, stock_id)
    expander = EXPANDERS[aspect]
    if aspect in ('trade', 'fundamental'):
        expanded = expander(raw)
    else:
        expanded = expander(raw)

    labels = expanded.iloc[:, -4:]
    feats = expanded.iloc[:, :-4]

    start_idx = _detect_non_zero_date(expanded)
    feats = feats.loc[start_idx:]
    labels = labels.loc[start_idx:]

    train_slice = feats.loc[TRAIN_START:TRAIN_END]
    if train_slice.empty:
        train_slice = feats
    stats = fit_sanitize_statistics(train_slice)
    feats = apply_sanitize_statistics(feats, stats)

    # Fit robust scaler on train slice only.
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(feats.loc[TRAIN_START:TRAIN_END].to_numpy())
    scaled = scaler.transform(feats.to_numpy())
    feats = pd.DataFrame(scaled, index=feats.index, columns=feats.columns).clip(-8, 8)

    if save_scaler:
        joblib.dump(
            {'sanitize_stats': stats, 'scaler': scaler, 'columns': list(feats.columns)},
            SCALER_DIR / f'{aspect}_{stock_id}.joblib',
        )

    labels = labels.reindex(feats.index).ffill()
    labels = labels.astype(int).clip(0, 1)  # only need direction of y_20 during tuning
    return pd.concat([feats, labels], axis=1)


# =============================================================================
# Windowing + walk-forward split
# =============================================================================


def build_windows(df: pd.DataFrame, lookback: int, y_col: str = 'y_20'):
    """Return (X, y, dates_end) where dates_end aligns with the last row of each window."""
    n_features = df.shape[1] - 4
    feat = df.iloc[:, :-4].to_numpy(dtype=np.float64, copy=False)
    y = df[y_col].to_numpy(dtype=np.int32)
    n_windows = feat.shape[0] - lookback + 1
    if n_windows <= 0:
        raise ValueError(f'Not enough rows ({feat.shape[0]}) for lookback={lookback}.')
    shape = (n_windows, lookback, n_features)
    strides = (feat.strides[0],) + feat.strides
    X = np.lib.stride_tricks.as_strided(feat, shape=shape, strides=strides).copy()
    y_win = y[lookback - 1:]
    dates_end = df.index[lookback - 1:]
    return X, y_win, dates_end


def walk_forward_folds(n_samples: int, n_splits: int = WF_N_SPLITS,
                        val_ratio: float = WF_VAL_RATIO, gap: int = WF_GAP):
    """Yield (train_idx, val_idx) tuples for rolling-window CV.

    - Val blocks are equal size and non-overlapping, all placed at the tail.
    - Train window has fixed length (rolling); ends `gap` samples before its val.
    """
    if n_samples <= gap + 2:
        return
    val_size = max(1, int(n_samples * val_ratio / n_splits))
    total_val = val_size * n_splits
    if total_val + gap + 1 >= n_samples:
        val_size = max(1, (n_samples - gap - 2) // (n_splits + 1))
        total_val = val_size * n_splits

    first_val_start = n_samples - total_val
    first_train_end = first_val_start - gap
    rolling_train_len = max(1, first_train_end)

    for i in range(n_splits):
        val_start = n_samples - total_val + i * val_size
        val_end = val_start + val_size
        train_end = val_start - gap
        if train_end <= 0:
            continue
        train_start = max(0, train_end - rolling_train_len)
        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, min(val_end, n_samples))
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


# =============================================================================
# Model
# =============================================================================


class SinusoidalPositionalEncoding(layers.Layer):
    """Fixed sin/cos positional encoding; mixed-precision safe."""

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        d_model_f = tf.cast(d_model, tf.float32)
        pos = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
        dims = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
        angles = pos / tf.pow(10000.0, 2.0 * tf.math.floor(dims / 2.0) / d_model_f)
        sin_mask = tf.math.mod(tf.range(d_model), 2)
        cos_mask = 1 - sin_mask
        enc = (tf.sin(angles) * tf.cast(cos_mask, tf.float32)
               + tf.cos(angles) * tf.cast(sin_mask, tf.float32))
        enc = tf.cast(enc[tf.newaxis, :, :], x.dtype)
        return x + enc


class CausalMask(layers.Layer):
    """Lower-triangular mask so attention only sees past positions."""

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return tf.linalg.band_part(tf.ones((1, seq_len, seq_len), dtype=tf.float32), -1, 0)


def _mha_block(x, num_heads, key_dim, dropout_rate, prefix):
    mask = CausalMask(name=f'{prefix}_mask')(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate,
        name=f'{prefix}_mha',
    )(x, x, attention_mask=mask)
    attn = layers.Dropout(dropout_rate, name=f'{prefix}_drop')(attn)
    x = layers.Add(name=f'{prefix}_add')([x, attn])
    return layers.LayerNormalization(name=f'{prefix}_ln')(x)


class FloodingModel(keras.Model):
    """Keras model that applies flooding regularization to the training loss."""

    flooding_b = 0.10

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight,
                                       regularization_losses=self.losses)
            if self.flooding_b > 0:
                loss = tf.abs(loss - self.flooding_b) + self.flooding_b
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}


class HyperATT(kt.HyperModel):
    """ATT hyper-model: positional encoding + causal MHA stack + dense head."""

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp: kt.HyperParameters):
        inputs = keras.Input(shape=self.input_shape, name='inputs')
        x = SinusoidalPositionalEncoding(name='pos')(inputs)

        n_layers = hp.Int('attn_layers', 1, 3, step=1)
        heads = [hp.Int(f'attn_heads_{i+1}', 2, 3, step=1) for i in range(3)]
        key_dims = [hp.Int(f'attn_key_dim_{i+1}', 16, 32, step=16) for i in range(3)]
        drops = [hp.Float(f'attn_dropout_{i+1}', 0.0, 0.2, step=0.1) for i in range(3)]

        for i in range(n_layers):
            x = _mha_block(x, heads[i], key_dims[i], drops[i], prefix=f'a{i+1}')

        x = layers.GlobalAveragePooling1D(name='pool')(x)
        x = layers.LayerNormalization(name='pool_ln')(x)
        x = layers.Dense(
            hp.Int('dense_units', 64, 192, step=32),
            kernel_initializer=hp.Choice('dense_kernel_1',
                                          ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
            activation=hp.Choice('activation', ['relu', 'elu', 'selu', 'tanh', 'swish']),
            name='dense',
        )(x)
        x = layers.LayerNormalization(name='dense_ln')(x)
        logits = layers.Dense(
            2,
            kernel_initializer=hp.Choice('dense_kernel_2',
                                          ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
            name='logits',
        )(x)
        temp = hp.Float('temperature', 0.3, 1.0, step=0.1)
        logits = layers.Lambda(lambda t: t / temp, name='temp')(logits)
        outputs = layers.Activation('softmax', dtype='float32', name='softmax')(logits)

        model = FloodingModel(inputs, outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=[
                keras.metrics.Recall(class_id=1, name='recall'),
                keras.metrics.AUC(curve='ROC', name='auc'),
                keras.metrics.AUC(curve='PR', name='pr_auc'),
            ],
        )
        return model


# =============================================================================
# Tuner
# =============================================================================


class TunerCV(kt.Tuner):
    """Bayesian Tuner that scores each trial by mean recall over WF folds."""

    def run_trial(self, trial, X_by_lookback, y_by_lookback, batch_size=64, epochs=60):
        lookback = trial.hyperparameters.Choice('lookback_window', values=list(LOOKBACK_CHOICES))
        if lookback not in X_by_lookback:
            lookback = min(X_by_lookback.keys())
        X, y_onehot = X_by_lookback[lookback], y_by_lookback[lookback]

        # Flooding grid cycled by trial index.
        try:
            trial_idx = int(trial.trial_id)
        except (TypeError, ValueError):
            trial_idx = 0
        flooding_b = FLOODING_GRID[trial_idx % len(FLOODING_GRID)]

        fold_recalls: list[float] = []
        fold_fbetas: list[float] = []
        started_at = time.time()

        for fold_idx, (tr_idx, va_idx) in enumerate(
            walk_forward_folds(len(X)), start=1
        ):
            x_tr, x_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y_onehot[tr_idx], y_onehot[va_idx]

            self.hypermodel.input_shape = (x_tr.shape[1], x_tr.shape[2])
            model = self.hypermodel.build(trial.hyperparameters)
            model.flooding_b = float(flooding_b)

            # Class weights and monotonic validation weights.
            labels_int = np.argmax(y_tr, axis=1)
            uniq, counts = np.unique(labels_int, return_counts=True)
            cw = {int(u): float((1.0 / c) * len(labels_int)) for u, c in zip(uniq, counts)}

            val_w = np.linspace(0.2, 0.6, len(y_va))

            model.fit(
                x_tr, y_tr,
                batch_size=batch_size, epochs=epochs, verbose=0, shuffle=False,
                validation_data=(x_va, y_va, val_w),
                class_weight=cw,
                callbacks=[
                    ReduceLROnPlateau(monitor='val_recall', mode='max',
                                       factor=0.5, patience=15, min_delta=1e-6, verbose=0),
                    EarlyStopping(monitor='val_recall', mode='max',
                                   patience=30, restore_best_weights=True, verbose=0),
                ],
            )
            preds = model.predict(x_va, batch_size=batch_size, verbose=0)
            y_hat = (preds[:, 1] >= 0.5).astype(int)
            fold_recalls.append(float(recall_score(y_va[:, 1], y_hat, zero_division=0)))
            fold_fbetas.append(float(fbeta_score(y_va[:, 1], y_hat, beta=1.0, zero_division=0)))
            K.clear_session()
            gc.collect()

        mean_recall = float(np.mean(fold_recalls)) if fold_recalls else 0.0
        mean_fbeta = float(np.mean(fold_fbetas)) if fold_fbetas else 0.0
        wall = time.time() - started_at
        print(
            f'[Trial {trial.trial_id}] lookback={lookback}, flooding_b={flooding_b:.2f}, '
            f'recall={mean_recall:.4f}, fbeta={mean_fbeta:.4f}, wall={wall:.1f}s'
        )
        self.oracle.update_trial(
            trial.trial_id,
            {'val_recall_score': mean_recall, 'val_fbeta_score': mean_fbeta},
        )


# =============================================================================
# Pipeline
# =============================================================================


def prepare_windowed_data(processed: pd.DataFrame):
    """Slice train part, then window it across all lookback choices."""
    train_df = processed.loc[TRAIN_START:TRAIN_END]
    if len(train_df) < max(LOOKBACK_CHOICES) + 100:
        raise RuntimeError(
            f'Training window too short: {len(train_df)} rows for '
            f'lookback max {max(LOOKBACK_CHOICES)}.'
        )
    X_by_lookback = {}
    y_by_lookback = {}
    for lb in LOOKBACK_CHOICES:
        X, y, _ = build_windows(train_df, lookback=lb)
        X_by_lookback[lb] = X.astype(np.float32)
        y_by_lookback[lb] = to_categorical(y, num_classes=2).astype(np.float32)
    return X_by_lookback, y_by_lookback


def run_search(aspect: str, stock_id: str, trials: int, epochs: int, batch_size: int,
               seed: int = DEFAULT_SEED) -> dict:
    print(f'\n=== FLOOD SEARCH: stock={stock_id}, aspect={aspect} ===')
    processed = preprocess_features(aspect, stock_id)
    X_by_lookback, y_by_lookback = prepare_windowed_data(processed)

    dummy_shape = (max(LOOKBACK_CHOICES), X_by_lookback[max(LOOKBACK_CHOICES)].shape[2])
    hypermodel = HyperATT(input_shape=dummy_shape)
    project_name = f'ATT_{aspect}_{stock_id}'
    tuner_dir = HYPERBAYES_DIR

    tuner = TunerCV(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective('val_recall_score', direction='max'),
            max_trials=trials,
            seed=seed,
        ),
        hypermodel=hypermodel,
        directory=str(tuner_dir),
        project_name=project_name,
        overwrite=True,
    )
    tuner.search(
        X_by_lookback, y_by_lookback,
        batch_size=batch_size, epochs=epochs,
    )
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_hp = dict(best_trial.hyperparameters.values)
    best_lookback = best_hp.get('lookback_window', LOOKBACK_CHOICES[0])
    best_flooding_b = FLOODING_GRID[int(best_trial.trial_id) % len(FLOODING_GRID)]

    summary = {
        'stock_id': stock_id,
        'aspect': aspect,
        'best_trial_id': best_trial.trial_id,
        'score': best_trial.score,
        'lookback_window': best_lookback,
        'flooding_b': best_flooding_b,
        'hyperparameters': best_hp,
        'train_window': [TRAIN_START, TRAIN_END],
        'wf_config': {'n_splits': WF_N_SPLITS, 'val_ratio': WF_VAL_RATIO, 'gap': WF_GAP},
        'flooding_grid': list(FLOODING_GRID),
    }
    out_dir = tuner_dir / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'best_trial_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f'[BEST] trial={best_trial.trial_id}, score={best_trial.score}, '
          f'flooding_b={best_flooding_b}, lookback={best_lookback}')
    return summary


# =============================================================================
# CLI
# =============================================================================


def load_top50_ids() -> list[str]:
    fp = REPO_ROOT / 'tw50_top50.csv'
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp, dtype={'stock_id': str})['stock_id'].tolist()


def parse_stock_ids(arg_stock_ids: str | None, use_top50: bool) -> list[str]:
    if use_top50:
        return load_top50_ids()
    if not arg_stock_ids:
        raise SystemExit('Provide --stock-ids or --top50.')
    return [s.strip() for s in arg_stock_ids.split(',') if s.strip()]


def parse_aspects(arg_aspect: str) -> list[str]:
    if arg_aspect == 'all':
        return list(ASPECTS)
    aspects = [a.strip() for a in arg_aspect.split(',') if a.strip()]
    for a in aspects:
        if a not in ASPECTS:
            raise SystemExit(f'Unknown aspect: {a} (valid: {ASPECTS})')
    return aspects


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', help='comma-separated list, e.g. 2330,2454')
    p.add_argument('--top50', action='store_true', help='use all stocks from tw50_top50.csv')
    p.add_argument('--aspect', default='all',
                   help=f'comma-separated aspects or "all" (default: all). Valid: {ASPECTS}')
    p.add_argument('--trials', type=int, default=12)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED,
                   help='Global RNG seed for PYTHONHASHSEED/random/numpy/tf and BayesianOptimizationOracle.')
    args = p.parse_args(argv)

    _set_global_seed(args.seed)
    configure_gpu()
    stock_ids = parse_stock_ids(args.stock_ids, args.top50)
    aspects = parse_aspects(args.aspect)
    print(f'[PLAN] stocks={stock_ids}, aspects={aspects}, trials={args.trials}, epochs={args.epochs}')

    for sid in stock_ids:
        for aspect in aspects:
            try:
                run_search(aspect, sid, args.trials, args.epochs, args.batch_size,
                           seed=args.seed)
            except Exception as exc:  # noqa: BLE001
                print(f'[FAIL] {sid}/{aspect}: {exc}')
                import traceback
                traceback.print_exc()
    return 0


if __name__ == '__main__':
    sys.exit(main())
