import os, json, gc, sys

"""ATT 固定超參數訓練腳本（Dynamic Flooding + 因果注意力）。

此檔案與 Keras Tuner 版的差異：
- 不執行 trial 搜尋，而是讀取既有最佳參數後重複訓練
- 著重在穩定重跑、保存多個實驗模型與預測輸出
- 適合作為量產/批次更新流程
"""

# 降低 TensorFlow/XLA 日誌噪音與首次編譯卡頓（需在 import tensorflow 前設定）
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

# 若環境無法使用 tensorflow-addons（例如較新的 TF/Keras 組合），自動回退到原生 Adam
USE_TFA_OPTIMIZER = (os.getenv('ENABLE_TFA_OPTIMIZER', '0') == '1') and (tfa is not None)
FORCE_MODEL_BUILD_ON_CPU = os.getenv('FORCE_MODEL_BUILD_ON_CPU', '0') == '1'
ENABLE_XLA = os.getenv('ENABLE_XLA', '0') == '1'
ENABLE_TF32 = os.getenv('ENABLE_TF32', '1') == '1'
USE_MIXED_PRECISION = os.getenv('ENABLE_MIXED_PRECISION', '1') == '1'
GPU_MEMORY_LIMIT_MB = int(os.getenv('GPU_MEMORY_LIMIT_MB', '0'))
FIT_VERBOSE = int(os.getenv('FIT_VERBOSE', '1'))

def platform_path(path_str):
    """將路徑字串正規化為當前執行環境可用格式。

    參數:
        path_str: 原始路徑（可能為 Windows `D:/...` 格式）。

    回傳:
        Windows 下回傳原值；Linux/WSL 下回傳 `/mnt/<drive>/...`。

    副作用:
        無。
    """
    if os.name != 'nt' and len(path_str) >= 2 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path_str

DATA_ROOT = platform_path('D:/Feature_new')
HYPER_ROOT = platform_path(os.getenv('HYPER_ROOT', 'D:/hyperbayes_ATT'))
FEATURE_SELECTION_ROOT = platform_path('D:/feature_selection_ATT')
SCALER_ROOT = platform_path('D:/feature_scaler_ATT')
EXPERIMENT_ROOT = platform_path('D:/experiments_ATT')

# ===================== Flooding 消融實驗參數（env 控制；預設=正式行為） =====================
# FLOOD_MODE: 'none'    -> 不使用 flooding（flooding_b=0，不掛 DynamicFloodingCallback）
#             'static'  -> 固定水位 flooding_b=STATIC_FLOOD_B（不掛 DynamicFloodingCallback）
#             'dynamic' -> 現行 dynamic flooding（初始 b 輪替 + DynamicFloodingCallback）
FLOOD_MODE = os.getenv('FLOOD_MODE', 'dynamic').strip().lower()
if FLOOD_MODE not in {'none', 'static', 'dynamic'}:
    print(f"[WARN] Unknown FLOOD_MODE={FLOOD_MODE!r}, fallback to 'dynamic'.")
    FLOOD_MODE = 'dynamic'
STATIC_FLOOD_B = float(os.getenv('STATIC_FLOOD_B', '0.3'))
# NUM_REPEATS / MAX_EPOCHS：0 代表沿用腳本內預設值
NUM_REPEATS = int(os.getenv('NUM_REPEATS', '0'))
MAX_EPOCHS_ENV = int(os.getenv('MAX_EPOCHS', '0'))
DISABLE_EARLY_STOPPING = os.getenv('DISABLE_EARLY_STOPPING', '0') == '1'
# Date range env vars (combined-5 v2 production split, see ATT+Flood_combined.py)
TRAIN_END_DATE = os.getenv('TRAIN_END_DATE', '2023-12-31')
TEST_START_DATE = os.getenv('TEST_START_DATE', '2024-01-01')
TEST_END_DATE = os.getenv('TEST_END_DATE', '2026-06-03')
EXPERIMENT_OUTPUT_DIR = platform_path(os.getenv('EXPERIMENT_OUTPUT_DIR', 'D:/experiment_flood'))
EVAL_PLOT_DIR = platform_path(os.getenv('EVAL_PLOT_DIR', 'D:/evaluation_plot'))
# 實驗輸出統一導向 experiment_flood/<flood_mode>，避免三組互相覆蓋
EXPERIMENT_ROOT = f'{EXPERIMENT_OUTPUT_DIR}/{FLOOD_MODE}'
print(
    f"[FLOOD-EXP] FLOOD_MODE={FLOOD_MODE} static_b={STATIC_FLOOD_B} "
    f"num_repeats_override={NUM_REPEATS or 'default'} max_epochs_override={MAX_EPOCHS_ENV or 'default'} "
    f"disable_early_stopping={DISABLE_EARLY_STOPPING}\n"
    f"[FLOOD-EXP] experiment_root={EXPERIMENT_ROOT}\n"
    f"[FLOOD-EXP] eval_plot_dir={EVAL_PLOT_DIR}"
)

# Walk-Forward Validation options
# VALIDATION_MODE: 'blocking' (default, single block CV — prior behavior),
#                  'walk_forward_rolling'   — fixed-size train window slides forward,
#                  'walk_forward_expanding' — train start fixed, train window grows.
# Dflooding 為「最終訓練」階段，會選用最後（最近）一個 walk-forward fold 作為 train/val
# 切分以最大限度逼近實盤推論前的分布。


def _normalize_validation_mode(raw_mode: str) -> str:
    """將 validation mode 別名正規化為內部使用值。"""
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
    """決定 validation mode。

    解析順序：
      1) 環境變數 `VALIDATION_MODE`（支援 traditional/blocking/rolling/expanding 別名）。
      2) 若未設定且為互動式 TTY：詢問使用者（預設 rolling）。
      3) 否則預設 `blocking`（保留舊行為）。
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

    print("請選擇 validation mode:")
    print("  1) traditional (blocking)")
    print("  2) walk-forward expanding")
    print("  3) walk-forward rolling [default]")
    try:
        answer = input("請輸入 1/2/3（Enter=3）: ").strip().lower()
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
WF_GAP = int(os.getenv('WF_GAP', '10'))
WF_VAL_YEARS = float(os.getenv('WF_VAL_YEARS', '0'))
WF_TRADING_DAYS_PER_YEAR = int(os.getenv('WF_TRADING_DAYS_PER_YEAR', '252'))
WF_VAL_SAMPLES = int(round(WF_VAL_YEARS * WF_TRADING_DAYS_PER_YEAR)) if WF_VAL_YEARS > 0 else 0

Path(FEATURE_SELECTION_ROOT).mkdir(parents=True, exist_ok=True)
Path(SCALER_ROOT).mkdir(parents=True, exist_ok=True)
Path(EXPERIMENT_ROOT).mkdir(parents=True, exist_ok=True)


# trade CSV 裡重尾特徵（pct_change / abs(net / MA20) 等可能爆量）；進 sanitize
# 之前先 signed log1p，避免 Yeo-Johnson 與 quantile 被極端尾部帶偏。
TRADE_HEAVY_TAIL_COLS = (
    'foreign_cap_ratio', 'invst_cap_ratio', 'ins_nbd', 'Force_nbd', 'smr'
)


def _prepare_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """對 trade CSV 的重尾欄位做 signed log1p，其餘欄位保持原樣。"""
    out = df.copy()
    for c in TRADE_HEAVY_TAIL_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors='coerce').astype(np.float64)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s))
    return out


def _detect_non_zero_date(df: pd.DataFrame, ratio_threshold: float = 0.0):
    """回傳首個「非零欄位比例達 ratio_threshold」的日期。
    ratio_threshold=0 代表只要有任一欄非零（舊行為）。"""
    feat = df.iloc[:, :-4]
    if ratio_threshold <= 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    else:
        idx = feat.index[(feat != 0).mean(axis=1) >= ratio_threshold]
    if len(idx) == 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    return idx[0] if len(idx) else feat.index[0]


def _expand_sentiment_features(df: pd.DataFrame, stock_id) -> pd.DataFrame:
    """將 sentiment CSV（US / TW / ticker 三欄 0-100 分數）展開為衍生特徵。

    輸入 df：index=date，欄位包含 'US_sentiment_score','TW_sentiment_score',
    str(stock_id)，最後 4 欄為 y_10..y_60 標籤。
    回傳：concat(衍生特徵, 原最後 4 欄標籤)。後續共通 preprocess (corr 過濾 /
    scaler / sanitize) 會在本函式輸出之上再處理。
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


# macro 欄位分組（詳見 ATT+Flood.py 中 _expand_macro_features 的說明）
MACRO_STATIONARY_COLS = ('Price_rate_3m', 'Price_rate_10y', 'Price_FX', 'Price_VIX')
MACRO_PRICE_LEVEL_COLS = (
    'Price_oil', 'Price_gold', 'Price_copper',
    'Price_S&P500', 'Price_Nasdaq', 'Price_SOX',
    'Price_CRB', 'Price_BDI',
)
MACRO_VOL_COLS = ('Price_TX03C', 'Price_TX03P')
MACRO_SIGNED_LEVEL_COLS = ('Price_TX03F',)


def _expand_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """將 macro CSV 轉為近平穩特徵：非平穩價格 → log-return / z-score；
    有界欄位保留 level；選擇權量 → log1p + diff20；期貨淨額 → signed log1p。

    0 在 macro CSV 裡屬缺值（假日/資料缺漏），先 replace(0→NaN).ffill() 再轉換。
    必須與 ATT+Flood.py / prediction_update_tony_2026 .py 的版本完全一致。
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
# tech_trend / fundamental / moment expander（詳見 ATT+Flood.py 同名函式的說明）
# 必須與 ATT+Flood.py / prediction_update_tony_2026 .py 的實作完全一致。
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


# 預設啟用 XLA JIT 提升訓練速度；若不穩定可用 DISABLE_XLA=1 關閉
tf.config.optimizer.set_jit(ENABLE_XLA)
DEFAULT_LOOKBACK_WINDOW = 20

# 預設啟用 mixed precision 以加快訓練；若發生 NaN/Inf 可用 DISABLE_MIXED_PRECISION=1 關閉。
if USE_MIXED_PRECISION:
    try:
        mixed_precision.set_global_policy('mixed_float16')
    except Exception as e:
        print(f"[WARN] mixed precision setup failed: {e}")

def add_sinusoidal_positional_encoding(x, seq_len, feature_dim):
    """建立固定式 sin/cos 位置編碼並加到輸入張量。

    參數:
        x: 輸入張量。
        seq_len: 序列長度。
        feature_dim: 特徵維度。

    回傳:
        加入位置編碼後的張量。

    副作用:
        無。
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
            # 啟用 memory growth，避免一次佔滿顯存
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
    """從 dataset 取 1 個 batch 暖機，前移首次編譯延遲。

    參數:
        model: 已 compile 的 Keras 模型。
        train_dataset: `tf.data.Dataset` 訓練資料管線。

    回傳:
        無。

    副作用:
        觸發一次評估/推論圖編譯，不更新模型權重。
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
    """將連續序列切為固定長度滑動視窗。

    參數:
        seq: 特徵序列 DataFrame。
        y: 標籤序列 Series/array。
        n_steps: 視窗長度。

    回傳:
        (_X, _y)
        - _X: shape=(樣本數, n_steps, 特徵數)
        - _y: shape=(樣本數,)

    副作用:
        無。
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
    """依索引區間擷取窗口資料並套用最小歷史長度保護。

    參數:
        X: 已切窗特徵陣列。
        y: 已切窗標籤陣列。
        slice: 日期索引切片結果（含 start/stop）。
        steps: 視窗長度。

    回傳:
        (X_slice, y_slice) 子集。

    副作用:
        無。
    """
    min_idx = 1 * 250 - 1
    start_idx, end_idx, _ = slice.start, slice.stop, slice.step
    start_idx = max(min_idx, start_idx - steps + 1)
    end_idx = end_idx - steps + 1
    return X[start_idx:end_idx], y[start_idx:end_idx]


def window_indices_to_dates(window_indices, slc, n_steps_, data_index):
    """將 `get_windows` 輸出空間內的窗口索引換算回 `data_index` 上的日期。

    `get_windows` 內部 start_idx = max(min_idx, slc.start - n_steps + 1)；
    第 j 個輸出窗口對應原 X 的窗口 (start_idx + j)，再 +n_steps-1 即為窗口尾日。
    """
    min_idx = 1 * 250 - 1
    base = max(min_idx, slc.start - n_steps_ + 1)
    arr = np.asarray(window_indices, dtype=int)
    return data_index[base + arr + n_steps_ - 1]

def val_windows(data, ref_day=60, period=20):
    """將資料轉為監督式學習窗口，最後 4 欄視為標籤/保留欄。

    參數:
        data: 原始 DataFrame。
        ref_day: 回看視窗長度。
        period: 標籤欄位後綴，讀取 `y_{period}`。

    回傳:
        (X_val, y_val) 視窗化結果。

    副作用:
        無。
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
    """依日期切資料，並保留最小歷史區段以提升特徵穩定性。

    參數:
        X: 視窗化特徵陣列。
        y: 視窗化標籤陣列。
        idx: 原始日期索引。
        start: 起始日期字串。
        end: 結束日期字串。
        ref_day: 視窗長度。

    回傳:
        (X_slice, y_slice) 且轉為 Keras floatx。

    副作用:
        無。
    """
    start_idx = idx.get_loc(idx.to_series()[start:].iloc[0])
    end_idx = idx.get_loc(idx.to_series()[:end].iloc[-1])
    st = max(1*250-1, start_idx-ref_day+1)
    ed = end_idx-ref_day+2
    return K.cast_to_floatx(X[st:ed]), K.cast_to_floatx(y[st:ed])

class BlockingTimeSeriesSplit:
    """時間序列分割器（Blocking CV）。

    用途:
        維持時間順序切分資料，並在 train/val 間加上 gap 降低洩漏風險。

    參數:
        n_splits: 分割數。
        val_ratio: 每個 block 的驗證比例。
        gap: train 與 val 的間隔樣本數。

    屬性:
        n_splits, val_ratio, gap。

    副作用:
        無。
    """
    def __init__(self, n_splits, val_ratio=0.25, gap=10):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.gap = gap
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    # 依時間切分，訓練與驗證中間留 gap 天避免資料洩漏
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
    """Walk-Forward Validation 切分器（支援 rolling / expanding）。

    用途:
        模擬「以過去訓練、在緊接著的未來驗證」的滾動評估模式，
        以避免前瞻偏差並捕捉機制轉變。

    參數:
        n_splits: 切分折數，每折一個驗證區間。
        val_ratio: 每折驗證區間佔全部資料的比例（預設 0.2）。
        val_samples: 每折驗證樣本數（>0 時優先於 val_ratio，可用於固定年數）。
        gap: train 與 val 之間留出的樣本數，避免 label leakage。
        mode: 'rolling' 或 'expanding'。

    屬性:
        n_splits, val_ratio, gap, mode。

    副作用:
        無。
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
    """依 `VALIDATION_MODE` 環境變數選擇指定的時間序列切分器。"""
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
    """套用 Flooding 訓練策略的自訂 Keras 模型。

    用途:
        覆寫 `train_step`，將 batch loss 轉為 flooding loss 以抑制過度擬合。

    參數:
        與 `keras.Model` 建構參數一致（由 Functional API 建立）。

    屬性:
        flooding_b: Flooding 水位（float，預設 0.10）。

    副作用:
        改變訓練 loss 計算方式。
    """
    # 固定 flooding 水位；可依需要調整
    flooding_b = 0.10

    def train_step(self, data):
        """執行單一訓練步驟並套用 flooding loss。

        用途:
            取代預設 `Model.train_step`，在每個 batch 內改寫 loss 計算。

        參數:
            data: Keras batch 輸入，格式可為 `(x, y)` 或 `(x, y, sample_weight)`。

        回傳:
            dict，鍵為 metric 名稱、值為當前 metric 結果。

        副作用:
            會更新模型權重與 metric 狀態。
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
        # FBetaScore / F1Score 需要 one-hot 2D y_true；先將整數標籤轉換
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
    """依驗證指標動態調整 `flooding_b` 的 callback。

    用途:
        每個 epoch 根據監控指標改善與否，調整 flooding 水位。

    參數:
        monitor: 監控指標名稱。
        min_b/max_b: flooding_b 上下界。
        step_up/step_down: 調整步長。
        patience: 未改善容忍 epoch 數。
        min_delta: 判定改善的最小幅度。
        verbose: 是否輸出調整資訊。

    屬性:
        best, wait 與調整相關控制參數。

    副作用:
        會直接修改 `self.model.flooding_b`。
    """
    # 根據 val_recall 表現動態調整 flooding_b
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


class FloodingLevelLogger(tf.keras.callbacks.Callback):
    """將每個 epoch 結束時的 `model.flooding_b` 記錄到 logs（供 history 保存）。

    放在 callbacks 清單最後，確保讀到的是 DynamicFloodingCallback 調整後的值；
    none/static 模式下則記錄固定水位，使三組 history 都有一致的 `flooding_b` 欄位。
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs['flooding_b'] = float(getattr(self.model, 'flooding_b', 0.0))


def select_uncorrelated_features(feature_df, cutoff=0.85):
    """以相關係數門檻保留低共線性特徵。"""
    corr_matrix = feature_df.corr()
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [var for var in corr_matrix.columns if any(corr_matrix[var] > cutoff)]
    return corr_matrix.drop(drop_cols, axis=1).columns.tolist()


def fit_sanitize_statistics(df, max_abs=1e6, q=0.001):
    """以訓練資料估計 sanitize 所需統計量，避免未來資料洩漏。"""
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
    """套用固定 sanitize 統計量，確保訓練/推論一致。"""
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
    """向後相容：單次資料 sanitize（統計量由同一批資料估計）。"""
    stats = fit_sanitize_statistics(df, max_abs=max_abs, q=q)
    return apply_sanitize_statistics(df, stats)

# ===================== 學術專業繪圖風格 =====================
# 色盲友善固定配色：三種 flooding 模式各一色，跨圖一致
MODE_COLORS = {
    'none': '#4C72B0',     # blue
    'static': '#DD8452',   # orange
    'dynamic': '#55A868',  # green
}
MODE_LABELS = {
    'none': 'No flooding',
    'static': 'Static flooding',
    'dynamic': 'Dynamic flooding',
}


def setup_academic_style():
    """設定 matplotlib 全域學術風格（serif、淺格線、無上/右框線、高解析度）。"""
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


setup_academic_style()


def plot_history(history, out_path, flood_mode='dynamic', static_b=0.2, title=None):
    """繪製單次訓練的 Accuracy 與 Loss(+flooding b) 曲線（學術風格）。

    參數:
        history: keras History.history dict（含 accuracy/val_accuracy/loss/val_loss，
                 可選 ce(原始訓練 CE) 與 flooding_b）。
        out_path: 圖檔輸出路徑。
        flood_mode: 'none'/'static'/'dynamic'，決定 b 水位畫法。
        static_b: static 模式的固定 b 值。
        title: 圖標題（選填）。
    """
    epochs = np.arange(1, len(history.get('loss', [])) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)

    # --- (1) Accuracy ---
    if 'accuracy' in history:
        ax[0].plot(epochs, history['accuracy'], color='#4C72B0', label='Training')
    if 'val_accuracy' in history:
        ax[0].plot(epochs, history['val_accuracy'], color='#DD8452', label='Validation')
    ax[0].set_title('Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')

    # --- (2) Loss ---
    if 'ce' in history:
        ax[1].plot(epochs, history['ce'], color='#4C72B0', linestyle='-', label='Training loss')
    elif 'loss' in history:
        ax[1].plot(epochs, history['loss'], color='#4C72B0', linestyle='-', label='Training loss')
    if 'val_loss' in history:
        ax[1].plot(epochs, history['val_loss'], color='#DD8452', linestyle='-', label='Validation loss')

    ax[1].set_title('Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='upper right')

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)

def plot_prediction(symbol, prediction, out_path):
    import pyodbc
    conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER=data.autoquant.ai,3333;DATABASE=AutoQuant;UID=aq;PWD=2020@autoquant;MARS_Connection=Yes")
    actual_price = pd.read_sql(f"SELECT 日期, 收盤價 FROM sysdbase WHERE 股票代號 = '{symbol}' AND 日期 BETWEEN '{prediction.index.min().strftime('%Y-%m-%d')}' AND '{prediction.index.max().strftime('%Y-%m-%d')}' ORDER BY 日期 ASC", conn, index_col='日期', parse_dates=True).iloc[:, 0]
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
    """建立因果式多頭注意力區塊（動態長度 mask 版本）。

    參數:
        x: 輸入張量。
        num_heads: 注意力頭數。
        key_dim: key 維度。
        dropout_rate: dropout 比例。
        name_prefix: 層命名前綴。
        seq_len: 序列長度（用於建立初始 mask）。

    回傳:
        區塊輸出張量。

    副作用:
        無。
    """
    # 動態生成因果 mask，避免在圖模式下固定 tensor 導致的相容性問題
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
    """將外部載入的 hp 字典正規化為模型可直接使用格式。

    參數:
        hp: 可能包含別名鍵、字串數值的超參數字典。

    回傳:
        具完整預設值且型別正確的超參數字典。

    副作用:
        無。
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
    """依正規化超參數建立 FloodingModel。

    參數:
        hp: 超參數字典（可含原始/別名鍵）。
        input_shape: 輸入 shape（time, features）。

    回傳:
        尚未 compile 的 `FloodingModel` 實例。

    副作用:
        無。
    """
    # 建立與 ATT+Flood AutoML 相同的架構：PositionEmbedding + Nx Attention + FC
    hp = normalize_hyperparameters(hp)
    inputs = keras.Input(shape=input_shape, name="inputs")

    # 使用無權重 sinusoidal 位置編碼，避免第三方 PositionEmbedding 在此環境觸發 CUDA cast 錯誤
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

    # 全連接層做特徵融合
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
    """讀取 ATT+Flood 的最佳參數（優先 best_trial_summary，再 fallback trial.json）。"""
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
    """依模型類型與股票代碼準備訓練/測試資料。

    參數:
        m: 模型類型（如 macro、fundamental）。
        n: 股票代碼。

    回傳:
        目前流程主要在函式內完成前處理與切分；回傳依原實作為準。

    副作用:
        讀取檔案、進行縮放與資料轉換。
    """
    # 依股票與模型類型準備訓練/測試資料
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
    # Combined-5 v2 split (env-driven; see TRAIN_END_DATE / TEST_START_DATE / TEST_END_DATE)
    val_end = TRAIN_END_DATE
    train_end = TRAIN_END_DATE
    test_start = TEST_START_DATE
    data_max = X_y_all.index.max().strftime('%Y-%m-%d')
    test_end = min(TEST_END_DATE, data_max)
    print(f"[SPLIT] train_start={train_start}  train/val_end={train_end}  test={test_start}~{test_end}")

    # Feature selection and normalization =======================================================================================================================
    train_feature_slice = X_y_all.loc[:train_end].iloc[:, :-4]
    selected_feature_cols = select_uncorrelated_features(train_feature_slice, cutoff=0.85)
    X_y_all = X_y_all[selected_feature_cols + X_y_all.iloc[:, -4:].columns.tolist()]

    # 以 PowerTransformer 讓特徵分布更接近常態
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

    # 時序視窗長度
    
    X, y = val_windows(X_y_all, ref_day=DEFAULT_LOOKBACK_WINDOW, period=20)

    X_train, y_train = make_datasets(X, y, X_y_all.index, train_start, train_end, DEFAULT_LOOKBACK_WINDOW)                      
    X_test, y_test = make_datasets(X, y, X_y_all.index, test_start, test_end, DEFAULT_LOOKBACK_WINDOW)

    return X_train, X_test, y_train, y_test, X_y_all.index

test_mode = False           # <=================== TEST MODE
root_dir = HYPER_ROOT
des_dir = 'dynamicFlooding'
symbols = [x.strip() for x in os.getenv('STOCK_IDS', '2330').split(',') if x.strip()]
debug = True

# Flooding 水位候選值（可自行調整）
flooding_b_candidates = [0.05, 0.10, 0.20]
repeats_per_flooding = 2
# Combined-5 v2 production defaults（1.5× baseline 18 -> 27 repeats）
total_repeats = 27

model_types_list = ['combined5']  # combined 5-aspect variant; ignore MODEL_TYPES env
from combined_features import COMBINED_ASPECT, load_and_combine_features  # noqa: E402


def _resolve_feature_preprocess():
    """決定是否執行特徵前處理（相關過濾 + sanitize + scaler）。

    解析順序：
      1) 環境變數 `FEATURE_PREPROCESS`（0/no/false 關閉；1/yes/true 開啟）
      2) 互動式 TTY：詢問使用者
      3) 否則預設開啟
    """
    raw = os.getenv('FEATURE_PREPROCESS')
    if raw is not None:
        return raw.strip().lower() not in {'0', 'no', 'false', 'off', ''}
    if not sys.stdin.isatty():
        return True
    try:
        answer = input("是否執行特徵前處理（相關過濾 + sanitize + Yeo-Johnson/Robust scaler）？[Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer not in {'n', 'no', '0', 'false'}


DO_FEATURE_PREPROCESS = _resolve_feature_preprocess()
os.environ['FEATURE_PREPROCESS'] = '1' if DO_FEATURE_PREPROCESS else '0'
print(f"[PREPROCESS] feature preprocessing = {'ON' if DO_FEATURE_PREPROCESS else 'OFF (pass-through)'}")
print(
    f"[CV] VALIDATION_MODE={VALIDATION_MODE} WF_N_SPLITS={WF_N_SPLITS} "
    f"WF_VAL_RATIO={WF_VAL_RATIO} WF_VAL_YEARS={WF_VAL_YEARS} "
    f"WF_VAL_SAMPLES={WF_VAL_SAMPLES} WF_GAP={WF_GAP}"
)


for symbol in symbols:
    # 逐股票、逐模型類型訓練
    for train_model in model_types_list:  # 'tech_trend','moment','sentiment', 'trade' --- IGNORE ---
 
        # 若 ATT AutoML trial 存在且未產生過實驗檔，才進行訓練
        best_hp_values = load_best_att_hyperparameters(root_dir, train_model, symbol)
        if best_hp_values is not None and not os.path.exists(f'experiments_test/{train_model}{"_test" if test_mode else ""}'):
            best_lookback_window = int(best_hp_values.get('lookback_window', DEFAULT_LOOKBACK_WINDOW))
            print(f"[{train_model}_{symbol}] best lookback_window from AutoML: {best_lookback_window}")

            # ===================== 1) 載入資料與時間區間設定（combined 5 aspects）=====================
            data, train_start = load_and_combine_features(symbol)

            # 序列視窗長度
            n_steps = best_lookback_window
            print(f"[{train_model}_{symbol}] final lookback_window in use = {n_steps}")
            forecast_days = 20

            # Combined-5 v2 split (env-driven; same vars as outer block)
            train_end = TRAIN_END_DATE
            data_max = data.index.max().strftime('%Y-%m-%d')
            test_start, test_end = TEST_START_DATE, min(TEST_END_DATE, data_max)
            print(f"[SPLIT] {train_model}_{symbol}  train_end={train_end}  test={test_start}~{test_end}")

            # ===================== 2) 特徵清理與縮放 =====================
            if not DO_FEATURE_PREPROCESS:
                # pass-through：不做相關過濾、不做 sanitize、不做 scaler
                label_cols = data.columns[-4:]
                data = data.astype({col: np.float64 for col in label_cols})
                data = data[~data.index.duplicated(keep='last')]

                # 僅對非標籤欄做最基本 NaN/inf 清理
                feat_df = data.iloc[:, :-4].apply(pd.to_numeric, errors='coerce').astype(np.float64)
                feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                data = pd.concat([feat_df, data.iloc[:, -4:]], axis=1)

                # 記錄本次使用的特徵欄位（與啟用前處理時的流程對齊）
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
                    # 衍生特徵刻意保留互補訊號（短/中/長 return、成長率各期等），跳過 corr 過濾
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

                # expander 後的特徵多為 bounded / z-score / log-return / signed log1p，用
                # RobustScaler 穩健縮放；其餘維持 PowerTransformer（Yeo-Johnson）
                if train_model in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment', 'combined5'):
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
            
            # ===================== 3) 建立監督式窗口資料 =====================
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
                # Fallback: 若 blocking split 無法產生驗證集，改用末段時間切分
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


            # 標籤保留整數格式（配合 SparseCategoricalCrossentropy）
            y_train_int = y_train.astype('int32')
            y_val_int = y_val.astype('int32')
            X_test, y_test = get_windows(X, y, test_slice, n_steps)

            batch_size = 120
            nb_epoch = 32
            repeats = NUM_REPEATS if NUM_REPEATS > 0 else total_repeats

            # 建立訓練/驗證資料管線
            # class_weight → per-sample weights（tf.data 不支援 fit(class_weight=...)，
            # 改把樣本權重包進 dataset 第三元素；自訂 train_step 已支援 3-tuple）
            _cls, _cnt = np.unique(y_train_int, return_counts=True)
            _n_total = float(len(y_train_int))
            _n_cls = float(len(_cls))
            _cls_weight = {int(c): _n_total / (_n_cls * float(cnt)) for c, cnt in zip(_cls, _cnt)}
            sample_weight_train = np.array(
                [_cls_weight[int(v)] for v in y_train_int], dtype='float32'
            )
            print(f'[CLASS_WEIGHT][{train_model}_{symbol}] {_cls_weight}')
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (X_train, y_train_int, sample_weight_train)
            )
            ds_options = tf.data.Options()
            ds_options.experimental_deterministic = False
            train_dataset = train_dataset.with_options(ds_options).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_int))
            val_dataset = val_dataset.with_options(ds_options).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)



            # ===================== 4) 以最佳參數重複訓練與輸出 =====================
            # 再由最佳的模型取出做 n 次實驗，並將 n 次的模型及預測儲存
            Path(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}').mkdir(parents=True, exist_ok=True)
            # 學術風格圖輸出資料夾（依 flood_mode 分層）
            plot_out_dir = f'{EVAL_PLOT_DIR}/{FLOOD_MODE}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}'
            Path(plot_out_dir).mkdir(parents=True, exist_ok=True)

            top_acc = list()
            max_acc = 0 
            
            
            # ============================================================= RAdam ==============================================================
                                         #<================================== tunable ===========
            '''learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 0.001, 
                decay_steps = total_steps * .7, 
                name = 'CosineDecay')'''
            # RAdam + Lookahead
            # 多次重複訓練，取平均表現
            for r in range(repeats):

                # 依 FLOOD_MODE 決定每次 repeat 的 flooding 水位：
                #   none    -> 0（train_step 內 if flooding_b>0 直接跳過）
                #   static  -> 固定 STATIC_FLOOD_B
                #   dynamic -> 起始 b 取 0.05 步長 grid [0.10..0.45]，
                #             後續由 DynamicFloodingCallback 在同範圍內調整
                if FLOOD_MODE == 'none':
                    flooding_b = 0.0
                elif FLOOD_MODE == 'static':
                    flooding_b = STATIC_FLOOD_B
                else:
                    # 起始 b: NUM_REPEATS 個均勻覆蓋 [0.0, 0.4]。
                    # callback 後續在 [0.0, 0.4] 範圍內以 ±0.03 步長微調。
                    b_grid = np.linspace(0.0, 0.4, max(repeats, 1))
                    flooding_b = float(b_grid[r % len(b_grid)])

                print(f'{train_model}_{symbol} => Repeat {r+1}, mode={FLOOD_MODE}, flooding_b={flooding_b:.2f}')

                # 使用 AutoML 最佳參數建立模型
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

                # 每次重複都建立新的 optimizer，避免 slot 變數衝突
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
                max_epochs = MAX_EPOCHS_ENV if MAX_EPOCHS_ENV > 0 else 180

                metrics_list = [
                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.CategoricalCrossentropy(name='ce'),
                    tf.keras.metrics.Recall(class_id=1, name='recall'),
                    tf.keras.metrics.AUC(curve='ROC', name='auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc')
                ]
                if tfa is not None:
                    metrics_list.insert(
                        3,
                        tfa.metrics.FBetaScore(num_classes=2, beta=1.0, threshold=0.5, average='weighted')
                    )
                else:
                    metrics_list.insert(3, tf.keras.metrics.F1Score(name='f1', average='weighted', threshold=0.5))

                model.compile(
                    optimizer=optimizer,
                    loss=SparseCategoricalCrossentropy(from_logits=False),
                    metrics=metrics_list,
                )

                warmup_one_batch_from_dataset(model, train_dataset)

                # 依 FLOOD_MODE / DISABLE_EARLY_STOPPING 動態組裝 callbacks
                fit_callbacks = []
                if FLOOD_MODE == 'dynamic':
                    fit_callbacks.append(
                        DynamicFloodingCallback(
                            monitor='val_pr_auc',
                            min_b=0.0,
                            max_b=0.4,
                            step_up=0.03,
                            step_down=0.03,
                            patience=4,
                            min_delta=1e-4,
                            verbose=1
                        )
                    )
                # 保留 ReduceLROnPlateau（不會中止訓練）
                fit_callbacks.append(
                    ReduceLROnPlateau(
                        monitor='val_pr_auc',
                        mode='max',
                        factor=0.5,
                        patience=25,
                        min_delta=1e-7,
                        verbose=1
                    )
                )
                if not DISABLE_EARLY_STOPPING:
                    fit_callbacks.append(
                        EarlyStopping(
                            monitor='val_pr_auc',
                            mode='max',
                            patience=65,
                            min_delta=1e-7,
                            restore_best_weights=True,
                            verbose=1
                        )
                    )
                # 放最後：記錄每 epoch 的 flooding_b 到 history（三組欄位一致）
                fit_callbacks.append(FloodingLevelLogger())

                history_obj = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=max_epochs,
                    verbose=FIT_VERBOSE,
                    callbacks=fit_callbacks
                )
                history = history_obj.history

                # 保存 training history（供後續視覺化比較）
                history_df = pd.DataFrame(history)
                history_df.insert(0, 'epoch', np.arange(1, len(history_df) + 1))
                history_csv = (
                    f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}'
                    f'{"_test" if test_mode else ""}/history_{r+1}.csv'
                )
                history_df.to_csv(history_csv, index=False)
                # 每個 run 的學術風格圖（Accuracy + Loss/flooding b）
                plot_history(
                    history,
                    out_path=f'{plot_out_dir}/history_{r+1}.png',
                    flood_mode=FLOOD_MODE,
                    static_b=STATIC_FLOOD_B,
                    title=f'ATT {train_model} {symbol} (run {r+1})'
                )

                mean_val_pr_auc = float(np.mean(np.array(history.get('val_pr_auc', [0.0]))))
                top_acc.append(mean_val_pr_auc)

                if mean_val_pr_auc >= max_acc:
                    max_acc = mean_val_pr_auc

                # --- val 診斷：正類基準率 / 預測正類比例 / ROC-AUC ---
                try:
                    val_prob = model.predict(X_val, batch_size=128, verbose=0)[:, 1]
                    val_true = np.asarray(y_val_int).reshape(-1)
                    base_rate = float(np.mean(val_true == 1))
                    pred_pos_rate = float(np.mean(val_prob >= 0.5))
                    try:
                        from sklearn.metrics import roc_auc_score
                        val_roc_auc = (
                            float(roc_auc_score(val_true, val_prob))
                            if len(np.unique(val_true)) > 1 else float('nan')
                        )
                    except Exception:
                        val_roc_auc = float('nan')
                    print(
                        f'[DIAG][{train_model}_{symbol}] run={r+1} mode={FLOOD_MODE} '
                        f'val_pos_base_rate={base_rate:.3f} pred_pos_rate={pred_pos_rate:.3f} '
                        f'val_roc_auc={val_roc_auc:.3f}'
                    )
                except Exception as _diag_err:
                    print(f'[DIAG][WARN][{train_model}_{symbol}] {(_diag_err)}')

                print(f'[{train_model}_{symbol}] flooding_b={flooding_b:.2f} Average val_pr_auc: ', round(mean_val_pr_auc, 4))
                print(f'[{train_model}_{symbol}] Best val_pr_auc so far: ', round(max_acc, 4))

                y_pred = pd.Series(model.predict(X_test, batch_size=64, verbose=0)[:, 1], index=data[test_slice].index)
                #plot_prediction(symbol, y_pred, out_path="prediction.png")



                # Save the experimental models

                model.save(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_{r+1}.keras')

                # --- 為 DES 訓練保存分段預測（正確日期對齊）---
                # 新增三個檔案：train (ATT 訓練段) / val (ATT 驗證段) / test (2026+)
                # 舊版 experiment_result_{r+1}.csv 維持原行為（向後相容，不動）
                _exp_dir = f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}'
                _train_dates = window_indices_to_dates(train_indices, train_slice, n_steps, data.index)
                _val_dates = window_indices_to_dates(val_indices, train_slice, n_steps, data.index)
                _test_dates = window_indices_to_dates(np.arange(len(X_test)), test_slice, n_steps, data.index)

                _train_pred = model.predict(X_train, batch_size=64, verbose=0)[:, 1]
                _val_pred = model.predict(X_val, batch_size=64, verbose=0)[:, 1]
                _test_pred = model.predict(X_test, batch_size=64, verbose=0)[:, 1]

                pd.Series(_train_pred, index=_train_dates).to_csv(
                    f'{_exp_dir}/experiment_result_train_{r+1}.csv', header=False)
                pd.Series(_val_pred, index=_val_dates).to_csv(
                    f'{_exp_dir}/experiment_result_val_{r+1}.csv', header=False)
                pd.Series(_test_pred, index=_test_dates).to_csv(
                    f'{_exp_dir}/experiment_result_test_{r+1}.csv', header=False)

                # 舊版檔案：維持與先前一致的（train+test 串接、索引取 data.index 尾段）
                y_pred = pd.Series(
                    np.concatenate([_train_pred, _test_pred]),
                    index=data.index[-(len(X_train) + len(X_test)):]
                )
                y_pred.to_csv(f'{_exp_dir}/experiment_result_{r+1}.csv', header=False)

                tf.keras.backend.clear_session()
                #device = cuda.get_current_device()
                #device.reset()

        else: 
            # print(f'{train_model[-4:]} {train_model[4:-5]} exists')

            continue
        # To keep top 3 models and predictions, remove the rest
        sort_index = np.argsort(np.array(top_acc))
        for index in sort_index[:-3]:
            os.remove(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_{index+1}.keras')
            os.remove(f'{EXPERIMENT_ROOT}/ATT_{train_model}_{symbol}{"_test" if test_mode else ""}/experiment_result_{index+1}.csv')
        tf.keras.backend.clear_session()
        gc.collect()
    
  