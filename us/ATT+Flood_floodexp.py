import os
import gc
import json
import sys
import time
import warnings
from datetime import datetime, timedelta

"""Attention + FCN + Flooding + Keras Tuner 訓練腳本。

整體流程：
1) 讀取單一標的/因子的特徵與標籤資料
2) 進行特徵清理、縮放與時間序列切窗
3) 以 HyperModel 建構 Attention + FCN 模型，並用 Bayesian Optimization 搜參
4) 透過自訂 Tuner 在時間序列切分下評估 trial 指標
5) 於主流程中逐檔訓練並輸出最佳 trial 結果

設計目標：
- 優先避免時間序列資料洩漏
- 兼顧 Blackwell GPU（RTX 50 系列）在 WSL 上的穩定性
- 保留可觀察性（重點指標與告警）與可維護性（清楚分段）
"""

# 降低 TensorFlow/XLA 日誌噪音與首次編譯卡頓（需在 import tensorflow 前設定）
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

# 在 WSL/單卡情境避免一次性吃滿顯存，降低底層中止（core dumped）風險
try:
    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu_device in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu_device, True)
except Exception:
    pass

# from numba import cuda
warnings.filterwarnings('ignore')

# 為避免特定 TF/WSL/GPU 組合下的 native crash（double free），暫時固定走原生 Keras optimizer/metrics
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
    # speed mode（預設）: 避免被 shell 殘留 env 影響而誤開慢速 eager
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
ATT_HYPERBAYES_DIR = os.getenv('HYPERBAYES_ATT_DIR', 'D:/hyperbayes_ATT')
ATT_SCALER_DIR = os.getenv('FEATURE_SCALER_ATT_DIR', 'D:/feature_scaler_ATT')

# Walk-Forward Validation options
# VALIDATION_MODE: 'blocking' (default, single block CV — prior behavior),
#                  'walk_forward_rolling'   — fixed-size train window slides forward,
#                  'walk_forward_expanding' — train start fixed, train window grows.


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
      2) 若未設定且為互動式 TTY（非子程序隔離執行）：詢問使用者（預設 rolling）。
      3) 否則預設 `blocking`（保留舊行為）。
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


def _resolve_feature_preprocess():
    """決定是否執行特徵前處理（相關過濾 + sanitize + scaler）。

    解析順序：
      1) 環境變數 `FEATURE_PREPROCESS`（0/no/false 關閉；1/yes/true 開啟）。
      2) 若未設定且為互動式 TTY（非子程序隔離執行）：詢問使用者。
      3) 否則預設開啟。
    """
    raw = os.getenv('FEATURE_PREPROCESS')
    if raw is not None:
        return raw.strip().lower() not in {'0', 'no', 'false', 'off', ''}
    if ISOLATED_CHILD_RUN or not sys.stdin.isatty():
        return True
    try:
        answer = input("是否執行特徵前處理（相關過濾 + sanitize + Yeo-Johnson/Robust scaler）？[Y/n]: ").strip().lower()
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

# walk-forward 模式下，若未手動指定搜尋預算，預設自動降載以縮短 wall-time
# 可用 AUTO_REDUCE_SEARCH_FOR_WF=0 關閉
_stage_budget_overridden = any(
    os.getenv(k) is not None
    for k in ('STAGE1_MAX_TRIALS', 'STAGE2_MAX_TRIALS', 'STAGE1_EPOCHS', 'STAGE2_EPOCHS')
)
if (
    (not FAST_DEBUG)
    and (not _stage_budget_overridden)
    and (VALIDATION_MODE in ('walk_forward_rolling', 'walk_forward_expanding'))
    and (os.getenv('AUTO_REDUCE_SEARCH_FOR_WF', '1') == '1')
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
    """建立資料專屬 cache key，避免不同資料共用到錯誤索引。"""
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
    """將 Windows 路徑轉為可在當前系統使用的路徑。

    參數:
        win_path: 可能為 Windows 格式（如 `D:/...`）的路徑字串。

    回傳:
        在 Windows 回傳原路徑；在 Linux/WSL 回傳 `/mnt/<drive>/...` 路徑。

    副作用:
        無。
    """
    if platform.system() != 'Windows' and len(win_path) >= 2 and win_path[1] == ':':
        drive = win_path[0].lower()
        return '/mnt/' + drive + win_path[2:].replace('\\', '/')
    return win_path


SCALER_ROOT = platform_path(ATT_SCALER_DIR)
os.makedirs(SCALER_ROOT, exist_ok=True)

class SinusoidalPositionalEncoding(layers.Layer):
    """固定式正弦/餘弦位置編碼層。

    用途:
        取代第三方 PositionEmbedding，提供無可訓練參數的位置資訊。
        相容混合精度（自動轉換 encoding dtype）。

    參數:
        無（透過輸入張量 shape 動態推導）。

    屬性:
        無持久可訓練參數。

    副作用:
        無。
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
        # 強制轉換 encoding 到輸入的 dtype（支援混合精度 float16）
        encoding = tf.cast(encoding, x.dtype)
        return x + encoding
    
    def get_config(self):
        return super().get_config()

# ==============================================================
# 本檔案用途：
# 1) 讀取單一股票的特徵資料（X）與標籤（y）
# 2) 進行特徵清理/正規化與時間序列切窗
# 3) 使用可變層數 Attention + FCN 模型進行分類
# 4) 透過 Keras Tuner (Bayesian Optimization) 搜尋超參數
# 5) 以時間序列交叉驗證結果作為 trial objective
# ==============================================================

# 時間序列交叉驗證, 每個 block 中選 80% train, 20% test, 兩份資料中間又空出 10 個交易日資料, avoid data leaking
class BlockingTimeSeriesSplit:
    """時間序列分割器（Blocking CV）。

    用途:
        保持時間順序切分資料，並在 train/val 間加入 gap 防止資料洩漏。

    參數:
        n_splits: 分割數。
        val_ratio: 每個 block 的驗證比例。
        gap: train 與 val 之間的間隔樣本數。

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
        """回傳每一折的 train/validation 索引。

        參數
        ----
        X : array-like
            依時間排序的樣本序列。
        y, groups : 兼容 sklearn 介面保留參數。
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
        """依時間順序產生 (train_idx, val_idx)，n_splits 個滾動視窗。

        Val 區間等長且順序不重疊。Rolling 模式下 train 長度固定；
        Expanding 模式下 train 起點固定，長度隨視窗增長。
        """
        n_samples = int(len(X))
        if n_samples <= self.gap + 2:
            return

        if self.val_samples > 0:
            val_size = max(1, int(self.val_samples))
        else:
            val_size = max(1, int(n_samples * self.val_ratio / self.n_splits))
        # 預留出 n_splits 個 val 區間與至少一個 train 區間 + gap。
        total_val = val_size * self.n_splits
        if total_val + self.gap + 1 >= n_samples:
            # 限縮 val_size 使其可容納
            val_size = max(1, (n_samples - self.gap - 2) // (self.n_splits + 1))
            total_val = val_size * self.n_splits

        # rolling train_len: 所有折中 train 長度相同，使用第一折 train 可用空間。
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

def val_windows(data, ref_day=60, period=20): 
    """將時間序列資料轉為監督式學習窗口。

    參數:
        data: 含特徵與標籤欄位的 DataFrame（最後 4 欄視為標籤/保留欄）。
        ref_day: 每筆樣本回看天數（視窗長度）。
        period: 標籤欄位後綴，對應 `y_{period}`。

    回傳:
        (X_val, y_val)
        - X_val: shape=(樣本數, ref_day, 特徵數)
        - y_val: shape=(樣本數,)

    副作用:
        無。
    """
    n_features = data.shape[1] - 4
    feat_arr = np.ascontiguousarray(data.iloc[:, :-4].to_numpy(dtype=np.float64))
    n_windows = feat_arr.shape[0] - ref_day + 1
    shape = (n_windows, ref_day, n_features)
    strides = (feat_arr.strides[0],) + feat_arr.strides
    X_val = np.lib.stride_tricks.as_strided(feat_arr, shape=shape, strides=strides).copy()
    y_val = data[f"y_{period}"].to_numpy()[ref_day - 1:]
    return X_val, y_val
    
# 為儘量取得有效資料, 因 PE & PB 正規計算需至少參考到 3 年歷史資料,  故訓練起始日為至少 3*250 個交易日後, cast_to_floatx 將 numpy 轉為 Keras 浮點類型
def make_datasets(X, y, idx, start, end, ref_day):
    """依日期區間切出訓練/測試資料，並套用最小歷史樣本限制。

    參數:
        X: 已切窗完成的特徵陣列。
        y: 與 X 對齊的標籤陣列。
        idx: 原始資料日期索引（DatetimeIndex）。
        start: 起始日期字串（YYYY-MM-DD）。
        end: 結束日期字串（YYYY-MM-DD）。
        ref_day: 視窗長度，用於索引回推。

    回傳:
        (X_slice, y_slice)，皆為 Keras floatx dtype。

    副作用:
        無。
    """
    start_idx = idx.get_loc(idx.to_series()[start:].iloc[0])
    end_idx = idx.get_loc(idx.to_series()[:end].iloc[-1])
    return K.cast_to_floatx(X[max(1*250-1, start_idx-ref_day+1):end_idx-ref_day+2]), K.cast_to_floatx(y[max(1*250-1, start_idx-ref_day+1):end_idx-ref_day+2])

def fit_sanitize_statistics(df, max_abs=1e6, q=0.001):
    """以訓練區間估計 sanitize 統計量，避免未來資訊洩漏。"""
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


def sanitize_feature_values(feature_df, fit_start, fit_end):
    """向後相容：用 fit 區間估計統計後套用到全資料。"""
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
    """設定 GPU memory growth，降低一次性佔滿顯存的風險。

    參數:
        無。

    回傳:
        無。

    副作用:
        會修改 TensorFlow runtime 的 GPU memory growth 設定。
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
    """執行 1 個 batch 暖機，將首次 kernel 編譯開銷前移。

    參數:
        model: 已 compile 的 Keras 模型。
        x_train: 訓練特徵陣列。
        y_train: 訓練標籤陣列。
        batch_size: 暖機 batch 大小上限。

    回傳:
        暖機耗時（秒）。

    副作用:
        觸發一次前向推論/評估圖編譯，不更新模型權重。
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
    """快取每種資料窗口的切分索引與權重，減少 trial 前重複計算。

    回傳結構：``{'folds': [{train_indices, val_indices, class_weights, val_weights}, ...],
    'mode': VALIDATION_MODE}``。一般 blocking 模式只會有 1 個 fold；walk-forward 則為
    ``WF_N_SPLITS`` 個 fold（rolling 或 expanding）。
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
        """將序列化資料還原為可使用的 Keras 模型。"""
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
        """替 Keras Model 注入 `__reduce__`，讓 pickle 可序列化模型。"""

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
    """套用 Flooding 訓練策略的自訂 Keras 模型。

    用途:
        覆寫 `train_step`，將原始 loss 映射到 flooding loss，抑制過度擬合。

    參數:
        與 `keras.Model` 建構參數一致（由 Functional API 建立）。

    屬性:
        flooding_b: Flooding 水位（float，預設 0.10）。

    副作用:
        改變每個 batch 的訓練 loss 計算方式。
    """
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
    """依驗證表現動態調整 `flooding_b` 的 callback。

    用途:
        在每個 epoch 結束後，根據監控指標是否改善來上下調整 flooding 水位。

    參數:
        monitor: 監控指標名稱（預設 `val_recall`）。
        min_b/max_b: flooding_b 上下界。
        step_up/step_down: 調整步長。
        patience: 未改善容忍 epoch 數。
        min_delta: 視為改善的最小幅度。
        verbose: 是否印出調整訊息。

    屬性:
        best, wait 與上述控制參數。

    副作用:
        會直接修改 `self.model.flooding_b`。
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
    """每個 trial 的超時保護 callback。"""
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
    """產生下三角因果遮罩（只看過去）。

    用途:
        給 MultiHeadAttention 當 `attention_mask`，避免看到未來時間點。

    參數:
        無。

    屬性:
        無可訓練參數。

    副作用:
        無。
    """
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return tf.linalg.band_part(
            tf.ones((1, seq_len, seq_len), dtype=tf.float32), -1, 0
        )
    
    def get_config(self):
        return super().get_config()

def add_causal_mha_block(x, num_heads, key_dim, dropout_rate, name_prefix):
    """建立因果式多頭注意力區塊。

    參數:
        x: 輸入張量，shape=(batch, time, features)。
        num_heads: 注意力頭數。
        key_dim: 每個 head 的 key 維度。
        dropout_rate: 注意力輸出的 dropout 比例。
        name_prefix: layer 命名前綴。

    回傳:
        經 MHA + 殘差 + LayerNorm 後的輸出張量。

    副作用:
        無。
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
    """Keras Tuner 使用的超模型定義。

    用途:
        定義可搜尋的 Attention + FCN 網路與 compile 設定。

    參數:
        name: HyperModel 名稱。
        tunable: 是否允許 tuner 調整。
        input_shape: 模型輸入 shape。
        strategy: 分散式策略（保留欄位）。

    屬性:
        input_shape, strategy。

    副作用:
        `build()` 會依 hp 建立並 compile 新模型。
    """
    def __init__(self, name=None, tunable=True, input_shape=None, strategy=None):
        self.input_shape = input_shape
        self.strategy = strategy
        super().__init__(name, tunable)

    def build(self, hp):     
        """根據 hp 組合建構並編譯模型。"""
        inputs = keras.Input(shape=self.input_shape, name="inputs")

        # 位置編碼：以無參數正弦波編碼取代 PositionEmbedding
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

        # Temperature: 控制 softmax 平滑度
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

        # 使用自訂 FloodingModel 包裝，讓 train_step 生效
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
    """以時間序列切分評估 trial 的自訂 Tuner。

    用途:
        覆寫 `run_trial`，在每個 trial 內執行 blocking CV，並回報平均指標。

    參數:
        與 `kt.engine.tuner.Tuner` 相容。

    屬性:
        繼承自 Keras Tuner Tuner。

    副作用:
        執行訓練、更新 oracle trial metrics、清理 session。
    """
    def run_trial(self, trial, x=None, y=None, batch_size=40, epochs=1, callbacks=[], windowed_data=None, *args, **kwargs):
        """執行單一 trial 並回報時間序列驗證結果。

        用途:
            在 trial 內進行 blocking CV 訓練，彙整各項驗證指標後更新 oracle。

        參數:
            trial: 當前 Keras Tuner trial 物件。
            x: 訓練特徵陣列（未提供 `windowed_data` 時使用）。
            y: 訓練標籤陣列（未提供 `windowed_data` 時使用）。
            batch_size: 訓練 batch 大小。
            epochs: 訓練 epoch 數。
            callbacks: 額外 callback 清單（保留相容）。
            windowed_data: 可選，格式為 `{lookback_window: (X_train, y_train_onehot)}`。
            *args, **kwargs: 保留給父類別/調用端的擴充參數。

        回傳:
            無（結果透過 `self.oracle.update_trial(...)` 回報）。

        副作用:
            會執行模型訓練、推論、更新 trial metrics，並清理 Keras session。
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

        # 訓練資料切三份, 記錄每個 block 在驗證集上的分類表現, 取平均作為該組參數的 objective score
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
                shuffle=False, # 時間序列建議不打亂，以避免時間洩漏
                validation_data=(x_val, y_val, val_weights), 
                class_weight=class_weights,
                callbacks=[
                    timeout_callback,
                    DynamicFloodingCallback(
                        monitor='val_f_beta_score',
                        min_b=0.02,
                        max_b=0.20,
                        step_up=0.01,
                        step_down=0.005,
                        patience=4,
                        min_delta=1e-4,
                        verbose=1
                    ),
                    # 當訓練落到瓶頸, 降低 learning rate
                    #ReduceLROnPlateau(monitor='val_fbeta_score', mode='max', factor=0.2, patience=10, min_delta=1E-7, verbose=1)
                    #,EarlyStopping(monitor='val_fbeta_score', mode='max', patience= 10, verbose=1)
                    ReduceLROnPlateau(monitor='val_f_beta_score', mode='max', factor=0.5, patience=25, min_delta=1e-7, verbose=1)
                    ,EarlyStopping(monitor='val_f_beta_score', mode='max', patience=65, min_delta=1e-7, restore_best_weights=True, verbose=1)
                ]
            )

            if timeout_callback.timed_out:
                trial_timed_out = True
                K.clear_session()
                gc.collect()
                break

            eval_results = model.evaluate(x_val, y_val, verbose=0)
            # 因 metrics_names 與 evaluate 回傳順序一致，故可轉成 dict 方便取值
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
                # 這裡回報給 Bayesian Optimization 的是 trial 的綜合表現
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
    回傳：concat(衍生特徵, 原最後 4 欄標籤)。共通 preprocess (corr 過濾 / scaler /
    sanitize) 會在後續步驟對本函式輸出欄位再處理。
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


# macro 欄位分組：
#   - 穩態 / 有界：保留 level
#   - 非平穩價格：轉 log-return / rolling z-score
#   - 選擇權量、期貨淨額：先 ffill(0→NaN) 再做對應轉換
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
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 1) 穩態 / 有界欄位保留 level
    for c in MACRO_STATIONARY_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            # 利率/FX/VIX 的 0 通常也是缺值（尤其利率）→ ffill
            if c in ('Price_rate_3m', 'Price_rate_10y', 'Price_FX'):
                s = s.replace(0, np.nan).ffill()
            out[c] = s

    # 2) term spread（若 rate_3m / rate_10y 都有）
    if 'Price_rate_3m' in out.columns and 'Price_rate_10y' in out.columns:
        out['term_spread'] = out['Price_rate_10y'] - out['Price_rate_3m']

    # 3) VIX 另加 log(vix) 與 z-score
    if 'Price_VIX' in out.columns:
        vix = out['Price_VIX'].replace(0, np.nan).ffill()
        out['log_vix'] = np.log(vix)
        m = vix.rolling(60, min_periods=30).mean()
        sd = vix.rolling(60, min_periods=30).std().replace(0, np.nan)
        out['vix_z60'] = (vix - m) / sd

    # 4) 非平穩價格 → log-return + rolling z
    for c in MACRO_PRICE_LEVEL_COLS:
        if c not in df.columns:
            continue
        p = pd.to_numeric(df[c], errors='coerce').replace(0, np.nan).ffill()
        out[f'{c}_logret1']  = np.log(p / p.shift(1))
        out[f'{c}_logret20'] = np.log(p / p.shift(20))
        m = p.rolling(60, min_periods=30).mean()
        sd = p.rolling(60, min_periods=30).std().replace(0, np.nan)
        out[f'{c}_z60'] = (p - m) / sd

    # 5) 台指選擇權量 → log1p + diff20；另加 put/call log-ratio
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

    # 6) 期貨淨額（可正可負） → signed log1p + diff
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
# tech_trend / fundamental / moment expander（pass-through 友善版）
# 設計原則：
#   1. 所有輸出都是「確定性 + 硬 clip」，不依賴訓練區間分位數，
#      因此 FEATURE_PREPROCESS=off 時也能直接送入模型。
#   2. 僅改動明顯非平穩或重尾的欄位（raw OHLCV、成長率、PEG、cci、acc_*、vpt），
#      保留原本就已平穩/有界的欄位（sma/hullma/bias/alpha/RSI/K/D 等）不動，
#      將新舊模型差異降到最小。
#   3. 輸出值域大致落在 ±3 內，配合 signed log1p / 固定 clip 壓住極端尾部。
# ===================================================================


def _expand_tech_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """tech_trend CSV: 將 open/high/low/close/volume 轉成尺度無關特徵，
    osc 除以 close 做成比率；其餘（sma_*/hullma_*/mmi_*/aroon_osc/bb/bias/alpha）保持原樣。
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]

    out = pd.DataFrame(index=df.index)

    # 1) 原本就平穩/有界的欄位：保留
    passthrough_cols = [
        'sma_5', 'sma_10', 'sma_20', 'sma_60', 'sma_120',
        'hullma_20', 'hullma_60', 'hullma_120',
        'mmi_5', 'mmi_10', 'mmi_20',
        'aroon_osc', 'bb', 'bias', 'alpha',
    ]
    for c in passthrough_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors='coerce')

    # 2) osc：price-scaled → 轉為與 close 同尺度的比率
    if 'osc' in df.columns and 'close' in df.columns:
        close_raw = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        osc_raw = pd.to_numeric(df['osc'], errors='coerce')
        out['osc_pct'] = (osc_raw / close_raw).clip(-0.3, 0.3)

    # 3) close → log-return 1/5/20（丟掉絕對價格）
    if 'close' in df.columns:
        close = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        log_close = np.log(close)
        out['ret_1']  = (log_close - log_close.shift(1)).clip(-0.3, 0.3)
        out['ret_5']  = (log_close - log_close.shift(5)).clip(-0.5, 0.5)
        out['ret_20'] = (log_close - log_close.shift(20)).clip(-0.8, 0.8)

    # 4) (high - low) / close：日內波動；gap = log(open / prev_close)
    if {'high', 'low', 'close'}.issubset(df.columns):
        h = pd.to_numeric(df['high'], errors='coerce')
        l = pd.to_numeric(df['low'], errors='coerce')
        c = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        out['hl_range'] = ((h - l) / c).clip(0.0, 0.2)
    if {'open', 'close'}.issubset(df.columns):
        o = pd.to_numeric(df['open'], errors='coerce').replace(0, np.nan).ffill()
        c = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill()
        out['gap'] = (np.log(o) - np.log(c.shift(1))).clip(-0.15, 0.15)

    # 5) volume：轉為相對 20D 平均的比率
    if 'volume' in df.columns:
        v = pd.to_numeric(df['volume'], errors='coerce').clip(lower=0)
        v_ma = v.rolling(20, min_periods=5).mean().replace(0, np.nan)
        out['vol_ratio20'] = ((v / v_ma) - 1.0).clip(-5.0, 5.0)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


# fundamental 欄位分組（依 Feature_Cmoney_update.py）：
#   - PE_trailing / PBR：rolling 3Y river level percentile，已在 0..1
#   - DY：年化殖利率 %，典型 0..15
#   - Gross：毛利率 %，典型 0..100
#   - PEG：PE / growth，可正可負、可爆量
#   - R_*/E_*/Op_*/Gross_qoq/EPS_qoq：成長率 %，小基期時易爆量
FUNDAMENTAL_GROWTH_COLS = (
    'R_mom', 'R_yoy', 'R_acc_yoy',
    'E_qoq', 'E_yoy', 'E_acc_yoy',
    'Op_qoq', 'Op_yoy', 'Op_acc_yoy',
    'Gross_qoq', 'EPS_qoq',
)


def _expand_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """fundamental CSV: 對 bounded 欄位重新置中到 ±1 附近；
    對 PEG/成長率做 clip + signed log1p，壓制小基期爆量。
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 1) river-level 百分位（0..1） → 置中到 ±1
    for c in ('PE_trailing', 'PBR'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 1.0)
            out[c] = (s - 0.5) * 2.0

    # 2) DY：殖利率 %，clip 到 [0, 20] 後 /10 → 約 [0, 2]
    if 'DY' in df.columns:
        s = pd.to_numeric(df['DY'], errors='coerce').clip(0.0, 20.0)
        out['DY'] = s / 10.0

    # 3) Gross：毛利率 %，clip 到 [-20, 100] 後 /100 → 約 [-0.2, 1.0]
    if 'Gross' in df.columns:
        s = pd.to_numeric(df['Gross'], errors='coerce').clip(-20.0, 100.0)
        out['Gross'] = s / 100.0

    # 4) 成長率（%）：硬 clip ±300，再 sign * log1p(|x|/100)
    #    x=100% → 0.69；x=300% → 1.39；保留符號
    for c in FUNDAMENTAL_GROWTH_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(-300.0, 300.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s) / 100.0)

    # 5) PEG：ratio（可負可大）；clip ±10 後 sign * log1p(|x|)
    if 'PEG' in df.columns:
        s = pd.to_numeric(df['PEG'], errors='coerce').clip(-10.0, 10.0)
        out['PEG'] = np.sign(s) * np.log1p(np.abs(s))

    # 6) CMDTY（若存在）：商品指數 level → 20D log-return
    if 'CMDTY' in df.columns:
        p = pd.to_numeric(df['CMDTY'], errors='coerce').replace(0, np.nan).ffill()
        out['CMDTY_logret20'] = (np.log(p) - np.log(p.shift(20))).clip(-0.5, 0.5)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


def _expand_moment_features(df: pd.DataFrame) -> pd.DataFrame:
    """moment CSV: 將 RSI/K/D/ADX/WR 置中到 ±1；CCI clip ±300 後 /100；
    acc_* clip 後取 log；vpt 改為 diff 的 rolling z；beta clip ±3。
    """
    label_cols = df.columns[-4:].tolist()
    labels = df[label_cols]
    out = pd.DataFrame(index=df.index)

    # 1) 0..100 bounded → (x-50)/50 → [-1, 1]
    for c in ('rsi', 'k', 'd', 'adx'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.0, 100.0)
            out[c] = (s - 50.0) / 50.0

    # 2) wr 原本是 [-100, 0] → (x+50)/50 → [-1, 1]
    if 'wr' in df.columns:
        s = pd.to_numeric(df['wr'], errors='coerce').clip(-100.0, 0.0)
        out['wr'] = (s + 50.0) / 50.0

    # 3) cci：heavy-tailed 約 ±200~±400，clip ±300 後 /100 → [-3, 3]
    if 'cci' in df.columns:
        s = pd.to_numeric(df['cci'], errors='coerce').clip(-300.0, 300.0)
        out['cci'] = s / 100.0

    # 4) acc_*：價格比率中心 ~1.0；clip [0.5, 2.0] 後 log → [-0.69, 0.69]
    for c in ('acc_5', 'acc_10', 'acc_20', 'acc_60', 'acc_120'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').clip(0.5, 2.0)
            out[c] = np.log(s)

    # 5) vpt：累積量，非平穩 → 取 diff 後做 60D rolling z
    if 'vpt' in df.columns:
        s = pd.to_numeric(df['vpt'], errors='coerce')
        d = s.diff()
        m = d.rolling(60, min_periods=20).mean()
        sd = d.rolling(60, min_periods=20).std().replace(0, np.nan)
        out['vpt_z60'] = ((d - m) / sd).clip(-3.0, 3.0)

    # 6) beta：rolling OLS slope，clip 極端值
    if 'beta' in df.columns:
        s = pd.to_numeric(df['beta'], errors='coerce').clip(-3.0, 3.0)
        out['beta'] = s

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([out, labels], axis=1)


def run_isolated_stock_model_jobs(stock_ids, model_types):
    """每個 stock_id/model_type 以獨立子程序執行，避免長跑記憶體累積。"""
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

# 記錄本次批次訓練中失敗的 (stock_id, model_type)
error_list = []
#train_progress = symbols.reindex(sorted([os.path.basename(i)[6:-4] for i in glob('features/macro_*.csv')])).dropna().index.to_list() # ticker sorted

# 主訓練入口：可視需求擴充 stock_id 或 model_type 清單
stock_ids = [x.strip() for x in os.getenv('STOCK_IDS', '8299').split(',') if x.strip()]
model_types = [
    x.strip() for x in os.getenv(
        'MODEL_TYPES',
        'fundamental,trade,moment,sentiment,tech_trend,macro' #,trade,moment,sentiment,tech_trend,macro'moment,sentiment,tech_trend,
    ).split(',') if x.strip()
]

if ISOLATE_STOCK_MODEL_RUNS and (not ISOLATED_CHILD_RUN) and (len(stock_ids) * len(model_types) > 1):
    # 特徵前處理總開關：父程序先問（或讀 env），寫回 os.environ 讓所有子程序繼承
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

# 單程序模式（或子程序）：直接在此決定
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
            # ===================== 1) 讀取資料 =====================
            print(f"[TRAIN] {stock_id} {model_type}")
            X_y_all = pd.read_csv(platform_path(f"D:/Feature_new/{model_type}_{stock_id}.csv"), index_col=0, parse_dates=True)                             

            # sentiment 面向：原始 CSV 僅 3 欄（US/TW/個股情緒分數），展開為動能/相對
            # 強弱/rolling z-score 等衍生特徵，避免模型看到過度稀疏或常數化的輸入
            if model_type == 'sentiment':
                X_y_all = _expand_sentiment_features(X_y_all, stock_id)

            # trade 面向：先壓制重尾欄位（signed log1p），讓後續 sanitize / scaler 穩定
            if model_type == 'trade':
                X_y_all = _prepare_trade_features(X_y_all)

            # macro 面向：價格水準高度非平穩（近期屢破訓練期最高點），一律轉為
            # log-return / rolling z / log1p 等近平穩特徵；保留 rate / FX / VIX level
            if model_type == 'macro':
                X_y_all = _expand_macro_features(X_y_all)

            # tech_trend 面向：raw OHLCV 非平穩（近年價格遠高於訓練期），一律轉為
            # log-return / (hl)/close / vol_ratio 等尺度無關特徵；sma/hullma/bias/alpha 保留
            if model_type == 'tech_trend':
                X_y_all = _expand_tech_trend_features(X_y_all)

            # fundamental 面向：PE/PBR/DY/Gross 做 bounded rescale；
            # 成長率/PEG 做 clip + signed log1p，避免小基期爆量
            if model_type == 'fundamental':
                X_y_all = _expand_fundamental_features(X_y_all)

            # moment 面向：RSI/K/D/ADX/WR 置中到 ±1；CCI 硬 clip；acc_* 取 log；
            # vpt 改為 diff rolling z；beta clip
            if model_type == 'moment':
                X_y_all = _expand_moment_features(X_y_all)

            #Find the ealist vaild start date where all figures are positive
            if model_type == 'sentiment':
                # sentiment CSV 自 2015-01-05 起就是有效數據，直接採用
                train_start = '2015-01-06'
            elif model_type == 'trade':
                # trade 早期 (1999~2007) 大量欄位為 0，改以「≥50% 欄位非零」為準
                non_zero_date = _detect_non_zero_date(X_y_all, ratio_threshold=0.5)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            elif model_type == 'macro':
                # macro 早期 (1994~2007) 多數欄位為 0；用「≥90% 非零」為準
                non_zero_date = _detect_non_zero_date(X_y_all, ratio_threshold=0.9)
                train_start = datetime.strftime((non_zero_date.date()+timedelta(days=1)), '%Y-%m-%d')
            elif model_type in ('tech_trend', 'moment'):
                # tech_trend: sma_120 需 120 日；moment: acc_120/vpt_z60 需 60~120 日
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
            
            # ===================== 2) 特徵前處理 =====================
            if not DO_FEATURE_PREPROCESS:
                # pass-through：不做相關過濾、不做 sanitize、不做 scaler
                # 仍要對齊 lookback_start 起始並存一個「透明」bundle，讓推論端自動跳過 transform
                lookback_start = X_y_all.loc[:train_start].iloc[-n_timesteps+1:].index.min()
                feature_cols = X_y_all.columns[:-4]

                # 僅做最基本的 NaN/inf 清理，避免訓練階段崩潰（非 scaling）
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
                # (a) 移除高相關特徵，降低共線性（只用 train 區間估計）
                if model_type in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment'):
                    # 衍生特徵刻意保留互補訊號（短/中/長 return、成長率各期等），跳過 corr 過濾
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

                # (b) expander 後的特徵多為 bounded / z-score / log-return / signed log1p，用
                #     RobustScaler 穩健縮放；其餘（若未來新增的面向）維持 PowerTransformer。
                if model_type in ('sentiment', 'macro', 'fundamental', 'tech_trend', 'moment'):
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))
                else:
                    scaler = PowerTransformer(method='yeo-johnson')

                lookback_start = X_y_all.loc[:train_start].iloc[-n_timesteps+1:].index.min()
                feature_cols = X_y_all.columns[:-4]

                # (c) 先將原始特徵清理成可安全縮放的有限值
                sanitize_stats = fit_sanitize_statistics(X_y_all.loc[lookback_start:train_end, feature_cols])
                clean_features = apply_sanitize_statistics(X_y_all.loc[:, feature_cols], sanitize_stats)
                fit_features = clean_features.loc[lookback_start:train_end]

                # (d) 轉為 numpy 並再次防禦 NaN/inf
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

                # (e) 若 PowerTransformer 因資料形態不穩定而失敗，退回 RobustScaler
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
            
            # ===================== 3) 建立監督式資料 =====================
            # 為每個 lookback_window 預先建立訓練資料，供 AutoML trial 選擇
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
           
            # ===================== 4) 建立分散式訓練策略 =====================
            # 單卡用 OneDeviceStrategy 較穩定；多卡才使用 MirroredStrategy
            gpu_count = len(tf.config.list_physical_devices('GPU'))
            if gpu_count > 1:
                strategy = tf.distribute.MirroredStrategy()
            elif gpu_count == 1:
                strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
            else:
                strategy = tf.distribute.OneDeviceStrategy(device='/CPU:0')

            # ===================== 5) 兩階段超參數搜尋 =====================
            # Stage 1: 廣搜（較少 epoch，快速探索）
            project_name = f'ATT_{model_type}_{stock_id}'
            tuner_stage1 = TunerCV(
                hypermodel=HyperTCN(input_shape=init_input_shape, strategy=strategy), # 需先提供有效 shape；run_trial 會依 lookback 動態覆寫
                oracle=kt.oracles.BayesianOptimizationOracle(
                    objective=kt.Objective('val_fbeta_score', 'max'),
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

            # Stage 2: 精搜（延續同一 project，增加 trial 上限並用較多 epoch）
            try:
                tuner_stage2 = TunerCV(
                    hypermodel=HyperTCN(input_shape=init_input_shape, strategy=strategy),
                    oracle=kt.oracles.BayesianOptimizationOracle(
                        objective=kt.Objective('val_fbeta_score', 'max'),
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
                        objective=kt.Objective('val_fbeta_score', 'max'),
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

            # 將最佳 trial 摘要（含 lookback_window）寫入檔案，供固定參數訓練腳本直接讀取
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
            # 發生例外時保留最精簡 traceback，並記錄失敗組合，避免整批中斷
            traceback.print_exc(limit=1, file=sys.stdout)
            error_list.append([stock_id, model_type])
            continue
        # 每個標的訓練完成後清理 graph 與 Python 記憶體
        K.clear_session()    
        gc.collect()

