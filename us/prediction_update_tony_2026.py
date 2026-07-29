import sys, json, joblib, gc
import zipfile
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 預設走 CPU 以避免 WSL multiprocessing 下 CUDA 初始化失敗；
# 若要啟用 GPU，請設定 ATT_PREDICT_USE_GPU=1。
# 注意：不要在 import tensorflow 之前設定 CUDA_VISIBLE_DEVICES=-1，
# stable TF 2.21 在 WSL2 上會於首次 eager context 初始化時 glibc double free。
# 改成 import 之後用 tf.config.set_visible_devices 停用 GPU。
_USE_GPU = os.getenv('ATT_PREDICT_USE_GPU', '0') == '1'

import keras
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp

if not _USE_GPU:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass


def platform_path(path_str):
    """Normalize Windows paths for current runtime (Windows/WSL)."""
    if os.name == 'nt' or len(path_str) < 2 or path_str[1] != ':':
        return path_str

    drive = path_str[0].lower()
    rest = path_str[2:].replace('\\', '/')
    candidates = [
        f"/mnt/{drive}{rest}",
        f"/mnt/host/{drive}{rest}",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # fallback for yet-to-be-created files
    return candidates[0]


FEATURE_ROOT = platform_path("D:/Feature_new")
FEATURE_SELECTION_ROOT = platform_path("D:/feature_selection_test")
EXPERIMENT_ROOT = platform_path("D:/experiments_df_test")
SCALER_ROOT = platform_path("D:/feature_scaler_test")
DES_MODEL_ROOT = platform_path("D:/DES_model_test")
DES_PRED_ROOT = platform_path("D:/model_pred_DES_test")


# trade 重尾欄位清單，必須與 ATT+Flood.py / ATT+Dflooding.py 保持一致
TRADE_HEAVY_TAIL_COLS = (
    'foreign_cap_ratio', 'invst_cap_ratio', 'ins_nbd', 'Force_nbd', 'smr'
)


def _prepare_trade_features(df):
    """推論端對 trade 重尾欄位做 signed log1p（與訓練時一致）。"""
    out = df.copy()
    for c in TRADE_HEAVY_TAIL_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors='coerce').astype(np.float64)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s))
    return out


def _expand_sentiment_features(df, stock_id):
    """推論端對 sentiment CSV 重現訓練時的衍生特徵（必須與 ATT+Flood.py /
    ATT+Dflooding.py 裡的 `_expand_sentiment_features` 完全一致，否則 scaler
    / feature_selection 欄位對不上）。"""
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


# macro 欄位分組（需與 ATT+Flood.py / ATT+Dflooding.py 保持完全一致）
MACRO_STATIONARY_COLS = ('Price_rate_3m', 'Price_rate_10y', 'Price_FX', 'Price_VIX')
MACRO_PRICE_LEVEL_COLS = (
    'Price_oil', 'Price_gold', 'Price_copper',
    'Price_S&P500', 'Price_Nasdaq', 'Price_SOX',
    'Price_CRB', 'Price_BDI',
)
MACRO_VOL_COLS = ('Price_TX03C', 'Price_TX03P')
MACRO_SIGNED_LEVEL_COLS = ('Price_TX03F',)


def _expand_macro_features(df):
    """推論端對 macro CSV 重現訓練時的衍生特徵（必須與訓練端完全一致）。"""
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
# tech_trend / fundamental / moment expander（推論端必須與 ATT+Flood.py /
# ATT+Dflooding.py 的實作完全一致，否則 scaler / feature 欄位會對不上）。
# ===================================================================


def _expand_tech_trend_features(df):
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


def _expand_fundamental_features(df):
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


def _expand_moment_features(df):
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


# 允許載入含 Lambda 層的模型（模型來自自己的 AutoML 訓練）
keras.config.enable_unsafe_deserialization()

# --------------------------------------------------------------------------
# Eager context 預熱：某些 TF build（特別是 WSL2 + stable TF 2.21）在首次
# 於 Lambda 反序列化路徑內呼叫 tf.constant 時，會在 _initialize_physical_devices
# 觸發 glibc 'free(): double free detected in tcache 2' 並 abort。於 import
# 階段先以一個 trivial tf.constant 強制 eager context 完成初始化，後續的
# tf.constant / MHA.build 便不會再進到那個崩潰路徑。
# --------------------------------------------------------------------------
try:
    _ = tf.constant([0.0]).numpy()
except Exception:
    pass

# Lambda 層反序列化時需要 tf/np 在 globals 中（bytecode 中的自由變數）
# 同時 closure 中的 __tensor__ 需轉回 tf.constant
import keras.src.utils.python_utils as _kpu
_kpu.tf = tf
_kpu.np = np
_orig_func_load = _kpu.func_load

def _patched_func_load(code, defaults=None, closure=None, globs=None):
    """Keras 3 將 Lambda closure 裡的 tf.Tensor 序列化為 {'class_name': '__tensor__', ...}
    dict；反序列化時需要轉回實際的 tf.constant，Lambda 的 bytecode 才能正確使用。
    eager context 已在 import 時預熱，這裡的 tf.constant 不會再觸發 double free。"""
    if closure is not None:
        new_closure = []
        for c in closure:
            if isinstance(c, dict) and c.get('class_name') == '__tensor__':
                cfg = c['config']
                dtype = cfg.get('dtype', 'float32')
                arr = np.ascontiguousarray(np.asarray(cfg['value'], dtype=np.dtype(dtype)))
                new_closure.append(tf.constant(arr))
            else:
                new_closure.append(c)
        closure = new_closure
    return _orig_func_load(code, defaults, closure, globs)
_kpu.func_load = _patched_func_load

# --------------------------------------------------------------------------
# Keras 2 時代存檔的 MultiHeadAttention legacy inbound_nodes 只帶一個
# positional 張量（value 走 kwargs），而 Keras 3 的 legacy 反序列化路徑
# 在呼叫 layer.build 時可能只帶 query_shape。若缺少 value_shape，就以
# query_shape 代入（自注意力模型皆 query==value==key 同 shape）。
# --------------------------------------------------------------------------
try:
    _mha_cls = tf.keras.layers.MultiHeadAttention
    _orig_mha_build = _mha_cls.build
    def _patched_mha_build(self, query_shape, value_shape=None, key_shape=None):
        if value_shape is None:
            value_shape = query_shape
        if key_shape is None:
            key_shape = value_shape
        return _orig_mha_build(self, query_shape, value_shape, key_shape)
    _mha_cls.build = _patched_mha_build
except Exception:
    pass

# --------------------------------------------------------------------------
# Keras 3 的 functional.deserialize_node 在 legacy 路徑下只會回傳一個
# positional arg（list of tensors），導致 MHA 等多位置參數的 layer 無法
# 正常 call。我們偵測自訂的「每個 input_data 都有獨立 4-tuple 且只有最
# 後一個帶 kwargs」模式時，改成回傳多個 positional arg。
# --------------------------------------------------------------------------
try:
    import keras.src.models.functional as _kf
    _orig_deserialize_node = _kf.deserialize_node

    def _patched_deserialize_node(node_data, created_layers):
        if isinstance(node_data, list) and node_data and all(
            isinstance(x, list) and len(x) >= 3 for x in node_data
        ):
            # 嘗試從 created_layers 找出目標 layer 類別：
            # 只有 MultiHeadAttention 這類「多個 positional 張量 + kwargs」的 layer 需要
            # 把 list of tensors 拆成多個 positional args；Add / Concatenate 等仍應
            # 走原本「單一 list 參數」路徑。
            # 由於 deserialize_node 不知道當前 layer 是哪個，我們以 node_data 中是否
            # 有任何 entry 攜帶 tensor-ref 形式的 kwarg（e.g. attention_mask）作為判斷：
            # 若有則屬於 MHA-style 呼叫，需要 unpack 成多個 positional args。
            has_tensor_kwarg = False
            for input_data in node_data:
                if len(input_data) >= 4 and isinstance(input_data[3], dict):
                    for v in input_data[3].values():
                        if (isinstance(v, list) and len(v) >= 3
                            and isinstance(v[0], str)
                            and v[0] in created_layers):
                            has_tensor_kwarg = True; break
                if has_tensor_kwarg: break
            if not has_tensor_kwarg:
                return _orig_deserialize_node(node_data, created_layers)
            input_tensors = []
            kwargs = {}
            for input_data in node_data:
                name = input_data[0]
                ni = input_data[1]
                ti = input_data[2]
                cur_kw = input_data[3] if len(input_data) >= 4 and isinstance(input_data[3], dict) else {}
                layer = created_layers.get(name)
                if layer is None:
                    return _orig_deserialize_node(node_data, created_layers)
                if len(layer._inbound_nodes) <= ni:
                    return _orig_deserialize_node(node_data, created_layers)
                node = layer._inbound_nodes[ni]
                input_tensors.append(node.output_tensors[ti])
                kwargs = cur_kw or kwargs
            # 解析 kwargs 中仍為 [layer,0,0] 形式的 tensor ref
            resolved_kwargs = {}
            for k, v in kwargs.items():
                if (isinstance(v, list) and len(v) >= 3
                    and isinstance(v[0], str) and v[0] in created_layers):
                    _layer = created_layers[v[0]]
                    if len(_layer._inbound_nodes) > v[1]:
                        resolved_kwargs[k] = _layer._inbound_nodes[v[1]].output_tensors[v[2]]
                    else:
                        resolved_kwargs[k] = v
                else:
                    resolved_kwargs[k] = v
            return input_tensors, resolved_kwargs
        return _orig_deserialize_node(node_data, created_layers)

    _kf.deserialize_node = _patched_deserialize_node
except Exception:
    pass

# Keras 3 Lambda 層 shape 推斷 workaround：pos_emb / temperature 皆保持原 shape
_orig_lambda_compute = keras.layers.Lambda.compute_output_shape
def _lambda_compute_fallback(self, input_shape):
    try:
        return _orig_lambda_compute(self, input_shape)
    except NotImplementedError:
        return input_shape
keras.layers.Lambda.compute_output_shape = _lambda_compute_fallback

# TF2 GPU 記憶體設定（按需增長，避免一次佔滿）
# 只在啟用 GPU 時才探測裝置；CPU 模式下呼叫 list_physical_devices 在部分 TF 自建版
# 會觸發 CUDA 初始化並導致 glibc double free，因此直接跳過。
if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '-1':
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as _e:
        print(f"[warn] GPU probe skipped: {_e}")

def sequence_to_windows(seq, y, n_steps):
    _X, _y = [], []
    for i in range(seq.shape[0] - n_steps + 1):
        end_ix = i + n_steps
        _X.append(np.array(seq.iloc[i:end_ix]))
        _y.append(np.array(y.iloc[end_ix-1]))
    _X, _y = np.array(_X), np.array(_y)
    return _X, _y

def get_windows(X, y, slice, steps):
    min_idx = 1 * 250 - 1
    start_idx, end_idx, _ = slice.start, slice.stop, slice.step
    start_idx = max(min_idx, start_idx - steps + 1)
    end_idx = end_idx - steps + 1
    return X[start_idx:end_idx], y[start_idx:end_idx]


def apply_sanitize_statistics(df, stats):
    """套用訓練期保存的 sanitize 統計量，避免推論分佈漂移。"""
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


def transform_features_with_pipeline(feature_df, pipeline_obj):
    """相容舊版 scaler 與新版 preprocess bundle（含 pass-through 模式）。"""
    if isinstance(pipeline_obj, dict) and 'scaler' in pipeline_obj:
        scaler = pipeline_obj['scaler']
        sanitize_stats = pipeline_obj.get('sanitize_stats')
        transformed_input = feature_df
        if sanitize_stats is not None:
            transformed_input = apply_sanitize_statistics(feature_df, sanitize_stats)
        if scaler is None:
            # pass-through bundle：訓練時關閉前處理，推論時也不做 scaling
            transformed = transformed_input.to_numpy(dtype=np.float64, copy=True)
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            transformed = scaler.transform(transformed_input)
    else:
        transformed = pipeline_obj.transform(feature_df)

    return pd.DataFrame(transformed, index=feature_df.index, columns=feature_df.columns)


# ====== 自定義層：SinusoidalPositionalEncoding（與訓練時同名，相容舊模型檔）======
@tf.keras.utils.register_keras_serializable()
class SinusoidalPositionalEncoding(layers.Layer):
    """固定式正弦/餘弦位置編碼層（無可訓練參數）。"""
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
        encoding = tf.cast(encoding, x.dtype)
        return x + encoding

    def get_config(self):
        return super().get_config()


# ====== 自定義層：TemperatureScaling ======
@tf.keras.utils.register_keras_serializable()
class TemperatureScaling(layers.Layer):
    """將 logits 除以固定 temperature（無可訓練參數），取代舊 Lambda 層。"""
    def __init__(self, temp=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temp = float(temp)

    def call(self, x):
        return x / tf.cast(self.temp, x.dtype)

    def get_config(self):
        base = super().get_config()
        base['temp'] = self.temp
        return base


def _is_hdf5_file(path):
    """偵測檔案內容是否為 HDF5（無論副檔名）。"""
    try:
        with open(path, 'rb') as f:
            return f.read(8) == b'\x89HDF\r\n\x1a\n'
    except Exception:
        return False


def _sanitize_h5_model_config(cfg, _parent_class=None):
    """遞迴清理 Keras 3 讀不懂的舊 H5 config：
    - dtype 欄位一律轉成純字串，避免舊 DTypePolicy 設定觸發
      'str' object has no attribute 'quantization_mode'。
    - MultiHeadAttention 的 config 在舊版存了 query_shape/key_shape/value_shape，
      新版以 build_config 處理，這些鍵會變成 Unrecognized keyword arguments。
    - 舊檔用 python marshal 序列化 Lambda bytecode，在不同 Python 版本
      會 'bad marshal data (unknown type code)'。若層名為 pos_emb / 含
      sinusoidal 字樣，直接改成我們自己的 SinusoidalPositionalEncoding 類別。
    """
    if isinstance(cfg, dict):
        cls = cfg.get('class_name')
        # 把舊版無法反序列化的 Lambda 改寫為 SinusoidalPositionalEncoding
        if cls == 'Lambda':
            inner = cfg.get('config') if isinstance(cfg.get('config'), dict) else {}
            lname = inner.get('name', '') if isinstance(inner, dict) else ''
            lname_lower = lname.lower()
            if lname == 'pos_emb' or 'sinusoidal' in lname_lower or 'pos_emb' in lname_lower:
                cfg['class_name'] = 'SinusoidalPositionalEncoding'
                cfg['registered_name'] = 'SinusoidalPositionalEncoding'
                cfg['module'] = None
                if isinstance(inner, dict):
                    for k in ('function', 'arguments', 'function_type',
                              'module', 'output_shape', 'output_shape_type',
                              'output_shape_module', 'mask'):
                        inner.pop(k, None)
                cfg.pop('build_config', None)
            elif 'temperature' in lname_lower or lname == 'temperature':
                # 從 closure 取出 temp 值；若找不到預設 1.0
                temp_val = 1.0
                try:
                    fn = inner.get('function') if isinstance(inner, dict) else None
                    closure = None
                    if isinstance(fn, dict):
                        fc = fn.get('config')
                        if isinstance(fc, dict):
                            closure = fc.get('closure')
                    if isinstance(closure, list):
                        for item in closure:
                            if isinstance(item, dict) and 'temp' in item and not item.get('class_name'):
                                temp_val = float(item['temp'])
                                break
                except Exception:
                    pass
                cfg['class_name'] = 'TemperatureScaling'
                cfg['registered_name'] = 'TemperatureScaling'
                cfg['module'] = None
                if isinstance(inner, dict):
                    for k in ('function', 'arguments', 'function_type',
                              'module', 'output_shape', 'output_shape_type',
                              'output_shape_module', 'mask'):
                        inner.pop(k, None)
                    inner['temp'] = temp_val
                cfg.pop('build_config', None)
        if 'dtype' in cfg:
            d = cfg['dtype']
            if isinstance(d, dict):
                name = None
                if isinstance(d.get('config'), dict):
                    name = d['config'].get('name')
                cfg['dtype'] = name or 'float32'
        # 對 MultiHeadAttention 的 inner config 去除不相容的 shape 鍵，
        # 並把它們搬到外層 build_config.shapes_dict，讓 Keras 3 的
        # build_from_config 能正確呼叫 build(query_shape, value_shape, key_shape)。
        if _parent_class == 'MultiHeadAttention' or cls == 'MultiHeadAttention':
            inner = cfg.get('config') if isinstance(cfg.get('config'), dict) else cfg
            qs = inner.pop('query_shape', None)
            ks = inner.pop('key_shape', None)
            vs = inner.pop('value_shape', None)
            if cls == 'MultiHeadAttention' and (qs or vs or ks):
                bc = cfg.get('build_config') if isinstance(cfg.get('build_config'), dict) else {}
                sd = bc.get('shapes_dict') if isinstance(bc.get('shapes_dict'), dict) else {}
                if qs is not None and 'query_shape' not in sd:
                    sd['query_shape'] = qs
                if vs is not None and 'value_shape' not in sd:
                    sd['value_shape'] = vs
                if ks is not None and 'key_shape' not in sd:
                    sd['key_shape'] = ks
                bc['shapes_dict'] = sd
                cfg['build_config'] = bc
        # MHA 的 legacy inbound_nodes 裡，`value` / `key` 走 kwargs，Keras 3
        # functional_from_config 不會把 kwargs 中的 [layer,0,0] 解析回 tensor。
        # 把 value / key 從 kwargs 提升到 positional args。
        if cls == 'MultiHeadAttention':
            ibn = cfg.get('inbound_nodes')
            if isinstance(ibn, list) and ibn and isinstance(ibn[0], list):
                new_nodes = []
                for node in ibn:
                    # legacy 格式：node 是 list of arg spec，每個 arg spec 形如
                    # [layer_name, node_idx, tensor_idx, kwargs_dict]
                    if not node or not isinstance(node[0], list):
                        new_nodes.append(node); continue
                    query_spec = node[0]
                    if not (isinstance(query_spec, list) and len(query_spec) >= 3):
                        new_nodes.append(node); continue
                    kw = query_spec[3] if len(query_spec) >= 4 and isinstance(query_spec[3], dict) else {}
                    new_args = [[query_spec[0], query_spec[1], query_spec[2], {}]]
                    new_kw = dict(kw)
                    val_ref = new_kw.pop('value', None)
                    key_ref = new_kw.pop('key', None)
                    if isinstance(val_ref, list) and len(val_ref) >= 3:
                        new_args.append([val_ref[0], val_ref[1], val_ref[2], {}])
                    if isinstance(key_ref, list) and len(key_ref) >= 3:
                        new_args.append([key_ref[0], key_ref[1], key_ref[2], {}])
                    # attention_mask 等其他 tensor kwargs 維持原樣（Keras 3 legacy
                    # 會保留它們為 list，但 MHA 走 compute_output_spec 只會檢查 shape
                    # 的是 query/value/key；attention_mask 在 call 時 resolve）。
                    # 把殘餘 kwargs 併到最後一個 positional arg 上。
                    if new_kw:
                        new_args[-1][3] = new_kw
                    new_nodes.append(new_args)
                cfg['inbound_nodes'] = new_nodes
        for k, v in cfg.items():
            _sanitize_h5_model_config(v, _parent_class=cls if cls else _parent_class)
    elif isinstance(cfg, list):
        for v in cfg:
            _sanitize_h5_model_config(v, _parent_class=_parent_class)
    return cfg


def _rewrite_h5_with_sanitized_config(src_path, dst_path):
    """複製 H5 到 dst_path，並將內嵌 model_config 字串中的 dtype 設定正規化。"""
    import h5py, shutil
    shutil.copyfile(src_path, dst_path)
    with h5py.File(dst_path, 'r+') as f:
        raw = f.attrs.get('model_config')
        if raw is None:
            return
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        cfg = json.loads(raw)
        _sanitize_h5_model_config(cfg)
        f.attrs['model_config'] = json.dumps(cfg)


def _load_hdf5_via_model_from_json(h5_path, custom_objects):
    """繞過 Keras 3 legacy H5 loader 的 native 崩潰：
    1) 讀 H5 內嵌 model_config、清理後用 model_from_json 重建結構
    2) 用 load_weights 把權重載回來（跳過優化器狀態）
    FloodingModel 以 Functional API 建構；在 from_config 時將 class_name 改為
    Functional，讓 Keras 走標準 Functional 重建流程。"""
    import h5py
    try:
        from keras.src.models.functional import Functional as _FunctionalCls
    except Exception:
        try:
            from tensorflow.keras.models import Functional as _FunctionalCls
        except Exception:
            _FunctionalCls = None

    with h5py.File(h5_path, 'r') as f:
        raw = f.attrs.get('model_config')
        if raw is None:
            raise ValueError(f"HDF5 has no embedded model_config: {h5_path}")
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        cfg = json.loads(raw)
    cfg = _sanitize_h5_model_config(cfg)
    if cfg.get('class_name') == 'FloodingModel':
        # 推論時 FloodingModel 與 Functional 行為一致
        cfg['class_name'] = 'Functional'
    merged_co = dict(custom_objects)
    if _FunctionalCls is not None:
        merged_co['Functional'] = _FunctionalCls
    model = tf.keras.models.model_from_json(json.dumps(cfg), custom_objects=merged_co)
    model.load_weights(h5_path)
    return model


def _load_keras_any_format(path, custom_objects):
    """載入 .keras（ZIP）或 HDF5 模型；推論用途一律 compile=False。
    對「.keras 副檔名但實際為 HDF5」的舊檔：
    因 Keras 3 legacy H5 loader 在特定 TF 2.20 自建版會發生原生崩潰
    （free(): double free in tcache），一律改走 model_from_json + load_weights，
    從 H5 讀出 model_config 清理後重建拓撲，再 load_weights 填回權重。
    """
    if _is_hdf5_file(path):
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp(prefix='keras_legacy_')
        try:
            src = os.path.join(tmp_dir, os.path.basename(path).rsplit('.', 1)[0] + '.h5')
            shutil.copyfile(path, src)
            fixed = os.path.join(tmp_dir, 'fixed.h5')
            _rewrite_h5_with_sanitized_config(src, fixed)
            # 優先用 Keras 3 內建 legacy H5 loader（已 sanitize 過 config）。
            try:
                return tf.keras.models.load_model(fixed, custom_objects=custom_objects, compile=False)
            except Exception:
                # 最後以 model_from_json + load_weights 重建。
                return _load_hdf5_via_model_from_json(fixed, custom_objects)
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)


def load_model_with_fallback(model_path):
    """優先一般載入，若遇到 CUDA 初始化錯誤則回退 CPU 載入。
    另外相容「.keras 副檔名但實際為 HDF5」的舊檔。"""
    custom_objects = {
        'FloodingModel': FloodingModel,
        'DynamicCausalMask': DynamicCausalMask,
        'SinusoidalPositionalEncoding': SinusoidalPositionalEncoding,
        'TemperatureScaling': TemperatureScaling,
    }
    try:
        return _load_keras_any_format(model_path, custom_objects)
    except tf.errors.InternalError as model_error:
        error_text = str(model_error)
        if ('CUDA_ERROR' in error_text) or ('cuDeviceGet' in error_text):
            print(f"[ATT][WARN] fallback to CPU model load: {os.path.basename(model_path)}")
            with tf.device('/CPU:0'):
                return _load_keras_any_format(model_path, custom_objects)
        raise

# ====== 自定義層：DynamicCausalMask（Attention 因果遮罩）======
@tf.keras.utils.register_keras_serializable()
class DynamicCausalMask(layers.Layer):
    """產生動態因果遮罩（下三角矩陣），使 Attention 只能看到過去的時間步。"""
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return tf.linalg.band_part(
            tf.ones((1, seq_len, seq_len), dtype=tf.float32), -1, 0
        )
    def get_config(self):
        return super().get_config()

# ====== 自定義模型：FloodingModel（Flooding 正則化訓練策略）======
@tf.keras.utils.register_keras_serializable()
class FloodingModel(tf.keras.Model):
    """使用 Flooding loss 的自定義 Keras Model。
    flooding_b: flooding 水位 (b 值)，loss = |loss - b| + b
    """
    flooding_b = 0.10

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred,
                sample_weight=sample_weight,
            )
            if self.flooding_b > 0:
                loss = tf.math.abs(loss - self.flooding_b) + self.flooding_b

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

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

def _get_n_steps_from_keras(filepath):
    """從 .keras 模型檔讀取 InputShape 中的 n_steps（序列長度）。
    支援 Keras 3 ZIP 格式和 HDF5 格式。"""
    try:
        with zipfile.ZipFile(filepath) as z:
            cfg = json.loads(z.read('config.json'))
            layers_list = cfg['config']['layers']
            input_layer = [l for l in layers_list if l['class_name'] == 'InputLayer'][0]
            return input_layer['config']['batch_shape'][1]  # [None, n_steps, n_features]
    except zipfile.BadZipFile:
        # HDF5 格式：需要載入模型後讀取
        import h5py
        with h5py.File(filepath, 'r') as f:
            if 'model_config' in f.attrs:
                cfg = json.loads(f.attrs['model_config'])
                layers_list = cfg['config']['layers']
                input_layer = [l for l in layers_list if l['class_name'] == 'InputLayer'][0]
                return input_layer['config']['batch_input_shape'][1]
        raise ValueError(f"Cannot read n_steps from {filepath}")
        
def _prediction_update(target):

    model_names = ['fundamental', 'tech_trend', 'moment', 'trade', 'macro', 'sentiment']
    X_all = pd.DataFrame()

    for trained_model in model_names:
        df_all = []

        data = pd.read_csv(f"{FEATURE_ROOT}/{trained_model}_{target}.csv", index_col=0, parse_dates=True, dtype={'y_10':str, 'y_20': str, 'y_40': str, 'y_60': str})

        # 確保索引為單調遞增且無重複（sentiment 等 CSV 可能因 append 累積導致亂序/重複，
        # 後續 data.index[-20] 與 slice_indexer 都需要排序好的索引）
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        if data.index.has_duplicates:
            data = data[~data.index.duplicated(keep='last')]

        # sentiment 面向：將 US/TW/ticker 三欄展開為衍生特徵，必須與訓練時一致
        if trained_model == 'sentiment':
            data = _expand_sentiment_features(data, target)

        # trade 面向：重尾欄位先做 signed log1p（與訓練時一致）
        if trained_model == 'trade':
            data = _prepare_trade_features(data)

        # macro 面向：展開為近平穩衍生特徵（與訓練時一致）
        if trained_model == 'macro':
            data = _expand_macro_features(data)

        # tech_trend 面向：raw OHLCV 轉為尺度無關特徵（與訓練時一致）
        if trained_model == 'tech_trend':
            data = _expand_tech_trend_features(data)

        # fundamental 面向：bounded rescale + 成長率 clip + signed log1p（與訓練時一致）
        if trained_model == 'fundamental':
            data = _expand_fundamental_features(data)

        # moment 面向：bounded 置中 + 硬 clip + vpt diff z（與訓練時一致）
        if trained_model == 'moment':
            data = _expand_moment_features(data)

        test_start, test_end = data.index[-20], data.index.max().strftime('%Y-%m-%d')

        try:
            fs = json.load(open(f"{FEATURE_SELECTION_ROOT}/{target}.json"))[trained_model]
            # 只使用 feature selection 和 CSV 中都存在的欄位
            available = [f for f in fs if f in data.columns]
            missing = [f for f in fs if f not in data.columns]
            # 先記住 y_* 標籤欄（在補欄位前抓，避免 append 後位置偏移）
            tail_cols = data.columns[-4:].tolist()
            if missing:
                print(f"[ATT][WARN] {trained_model}_{target}: {len(missing)} feature(s) missing, using {len(available)}/{len(fs)}")
                # 完全沒有可用特徵時直接跳過（如 sentiment 0/17）
                if not available:
                    print(f"[ATT][WARN] {trained_model}_{target}: no features available, skip")
                    continue
                # 補齊缺少欄位為 0，維持 fs 原順序，確保 scaler/model 輸入形狀一致
                for f in missing:
                    data[f] = 0.0
            data = data[fs + tail_cols]

            forecast_days = 20

            # 從第一個 .keras 模型檔動態讀取 n_steps（序列長度）
            file_path = glob(f"{EXPERIMENT_ROOT}/ATT_{trained_model}_{target}/experiment_*.keras")
            files = [y.replace('\\', '/') for y in file_path]
            if not files:
                print(f"[ATT] No model files found: ATT_{trained_model}_{target}")
                continue

            n_steps = _get_n_steps_from_keras(files[0])

            pipeline = joblib.load(f"{SCALER_ROOT}/scaler_{trained_model}_{target}.pkl")
            transformed_features = transform_features_with_pipeline(data.iloc[:, :-4], pipeline)
            data = pd.concat([transformed_features, data.iloc[:, -4:]], axis=1)

            X, y = sequence_to_windows(seq=data.iloc[:, :-4], y=data[f"y_{forecast_days}"], n_steps=n_steps)
            test_slice = data.index.slice_indexer(start=test_start, end=test_end)

            X_test, y_test = get_windows(X, y, test_slice, n_steps) # for the last month
            if len(X_test) == 0:
                print(f"[ATT][WARN] {trained_model}_{target}: no test window generated, skip")
                continue


            for file in files:
                model = load_model_with_fallback(f"{file}")
                csv_path = file.replace('experiment_','experiment_result_')
                csv_path = csv_path.replace('keras','csv')
                logits_new = pd.Series(model.predict(X_test)[:, 1], index=data.index[-len(X_test):])
                logits_old = pd.read_csv(csv_path, header=None, index_col=0, parse_dates=True).squeeze("columns")
                # 只追加歷史 cache 沒有的新日期；舊值優先 (keep='first')，避免
                # patched loader 重算的最後 20 天覆寫掉訓練時寫入的 logits。
                logits_new = pd.concat([logits_old, logits_new])
                logits_new = logits_new.loc[~logits_new.index.duplicated(keep='first')].sort_index()

                del model
                gc.collect()
                tf.keras.backend.clear_session()
                keras.config.enable_unsafe_deserialization()
                logits_new.to_csv(csv_path, header=False, encoding='utf-8')
                df_all.append(logits_new)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ATT][{exc_type.__name__}] line {exc_tb.tb_lineno}: {trained_model}_{target} — {e}")

        temp = pd.DataFrame(df_all).T
        temp = temp.mean(axis = 1)
        X_all = pd.concat([X_all,temp], axis = 1)

        tf.keras.backend.clear_session()
        keras.config.enable_unsafe_deserialization()

        
    X_all.index.name = 'Date'
    X_all.columns =  model_names
    # 與 DES_update_ATT-sentiment.py 的 X_all NaN 處理對齊，避免下游 DES_pred cache
    # 被改寫成「dropna 後子集 index」計算的機率，導致績效退化。
    X_all.dropna(how='all', inplace=True)
    X_all = X_all.ffill().bfill().fillna(0.5)
    X_all.index = pd.to_datetime(X_all.index)
    #cumSum = pd.read_csv(f'./features/cusum_{stock_id}.csv', index_col = 0, parse_dates=True)
    #X_all = X_all.merge(cumSum, left_index = True, right_index = True)
    X_all = X_all.astype('float64')   

    '''logits_new_all = pd.concat([
    pd.read_csv(f"{dir_path}/prediction/y_{i}_{target}.csv", header=None, index_col=0, parse_dates=True, squeeze=True)
    for i in model_names], axis=1).dropna()
    logits_new_all.columns = model_names'''

    #cusum_new = pd.read_csv(f'{dir_path}/features/cusum_{target}.csv', index_col=0, parse_dates=True, squeeze=True)
    #cusum_new.name = 'CUSUM'

    #logits_new_all = pd.concat([logits_new_all, cusum_new], axis=1, join='inner')
    logits_new_all = X_all
    if logits_new_all.empty:
        print(f"[ATT][WARN] {target}: no ATT logits available, skip DES update")
        return {
            'target': target,
            'status': 'skipped',
            'reason': 'no_att_logits'
        }

    des_model_candidates = glob(f"{DES_MODEL_ROOT}/DES_{target}_*.pkl")
    if not des_model_candidates:
        print(f"[DES][WARN] {target}: no DES model found")
        return {
            'target': target,
            'status': 'skipped',
            'reason': 'no_des_model'
        }

    DES_file_path = des_model_candidates[0]
    DES_file = DES_file_path.replace('\\', '/')
    DES_model = joblib.load(DES_file)

    des_pred_candidates = glob(f"{DES_PRED_ROOT}/DES_pred_{target}_*.csv")
    if not des_pred_candidates:
        print(f"[DES][WARN] {target}: no DES prediction csv found")
        return {
            'target': target,
            'status': 'skipped',
            'reason': 'no_des_prediction_csv'
        }
    DES_pred_path = [y.replace('\\', '/') for y in des_pred_candidates][-1]

    logits_ensemble_new = pd.Series(DES_model.predict_proba(logits_new_all)[:, 1], index=logits_new_all.index)
    logits_ensemble_old = pd.read_csv(DES_pred_path, header=None, index_col=0, parse_dates=True).squeeze("columns")[1:]
    # 只追加歷史 cache 沒有的新日期；舊值優先 (keep='first')，避免整段 DES_pred 被
    # 重算結果覆寫 (DES_update_ATT-sentiment.py 預設信任此 cache，覆寫會直接拖垮績效)。
    logits_ensemble_new = pd.concat([logits_ensemble_old, logits_ensemble_new])
    logits_ensemble_new.index = pd.to_datetime(logits_ensemble_new.index)
    logits_ensemble_new.index.name = 'Date'
    logits_ensemble_new = logits_ensemble_new.loc[~logits_ensemble_new.index.duplicated(keep='first')].sort_index()
    del DES_model
    gc.collect()
    logits_ensemble_new.to_csv(DES_pred_path, header=True, encoding='utf_8')
    
    return