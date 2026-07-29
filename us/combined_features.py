"""Combined 5-aspect feature loader for ATT+Flood_combined.py / ATT+Dflooding_combined.py.

把 fundamental / trade / moment / tech_trend / macro 五個面向的特徵在「aspect-specific
expand 之後 / sanitize-corr-scaler 之前」沿 column 軸 inner-join concat，作為單一
`combined5` aspect 進入後續訓練流程。

Sentiment 刻意排除，與 run5 5-aspect 比較對齊。

對外介面：
    COMBINED_ASPECT          'combined5'
    SOURCE_ASPECTS           ['fundamental', 'trade', 'moment', 'tech_trend', 'macro']
    load_and_combine_features(stock_id) -> (combined_df, train_start_str)

備註：
    這裡刻意「複製」`ATT+Dflooding_floodexp.py` 內的 `_expand_*` / `_prepare_trade_features` /
    `_detect_non_zero_date` 與其常數，避免 import 大型主腳本時觸發 TF 初始化與 main 迴圈。
    若日後 expand 規則改動，需同步更新此檔。
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 對外常數
# ----------------------------------------------------------------------

COMBINED_ASPECT = 'combined5'
SOURCE_ASPECTS: List[str] = ['fundamental', 'trade', 'moment', 'tech_trend', 'macro']

# 與 ATT+Dflooding_floodexp.py 的 _detect_non_zero_date 同步：對應每個 aspect 的最低
# 「非零欄位比例」門檻；找最早一個達到門檻的日期作為該 aspect 的 train_start 候選。
ASPECT_RATIO_THRESHOLD = {
    'fundamental': 0.0,
    'trade': 0.5,
    'moment': 0.5,
    'tech_trend': 0.5,
    'macro': 0.9,
}


def platform_path(path_str: str) -> str:
    """Windows D:/... → Linux/WSL /mnt/d/...；Windows 維持原樣。"""
    if os.name != 'nt' and len(path_str) >= 2 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path_str


DATA_ROOT = platform_path(os.getenv('DATA_ROOT', 'D:/Feature_new'))


# ----------------------------------------------------------------------
# 以下為 ATT+Dflooding_floodexp.py 的 expand 函式複製（保持邏輯一致）
# ----------------------------------------------------------------------

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


def _detect_non_zero_date(df: pd.DataFrame, ratio_threshold: float = 0.0):
    feat = df.iloc[:, :-4]
    if ratio_threshold <= 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    else:
        idx = feat.index[(feat != 0).mean(axis=1) >= ratio_threshold]
    if len(idx) == 0:
        idx = feat.index[~(feat == 0).all(axis=1)]
    return idx[0] if len(idx) else feat.index[0]


def _prepare_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in TRADE_HEAVY_TAIL_COLS:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors='coerce').astype(np.float64)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[c] = np.sign(s) * np.log1p(np.abs(s))
    return out


def _expand_macro_features(df: pd.DataFrame) -> pd.DataFrame:
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


EXPAND_DISPATCH = {
    'fundamental': _expand_fundamental_features,
    'trade': _prepare_trade_features,
    'moment': _expand_moment_features,
    'tech_trend': _expand_tech_trend_features,
    'macro': _expand_macro_features,
}


# ----------------------------------------------------------------------
# 對外 loader
# ----------------------------------------------------------------------

def _aspect_train_start(df: pd.DataFrame, aspect: str) -> pd.Timestamp:
    """依面向決定 train_start 候選日（與 _floodexp 內邏輯一致）。"""
    if aspect == 'fundamental':
        temp = df.iloc[:, :-4]
        nonzero = temp.index[~(temp == 0).all(axis=1)]
        if len(nonzero) == 0:
            return temp.index[0]
        return nonzero[0]
    ratio = ASPECT_RATIO_THRESHOLD.get(aspect, 0.5)
    return _detect_non_zero_date(df, ratio_threshold=ratio)


def load_and_combine_features(stock_id: str) -> Tuple[pd.DataFrame, str]:
    """讀取 5 個 aspect 的 CSV，套用各自 expand，沿 axis=1 inner-join concat。

    回傳：
        (combined_df, train_start_str)
        combined_df：欄位順序為 [aspect1_feat_1, aspect1_feat_2, ..., aspectN_feat_*, y_10, y_20, y_40, y_60]
        train_start_str：5 個 aspect 中最晚的 train_start + 1 天，格式 'YYYY-MM-DD'

    保證：
        - 標籤 y_10/y_20/y_40/y_60 來自第一個 aspect，並 assert 與其他 4 個 aspect 完全一致
        - 欄位名稱衝突時，後者加上 `_{aspect}` 後綴
    """
    aspect_frames = {}
    aspect_starts = {}

    for aspect in SOURCE_ASPECTS:
        csv_path = f'{DATA_ROOT}/{aspect}_{stock_id}.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'[combined_features] missing CSV: {csv_path}')
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        expander = EXPAND_DISPATCH[aspect]
        df = expander(df)

        # 確認最後 4 欄為 y_10/y_20/y_40/y_60（順序需與其他 aspect 對得上）
        if df.shape[1] < 5:
            raise ValueError(f'[combined_features] {aspect}: too few columns after expand ({df.shape[1]})')
        label_cols = df.columns[-4:].tolist()
        expected = ['y_10', 'y_20', 'y_40', 'y_60']
        if label_cols != expected:
            raise ValueError(
                f'[combined_features] {aspect}: label columns mismatch, got {label_cols}, expected {expected}'
            )

        aspect_frames[aspect] = df
        aspect_starts[aspect] = _aspect_train_start(df, aspect)

    # 取共同日期（inner join）
    common_index = None
    for aspect, df in aspect_frames.items():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) == 0:
        raise ValueError('[combined_features] no overlapping dates across aspects')
    common_index = common_index.sort_values()

    # 取第一個 aspect (fundamental) 的標籤作為基準；其他 aspect 若不一致只警告
    # （常見原因：trade 在無資料早期將 label 補 0；slice 到 train_start 後就只剩 fundamental 標籤）
    first_aspect = SOURCE_ASPECTS[0]
    base_labels = aspect_frames[first_aspect].loc[common_index].iloc[:, -4:]
    for aspect in SOURCE_ASPECTS[1:]:
        other = aspect_frames[aspect].loc[common_index].iloc[:, -4:]
        diff_mask = (base_labels.fillna(-999) != other.fillna(-999)).any(axis=1)
        n_diff = int(diff_mask.sum())
        if n_diff > 0:
            print(
                f'[combined_features][WARN] label mismatch {first_aspect} vs {aspect}: '
                f'{n_diff} rows differ (will use {first_aspect} labels)'
            )

    # 取每個 aspect 的特徵欄位（去掉最後 4 欄標籤），沿 axis=1 concat
    feature_blocks = []
    seen_cols: set = set()
    for aspect in SOURCE_ASPECTS:
        block = aspect_frames[aspect].loc[common_index].iloc[:, :-4].copy()
        # 欄位名衝突時加 `_{aspect}` 後綴
        rename_map = {}
        for col in block.columns:
            new_name = col
            if new_name in seen_cols:
                new_name = f'{col}_{aspect}'
            seen_cols.add(new_name)
            rename_map[col] = new_name
        block = block.rename(columns=rename_map)
        feature_blocks.append(block)

    combined = pd.concat(feature_blocks + [base_labels], axis=1)

    # train_start: 取所有 aspect 中最晚的，並 +1 天（與 _floodexp 行為一致）
    latest_start = max(aspect_starts.values())
    train_start_str = (latest_start.to_pydatetime().date() + timedelta(days=1)).strftime('%Y-%m-%d')

    print(
        f'[combined_features] stock={stock_id} n_dates={len(common_index)} '
        f'n_features={combined.shape[1] - 4} train_start={train_start_str}'
    )
    print(
        f'[combined_features] per-aspect n_features = '
        + ', '.join(f'{a}:{aspect_frames[a].shape[1] - 4}' for a in SOURCE_ASPECTS)
    )
    print(
        f'[combined_features] per-aspect train_start = '
        + ', '.join(f'{a}:{aspect_starts[a].date()}' for a in SOURCE_ASPECTS)
    )

    return combined, train_start_str


if __name__ == '__main__':
    # 簡易自測
    stock = os.getenv('STOCK_IDS', '2330').split(',')[0].strip()
    df, ts = load_and_combine_features(stock)
    print(f'[selftest] shape={df.shape} train_start={ts}')
    print(f'[selftest] head columns: {df.columns[:5].tolist()}')
    print(f'[selftest] tail columns: {df.columns[-6:].tolist()}')
