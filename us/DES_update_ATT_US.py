# =============================================================================
# DES_update_ATT_US.py
# 功能：美股 (Dow 30) 版 Dynamic Ensemble Selection (DES) 流程。
#       讀取 4 個面向 (fundamental, tech_trend, moment, macro) 的 ATT 預測結果，
#       訓練 KNORAE 集成模型並產出 DES 預測 CSV、回測圖與績效報告。
#
# 與台股版 DES_update_ATT-sentiment.py 的差異：
#   * 面向：4 個 (fundamental, tech_trend, moment, macro)，不含 trade/sentiment
#   * 模型輸出：DES_model_US/  (DES pkl) 、 RF_model_US/  (RF pkl)
#   * 預測輸出：model_pred_DES_US/  、  model_pred_RF_US/
#   * CUSUM 來源：
#       out_dir   = cumSum_prob_12/cusum_{ticker}.csv  (方向 filter)
#       out_dir_P = cumSum_prob_6/cusum_{ticker}.csv   (機率混合)
#       兩者皆由 CumsumPro_US.py 產生，值為 (1-prob)*sign(cumulative)
#   * 股價：feature._us_data.load_price_frames(tickers) (yfinance + 快取)
#   * 標籤：feature/fundamental_{ticker}.csv 的 y_20 欄
#   * 美股交易慣例：按股 (不再 *1000)、零佣金 (BUY_FEE=SELL_FEE=0)、初始 1M USD
#   * Ticker 不再有 .TT/.TW 後綴；CmoneyFactor 與 Stock_name.csv 依賴移除
# =============================================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt


# --- CJK 字型 (圖表中混雜中文時備援) -------------------------------------- #
def _configure_cjk_font():
    from matplotlib import font_manager
    candidates = [
        'Microsoft JhengHei', 'Microsoft YaHei', 'PingFang TC', 'Heiti TC',
        'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
        'Noto Serif CJK TC', 'Noto Serif CJK JP',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing TW',
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [name for name in candidates if name in installed]
    if not picked:
        picked = sorted({f.name for f in font_manager.fontManager.ttflist if 'CJK' in f.name})
    if not picked:
        picked = ['DejaVu Sans']
    return picked


_CJK_FONT_STACK = _configure_cjk_font()


def _apply_cjk_rcparams():
    plt.rcParams['font.sans-serif'] = _CJK_FONT_STACK + ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


_apply_cjk_rcparams()


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from deslib.des.knora_e import KNORAE

import joblib
import warnings
import os
import sys
import glob
from pathlib import Path
warnings.filterwarnings("ignore")


# --- deslib 0.3.7 vs scikit-learn>=1.7 相容修補 --------------------------- #
import sklearn.base
if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
    sklearn.base.BaseEstimator._validate_data = lambda self, *args, **kwargs: _sklearn_validate_data(self, *args, **kwargs)


# =============================================================================
# Workspace / 路徑設定 (workspace-relative; WSL 與 Windows 通用)
# =============================================================================
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from feature._us_data import load_price_frames  # noqa: E402

EXPERIMENT_ROOT = _THIS_DIR / 'experiment'
FEATURE_ROOT    = _THIS_DIR / 'feature'

DES_MODEL_DIR   = _THIS_DIR / 'DES_model_US'
PRED_DES_DIR    = _THIS_DIR / 'model_pred_DES_US'
RF_MODEL_DIR    = _THIS_DIR / 'RF_model_US'
PRED_RF_DIR     = _THIS_DIR / 'model_pred_RF_US'

PRED_DIR_ABL      = _THIS_DIR / 'model_pred_DES_US_ablation'
DES_MODEL_DIR_ABL = _THIS_DIR / 'DES_model_US_ablation'
RF_MODEL_DIR_ABL  = _THIS_DIR / 'RF_model_US_ablation'

SHAP_OUTPUT_DIR = _THIS_DIR / 'model_output_US' / 'shap'
SHAP_FIG_DPI    = 300

EVAL_DIR     = _THIS_DIR / 'evaluation'
ENSEMBLE_DIR = _THIS_DIR / 'model_output_US'

for _p in (DES_MODEL_DIR, PRED_DES_DIR, RF_MODEL_DIR, PRED_RF_DIR,
           PRED_DIR_ABL, DES_MODEL_DIR_ABL, RF_MODEL_DIR_ABL,
           SHAP_OUTPUT_DIR, EVAL_DIR, ENSEMBLE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 全域參數
# =============================================================================
train_start = '2007-08-01'
train_end   = '2025-12-31'
test_start  = '2026-01-01'
now = datetime.strftime(datetime.now(), '%Y-%m-%d')

# 美股 4 個面向 (固定順序，影響 ablation letters)
FEATURE_ORDER = ['fundamental', 'tech_trend', 'moment', 'macro']
sub_cats = FEATURE_ORDER.copy()

# CUSUM 來源 (workspace-relative; 由 CumsumPro_US.py 產生)
CUSUM_DIR_SIGN = _THIS_DIR / 'cumSum_prob_12'   # 方向 filter
CUSUM_DIR_PROB = _THIS_DIR / 'cumSum_prob_6'    # 機率混合
out_dir   = str(CUSUM_DIR_SIGN)
out_dir_P = str(CUSUM_DIR_PROB)

# 美股交易慣例 (per-share, 零佣金, 1M USD 初始資金)
BUY_FEE         = 0.0
SELL_FEE        = 0.0
INITIAL_CAPITAL = 1_000_000.0  # USD

# Dow 30 公司名稱對照 (顯示用)
DOW30_NAME = {
    'AAPL': 'Apple', 'AMGN': 'Amgen', 'AMZN': 'Amazon', 'AXP': 'American Express',
    'BA': 'Boeing', 'CAT': 'Caterpillar', 'CRM': 'Salesforce', 'CSCO': 'Cisco',
    'CVX': 'Chevron', 'DIS': 'Walt Disney', 'GS': 'Goldman Sachs', 'HD': 'Home Depot',
    'HON': 'Honeywell', 'IBM': 'IBM', 'JNJ': 'Johnson & Johnson', 'JPM': 'JPMorgan Chase',
    'KO': 'Coca-Cola', 'MCD': "McDonald's", 'MMM': '3M', 'MRK': 'Merck',
    'MSFT': 'Microsoft', 'NKE': 'Nike', 'NVDA': 'NVIDIA', 'PG': 'Procter & Gamble',
    'SHW': 'Sherwin-Williams', 'TRV': 'Travelers', 'UNH': 'UnitedHealth', 'V': 'Visa',
    'VZ': 'Verizon', 'WMT': 'Walmart',
}

# 特定 ticker 的資料別名：輸出仍維持原 ticker，僅在讀取輸入資料時 fallback。
TICKER_DATA_ALIAS = {
    'GOOG': 'GOOGL',
}


def compute_letters(used_feats):
    """依固定順序將使用中特徵轉成短碼，作為 ablation cache key。"""
    ordered = [f for f in FEATURE_ORDER if f in used_feats]
    return ''.join(f[:2] for f in ordered)


def _safe_slug(value):
    return str(value).replace(' ', '_').replace(':', '-').replace('/', '-').replace('\\', '-')


def build_shap_tag(stock_id, period_key, used_feats):
    mode = 'full' if used_feats is None else f"abl_{compute_letters(used_feats)}"
    return f"{stock_id}_{_safe_slug(period_key)}_{mode}"


def _predict_pos_proba(estimator, X):
    """Safely return positive-class probability even for single-class fits."""
    proba = np.asarray(estimator.predict_proba(X), dtype='float64')
    if proba.ndim == 1:
        return proba

    classes = list(getattr(estimator, 'classes_', []))
    if proba.shape[1] == 1:
        # 單一類別模型：若唯一類別是 1，則機率即為該欄；否則全為 0。
        if classes and classes[0] == 1:
            return proba[:, 0]
        return np.zeros(proba.shape[0], dtype='float64')

    if classes and 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


# =============================================================================
# SHAP helpers
# =============================================================================
def _get_shap_module():
    try:
        import shap
        return shap
    except Exception as e:
        print(f"[SHAP][WARN] shap 無法載入，略過 explainability：{e}")
        return None


def build_kernel_explainer(des_model, X_train, background_k=50):
    shap = _get_shap_module()
    if shap is None:
        return None
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train 為空，無法建立 SHAP 背景資料")
    bg_k = max(1, min(int(background_k), len(X_train)))
    background = shap.kmeans(np.asarray(X_train, dtype='float64'), bg_k)
    explainer = shap.KernelExplainer(
        lambda x: _predict_pos_proba(des_model, np.asarray(x, dtype='float64')),
        background,
    )
    return explainer


def compute_shap_local(explainer, X_target, chunk_size=20, nsamples='auto'):
    if explainer is None:
        return None
    if X_target is None or len(X_target) == 0:
        raise ValueError("X_target 為空，無法計算 SHAP")
    total = len(X_target)
    local_blocks = []
    for start in range(0, total, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), total)
        batch = X_target.iloc[start:end]
        shap_vals = explainer.shap_values(np.asarray(batch, dtype='float64'), nsamples=nsamples)
        shap_arr = np.asarray(shap_vals, dtype='float64')
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)
        local_blocks.append(shap_arr)
    local_matrix = np.vstack(local_blocks)
    return pd.DataFrame(local_matrix, index=X_target.index, columns=X_target.columns)


def compute_shap_global(local_shap_df):
    if local_shap_df is None or local_shap_df.empty:
        raise ValueError("local_shap_df 為空，無法計算 global importance")
    g = np.abs(local_shap_df).mean(axis=0).sort_values(ascending=False)
    return pd.DataFrame({'feature': g.index, 'mean_abs_shap': g.values})


def save_shap_summary_plot(tag, X_target, local_shap_df):
    shap = _get_shap_module()
    if shap is None:
        return
    base = Path(SHAP_OUTPUT_DIR)
    base.mkdir(parents=True, exist_ok=True)
    summary_png = base / f"summary_{tag}.png"

    local_aligned = local_shap_df.copy()
    local_aligned.index = pd.to_datetime(local_aligned.index)
    common_idx = X_target.index.intersection(local_aligned.index)
    if len(common_idx) == 0:
        raise ValueError("X_target 與 local_shap_df 無重疊日期，無法繪製 summary")

    x_aligned = X_target.loc[common_idx]
    shap_aligned = local_aligned.loc[common_idx]

    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_aligned.values, x_aligned, plot_type='dot', show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(summary_png, facecolor='white', dpi=SHAP_FIG_DPI)
    plt.close(fig)
    print(f"[SHAP] 輸出完成: {summary_png}")


def save_shap_artifacts(tag, explainer, X_target, local_shap_df, global_df, waterfall_row=0):
    shap = _get_shap_module()
    if shap is None:
        return

    base = Path(SHAP_OUTPUT_DIR)
    base.mkdir(parents=True, exist_ok=True)
    local_csv     = base / f"local_{tag}.csv"
    global_csv    = base / f"global_{tag}.csv"
    waterfall_png = base / f"waterfall_{tag}.png"
    force_png     = base / f"force_{tag}.png"

    local_shap_df.to_csv(local_csv, index=True, encoding='utf-8')
    global_df.to_csv(global_csv, index=False, encoding='utf-8')

    save_shap_summary_plot(tag, X_target, local_shap_df)

    w_idx = int(waterfall_row)
    if w_idx < 0:
        w_idx = len(local_shap_df) + w_idx
    w_idx = max(0, min(w_idx, len(local_shap_df) - 1))

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[0]

    exp = shap.Explanation(
        values=local_shap_df.iloc[w_idx].values,
        base_values=float(expected_value),
        data=X_target.iloc[w_idx].values,
        feature_names=X_target.columns.tolist(),
    )

    wf_fig = plt.figure(figsize=(12, 8))
    shap.plots.waterfall(exp, show=False)
    wf_fig = plt.gcf()
    dash_variants = ['−', '‐', '‑', '‒', '–', '—', '﹣', '－']
    for txt in plt.gca().texts:
        cleaned = txt.get_text()
        for d in dash_variants:
            cleaned = cleaned.replace(d, '-')
        txt.set_text(cleaned)
        txt.set_fontfamily('DejaVu Sans')
    wf_fig.tight_layout()
    wf_fig.savefig(waterfall_png, facecolor='white', dpi=SHAP_FIG_DPI)
    plt.close(wf_fig)

    force_fig = plt.figure(figsize=(18, 3))
    shap.force_plot(
        float(expected_value),
        local_shap_df.iloc[w_idx].values,
        X_target.iloc[w_idx].values,
        feature_names=X_target.columns.tolist(),
        matplotlib=True,
        show=False,
    )
    force_fig.tight_layout()
    force_fig.savefig(force_png, facecolor='white', dpi=SHAP_FIG_DPI)
    plt.close(force_fig)

    print(f"[SHAP] 輸出完成: {local_csv}")
    print(f"[SHAP] 輸出完成: {global_csv}")
    print(f"[SHAP] 輸出完成: {waterfall_png}")
    print(f"[SHAP] 輸出完成: {force_png}")


# =============================================================================
# 美股股價載入 (一次抓全部 ticker; 後續用 pivot 取單檔)
# =============================================================================
_PRICE_CACHE: dict = {}


def get_stock_price(ticker: str) -> pd.DataFrame:
    """回傳單檔 OHLCV DataFrame (columns: Open/High/Low/Close/Volume)。"""
    if ticker in _PRICE_CACHE:
        return _PRICE_CACHE[ticker]
    frames = load_price_frames([ticker])
    if ticker not in frames['Close'].columns:
        raise RuntimeError(f"{ticker}: 無法從 yfinance 取得股價")
    out = pd.DataFrame({
        'Open':   frames['Open'][ticker],
        'High':   frames['High'][ticker],
        'Low':    frames['Low'][ticker],
        'Close':  frames['Close'][ticker],
        'Volume': frames['Volume'][ticker],
    })
    out = out.dropna(how='all')
    out.index = pd.to_datetime(out.index)
    out.index.name = 'Date'
    _PRICE_CACHE[ticker] = out
    return out


# =============================================================================
# 回測 (美股慣例：per-share, 零佣金, INITIAL_CAPITAL USD)
# =============================================================================
def plot_backtest(stock_id, stock_name, y_pred, stock_price,
                  long, short, short_to_long, long_to_short, threshold,
                  clf, period, cumSum, prob):
    AGG_DES1 = (y_pred > threshold).astype(int)
    df = pd.DataFrame()

    # --- 產生買入信號 ---
    if AGG_DES1.iloc[0] == 0:
        sig_buy = []
        for i in range(len(AGG_DES1)):
            pat  = [0] * short_to_long + [1] * long
            pat1 = [0, 1] + [1] * long
            pat2 = [1, 0] + [1] * long
            if (i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat) or
                i >= (len(pat1)-1) and np.array_equal(AGG_DES1[i-(len(pat1)-1):i+1].values, pat1) or
                i >= (len(pat2)-1) and np.array_equal(AGG_DES1[i-(len(pat2)-1):i+1].values, pat2)):
                sig_buy.append(1)
            else:
                sig_buy.append(0)
    else:
        sig_buy = [1]
        for i in range(1, len(AGG_DES1)):
            pat  = [0] * short_to_long + [1] * long
            pat1 = [0, 1] + [1] * long
            pat2 = [1, 0] + [1] * long
            if (i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat) or
                i >= (len(pat1)-1) and np.array_equal(AGG_DES1[i-(len(pat1)-1):i+1].values, pat1) or
                i >= (len(pat2)-1) and np.array_equal(AGG_DES1[i-(len(pat2)-1):i+1].values, pat2)):
                sig_buy.append(1)
            else:
                sig_buy.append(0)

    # --- 產生賣出信號 ---
    sig_sell = []
    for i in range(len(AGG_DES1)):
        pat = [1] * long_to_short + [0] * short
        if i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat):
            sig_sell.append(-1)
        else:
            sig_sell.append(0)

    sig_buy  = pd.Series(sig_buy,  index=AGG_DES1.index)
    sig_sell = pd.Series(sig_sell, index=AGG_DES1.index)

    buy_action  = pd.Series(0, index=AGG_DES1.index)
    sell_action = pd.Series(0, index=AGG_DES1.index)
    cash   = pd.Series(0.0, index=AGG_DES1.index)
    cash.iloc[0] = INITIAL_CAPITAL
    shares = pd.Series(0.0, index=AGG_DES1.index)
    asset  = pd.Series(0.0, index=AGG_DES1.index)
    cost   = pd.Series(0.0, index=AGG_DES1.index)
    asset.iloc[0] = cash.iloc[0] + shares.iloc[0] * stock_price.iloc[0, 3] - cost.iloc[0]

    acc_buy = 0
    acc_sell = 0

    # 美股按股 (移除 *1000)，零佣金 (BUY_FEE = SELL_FEE = 0)
    if prob == 0:  # 啟用 CUSUM 方向 filter
        for i in range(1, len(sig_buy)):
            raw_v = cumSum.iloc[i]
            cusum_v = float(np.asarray(raw_v).flatten()[0]) if hasattr(raw_v, 'values') else float(raw_v)

            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0 and cusum_v > 0:
                shares.iloc[i] = cash.iloc[i-1] // stock_price.iloc[i, 0]
                cost.iloc[i] = shares.iloc[i] * stock_price.iloc[i, 0] * BUY_FEE
                cash.iloc[i] = cash.iloc[i-1] - cost.iloc[i] - shares.iloc[i] * stock_price.iloc[i, 0]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_buy += 1
                buy_action.iloc[i] = 1
                continue
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0 and cusum_v < 0:
                cost.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] * SELL_FEE
                cash.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] - cost.iloc[i] + cash.iloc[i-1]
                shares.iloc[i] = 0
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_sell += 1
                sell_action.iloc[i] = 1
                continue
            else:
                cash.iloc[i] = cash.iloc[i-1]
                shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
    else:  # 純信號模式
        for i in range(1, len(sig_buy)):
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0:
                shares.iloc[i] = cash.iloc[i-1] // stock_price.iloc[i, 0]
                cost.iloc[i] = shares.iloc[i] * stock_price.iloc[i, 0] * BUY_FEE
                cash.iloc[i] = cash.iloc[i-1] - cost.iloc[i] - shares.iloc[i] * stock_price.iloc[i, 0]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_buy += 1
                buy_action.iloc[i] = 1
                continue
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0:
                cost.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] * SELL_FEE
                cash.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] - cost.iloc[i] + cash.iloc[i-1]
                shares.iloc[i] = 0
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_sell += 1
                sell_action.iloc[i] = 1
                continue
            else:
                cash.iloc[i] = cash.iloc[i-1]
                shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]

    ret = asset / asset.shift(1) - 1
    ret_stock = stock_price['Close'] / stock_price['Close'].shift(1) - 1
    cumAsset = np.cumprod(1 + ret) - 1
    cumStock = np.cumprod(1 + ret_stock) - 1

    buy_action  = buy_action.where(buy_action == 1).dropna()
    sell_action = sell_action.where(sell_action == 1).dropna()

    df['cash']        = round(cash, 2)
    df['shares']      = round(shares, 2)
    df['cost']        = round(cost, 2)
    df['buy_action']  = buy_action
    df['sell_action'] = sell_action
    df['close']       = stock_price['Close']
    df['asset']       = round(asset, 2)

    from matplotlib.ticker import FormatStrFormatter
    with plt.style.context('seaborn-v0_8-whitegrid'):
        _apply_cjk_rcparams()
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(cumAsset * 100, label='Model Return', linewidth=2.3, color='#C44E52')
        ax.plot(cumStock * 100, label='Stock Return', linewidth=2.0, color='black', alpha=0.85)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
        ax.set_ylabel('Cumulative Return (%)', fontsize=13)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_title(
            f"{stock_id} {stock_name} ({clf})\nStock={cumStock.iloc[-1]*100:.2f}% vs Model={cumAsset.iloc[-1]*100:.2f}%",
            fontsize=15, fontweight='bold',
        )
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)

        buy_idx  = buy_action.index.intersection(cumAsset.index)
        sell_idx = sell_action.index.intersection(cumAsset.index)
        if len(buy_idx) > 0:
            ax.scatter(buy_idx, cumAsset.loc[buy_idx] * 100,
                       marker='^', s=95, color='#D62728', edgecolor='white',
                       linewidth=0.6, label='Buy', zorder=4)
        if len(sell_idx) > 0:
            ax.scatter(sell_idx, cumAsset.loc[sell_idx] * 100,
                       marker='v', s=95, color='#2E8B57', edgecolor='white',
                       linewidth=0.6, label='Sell', zorder=4)

        if len(period) > 1:
            for DES_update in period[1:]:
                try:
                    dt_u = pd.to_datetime(DES_update)
                    if cumAsset.index.min() <= dt_u <= cumAsset.index.max():
                        ax.axvline(dt_u, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
                except Exception:
                    pass

        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(),
                  loc='upper center', bbox_to_anchor=(0.5, -0.14),
                  ncol=4, fontsize=11, frameon=False,
                  columnspacing=1.2, handletextpad=0.5)
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    if save_fig:
        plt.savefig(EVAL_DIR / f"backtest_{stock_id}_L{long}S{short}.png", facecolor='white')
    if not show_fig:
        plt.close(fig)

    return (acc_buy, acc_sell, cumAsset, cumStock,
            sig_buy, sig_sell, buy_action, sell_action, df)


# =============================================================================
# findBestRF: RandomForest 超參數搜尋
# =============================================================================
def findBestRF(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)] + [None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_state = [int(x) for x in np.linspace(start=0, stop=5000, num=700)]
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'random_state': random_state,
    }
    unique, counts = np.unique(y_train, return_counts=True)
    counts = (1 / counts) * len(y_train)
    class_weights = dict(zip(unique, counts))

    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight=class_weights),
        param_distributions=random_grid,
        n_iter=30, cv=5, verbose=0, n_jobs=-1,
    )
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_


# =============================================================================
# update_DES: 主要訓練/載入流程
# =============================================================================
def update_DES(tickers, train_end, used_feats=None, force_retrain=False,
               return_explain_context=False):
    is_ablation = used_feats is not None
    feats = used_feats if is_ablation else sub_cats
    letters = compute_letters(feats) if is_ablation else None
    explain_context = None

    for stock_id in tickers:
        data_ticker = TICKER_DATA_ALIAS.get(stock_id, stock_id)
        # --- 讀取 4 個面向的 ATT 預測結果 ---
        X_all = pd.DataFrame()
        present_feats = []
        for cat in feats:
            df_all = []
            file_path = glob.glob(str(EXPERIMENT_ROOT / f"ATT_{cat}_{stock_id}" / "experiment_result_*.csv"))
            file = [y.replace('\\', '/') for y in file_path]
            try:
                for i in file:
                    df = pd.read_csv(i, index_col=0, parse_dates=True, header=None).squeeze("columns")
                    df = df[~df.index.duplicated(keep='last')]
                    df_all.append(df)
            except Exception:
                continue
            if not df_all:
                continue
            temp = pd.DataFrame(df_all).T.mean(axis=1)
            X_all = pd.concat([X_all, temp], axis=1)
            present_feats.append(cat)

        if X_all.shape[1] == 0:
            raise RuntimeError(f"{stock_id}: 無任何 ATT 預測檔可讀取 (experiment/ATT_*_{stock_id})")

        X_all.index.name = 'Date'
        X_all.columns = present_feats
        X_all.dropna(how='all', inplace=True)
        # 不同 facet 起始日可能不同 (e.g. moment 比 fundamental 早)，
        # concat(axis=1) 不會自動排序，需手動排序避免後續 .loc[start:end] 失敗。
        X_all = X_all.sort_index()
        X_all = X_all.ffill().bfill().fillna(0.5)
        X_all.index = pd.to_datetime(X_all.index)
        X_all = X_all.astype('float64')

        # 標籤：feature/fundamental_{ticker}.csv 的 y_20
        y_path = FEATURE_ROOT / f"fundamental_{stock_id}.csv"
        if not y_path.exists() and data_ticker != stock_id:
            alt_y_path = FEATURE_ROOT / f"fundamental_{data_ticker}.csv"
            if alt_y_path.exists():
                print(f"[INFO] {stock_id}: 使用特徵別名 {data_ticker} 讀取標籤")
                y_path = alt_y_path
        y_all = pd.read_csv(y_path, index_col=0, parse_dates=True)['y_20']
        y_all = y_all.reindex(X_all.index)

        # 股價 (yfinance + 快取)
        stock_price = get_stock_price(stock_id).loc['2021-12-31':]
        stock_price.index.name = 'Date'

        stock_name = DOW30_NAME.get(stock_id, stock_id)

        AGG_DES = pd.Series(dtype=float)
        AGG_RF  = pd.Series(dtype=float)
        X_train = X_all.loc[train_start:train_end]
        X_test  = X_all.loc[train_end:]
        y_train = y_all.loc[train_start:train_end]
        y_test  = y_all.loc[train_end:]

        last_model = None
        last_base_classifier = None
        last_paths = {}

        for update in period:
            if is_ablation:
                pred_des_path  = PRED_DIR_ABL      / f"DES_{stock_id}_{letters}.csv"
                pred_rf_path   = PRED_DIR_ABL      / f"RF_{stock_id}_{letters}.csv"
                des_model_path = DES_MODEL_DIR_ABL / f"DES_{stock_id}_{letters}.pkl"
                rf_model_path  = RF_MODEL_DIR_ABL  / f"RF_{stock_id}_{letters}.pkl"
            else:
                pred_des_path  = PRED_DES_DIR  / f"DES_pred_{stock_id}_{update}.csv"
                pred_rf_path   = PRED_RF_DIR   / f"RF_pred_{stock_id}_{update}.csv"
                des_model_path = DES_MODEL_DIR / f"DES_{stock_id}_{update}.pkl"
                rf_model_path  = RF_MODEL_DIR  / f"RF_{stock_id}_{update}.pkl"

            if (not force_retrain) and pred_des_path.exists() and pred_rf_path.exists():
                AGG_temp = pd.read_csv(pred_des_path, index_col=0, parse_dates=True).squeeze("columns")
                RF_temp  = pd.read_csv(pred_rf_path,  index_col=0, parse_dates=True).squeeze("columns")
                AGG_DES = pd.concat([AGG_DES, AGG_temp])
                AGG_RF  = pd.concat([AGG_RF,  RF_temp])
                if return_explain_context and des_model_path.exists() and rf_model_path.exists():
                    last_base_classifier = joblib.load(rf_model_path)
                    last_model = joblib.load(des_model_path)
                del AGG_temp, RF_temp
            else:
                if (not force_retrain) and des_model_path.exists() and rf_model_path.exists():
                    base_classifier = joblib.load(rf_model_path)
                    model = joblib.load(des_model_path)
                else:
                    base_classifier = findBestRF(X_train, y_train)
                    model = KNORAE(pool_classifiers=base_classifier, k=10, DFP=True)
                    model.fit(X_train, y_train)
                    joblib.dump(base_classifier, rf_model_path)
                    joblib.dump(model, des_model_path)
                testY_base = pd.Series(_predict_pos_proba(base_classifier, X_all), index=X_all.index)
                testY_base.to_csv(pred_rf_path, index=True)
                testY_ensemble = pd.Series(_predict_pos_proba(model, X_all), index=X_all.index)
                testY_ensemble.to_csv(pred_des_path, index=True, encoding='utf-8')
                AGG_DES = pd.concat([AGG_DES, testY_ensemble])
                AGG_RF  = pd.concat([AGG_RF,  testY_base])
                if return_explain_context:
                    last_base_classifier = base_classifier
                    last_model = model
                else:
                    del base_classifier, model
                del testY_base, testY_ensemble

            last_paths = {
                'pred_des_path': str(pred_des_path),
                'pred_rf_path':  str(pred_rf_path),
                'des_model_path': str(des_model_path),
                'rf_model_path':  str(rf_model_path),
                'period_key': update,
            }

        if return_explain_context:
            explain_context = {
                'model': last_model,
                'base_classifier': last_base_classifier,
                'X_train': X_train,
                'X_test':  X_test,
                'y_train': y_train,
                'y_test':  y_test,
                'paths':   last_paths,
            }

    if return_explain_context:
        return AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name, explain_context
    return AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name


# =============================================================================
# 繪圖：信號綜覽 (股價 + DES_output + 各面向)
# =============================================================================
# 可由環境變數覆寫，方便 batch / headless 執行：
#   SHOW_FIG=0  -> 不開互動視窗 (不呼叫 plt.show)
#   SAVE_FIG=0  -> 不寫出 PNG (backtest_*.png / ensemble_*.png)
show_fig = os.environ.get('SHOW_FIG', '1') != '0'
save_fig = os.environ.get('SAVE_FIG', '1') != '0'


def plot_performance(long, short, short_to_long, long_to_short, threshold,
                     AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum,
                     stock_id_display, stock_name, X_all, stock_price,
                     stock_id, use_cusum_filter=True):
    testX = X_all.reindex(stock_price.index)
    n_features = len(testX.columns.tolist())
    n_panels = 2 + n_features

    with plt.style.context('seaborn-v0_8-whitegrid'):
        _apply_cjk_rcparams()
        plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots(n_panels, 1, figsize=(24, max(12, 1.8 * n_panels)), sharex=True)
        if n_panels == 1:
            ax = [ax]

        x = np.arange(stock_price.shape[0])
        dt_index = pd.to_datetime(stock_price.index)
        quarter_positions = []
        quarter_labels = []
        prev_q = None
        for i, dt in enumerate(dt_index):
            q = (dt.year, dt.quarter)
            if q != prev_q:
                quarter_positions.append(i)
                quarter_labels.append(dt.strftime('%Y/%m/%d'))
                prev_q = q

        ax[0].plot(x, stock_price['Close'].values, color='black', linewidth=1.5)
        ax[0].set_title(f'Dynamic Ensemble Overview | {stock_id_display} {stock_name}',
                        fontsize=22, fontweight='bold')
        ax[0].set_ylabel('Price', fontsize=15, labelpad=10)

        color_long = '#C44E52'
        color_short = '#2E8B57'
        color_neutral = '#B0B0B0'

        ax[1].bar(x, AGG_DES_S.values,
                  color=[color_long if v > threshold else color_neutral if v == threshold else color_short
                         for v in AGG_DES_S.values],
                  width=0.85)
        ax[1].axhline(threshold, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
        ax[1].set_ylabel('DES_output', fontsize=15, labelpad=10)

        ax[0].yaxis.set_label_coords(-0.04, 0.5)
        ax[1].yaxis.set_label_coords(-0.04, 0.5)

        for i, col in enumerate(testX.columns.tolist()):
            vals = testX[col].values
            colors = ['#D9D9D9' if pd.isna(v) else (color_long if v > threshold else color_short)
                      for v in vals]
            ax[i + 2].bar(x, vals, color=colors, width=0.85)
            ax[i + 2].axhline(threshold, color='black', linestyle='--', linewidth=0.9, alpha=0.75)
            ax[i + 2].set_ylabel('')
            ax[i + 2].set_title(f"{col}", fontsize=19, loc='center', pad=6)

        for a in ax:
            a.set_xlim(0, len(x) - 1)
            a.margins(x=0)
            a.tick_params(axis='y', labelsize=14)
            a.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)

        for a in ax[:-1]:
            a.tick_params(axis='x', labelbottom=False)
        ax[-1].set_xticks(quarter_positions)
        ax[-1].set_xticklabels(quarter_labels, rotation=20, ha='right', fontsize=15)

    _prob_flag = 0 if use_cusum_filter else 1
    (acc_buy_DES_S, acc_sell_DES_S, cumAsset_DES_S, cumStock_DES_S,
     sig_buy_DES_S, sig_sell_DES_S, buy_action_DES_S, sell_action_DES_S, df_DES_S) = plot_backtest(
        stock_id_display, stock_name, AGG_DES_S, stock_price,
        long, short, short_to_long, long_to_short, threshold,
        AGG_DES_S.__class__.__name__, period, cumSum, _prob_flag,
    )

    buy_sig_DES_S  = [i for i, x in enumerate(stock_price.index) if x in [y for y in buy_action_DES_S.index]]
    sell_sig_DES_S = [j for j, w in enumerate(stock_price.index) if w in [z for z in sell_action_DES_S.index]]

    if len(buy_sig_DES_S) > 0:
        buy_close = stock_price['Close'].reindex(buy_action_DES_S.index).values
        ax[0].scatter(buy_sig_DES_S, buy_close, marker='^', s=150,
                      color='#D62728', edgecolor='white', linewidth=0.8, label='Buy')
    if len(sell_sig_DES_S) > 0:
        sell_close = stock_price['Close'].reindex(sell_action_DES_S.index).values
        ax[0].scatter(sell_sig_DES_S, sell_close, marker='v', s=150,
                      color=color_short, edgecolor='white', linewidth=0.8, label='Sell')
    if len(buy_sig_DES_S) > 0 or len(sell_sig_DES_S) > 0:
        ax[0].legend(loc='upper left', frameon=True, fontsize=18,
                     markerscale=1.4, borderpad=0.5, labelspacing=0.4)

    if len(buy_sig_DES_S) > 0:
        y_buy = AGG_DES_S.reindex(buy_action_DES_S.index).values
        ax[1].scatter(buy_sig_DES_S, y_buy, marker='^', s=90,
                      color='#D62728', edgecolor='white', linewidth=0.7)
    if len(sell_sig_DES_S) > 0:
        y_sell = AGG_DES_S.reindex(sell_action_DES_S.index).values
        ax[1].scatter(sell_sig_DES_S, y_sell, marker='v', s=90,
                      color=color_short, edgecolor='white', linewidth=0.7)

    for i, sig_b in enumerate(buy_sig_DES_S):
        ax[1].text(sig_b, 0.05, f'BUY_{i+1}', va='bottom', transform=ax[1].transData, ha='center',
                   bbox=dict(facecolor='white', edgecolor='#D62728', boxstyle='round,pad=0.2'),
                   fontdict=dict(fontsize=12, color='#D62728'))
    for j, sig_s in enumerate(sell_sig_DES_S):
        ax[1].text(sig_s, 0.35, f'SELL_{j+1}', va='bottom', transform=ax[1].transData, ha='center',
                   bbox=dict(facecolor='white', edgecolor=color_short, boxstyle='round,pad=0.2'),
                   fontdict=dict(fontsize=12, color=color_short))

    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.11, hspace=0.35)
    if save_fig:
        plt.savefig(ENSEMBLE_DIR / f"ensemble_{stock_id}.png", facecolor='white', dpi=300)
    if not show_fig:
        plt.close(fig)

    return (acc_buy_DES_S, acc_sell_DES_S, cumAsset_DES_S, cumStock_DES_S,
            sig_buy_DES_S, sig_sell_DES_S, buy_action_DES_S, sell_action_DES_S, df_DES_S)


def plot_academic_price_features(stock_id, stock_name, stock_price, X_all, threshold, used_feats=None):
    selected_cols = [f for f in FEATURE_ORDER if f in X_all.columns]
    if used_feats is not None:
        selected_cols = [f for f in selected_cols if f in used_feats]
    if len(selected_cols) == 0:
        print('[PLOT][WARN] 無可用面向可繪製，略過學術風格圖。')
        return None

    n_panels = 1 + len(selected_cols)
    fig_height = max(9, 2.1 * n_panels)

    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_height),
                                 sharex=True, constrained_layout=True)
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(
            f"{stock_id} {stock_name} | Price and Feature Signals (Academic Style)",
            fontsize=14, fontweight='bold', y=1.01,
        )
        _apply_cjk_rcparams()

        x_idx = stock_price.index
        axes[0].plot(x_idx, stock_price['Close'].values, color='black', linewidth=1.6)
        axes[0].set_ylabel('Close')
        axes[0].set_title('Stock Price', fontsize=11, loc='left')

        palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
        testX = X_all.reindex(stock_price.index)

        for i, col in enumerate(selected_cols, start=1):
            vals = testX[col].astype(float)
            color = palette[(i - 1) % len(palette)]
            axes[i].plot(x_idx, vals.values, color=color, linewidth=1.2)
            axes[i].axhline(threshold, color='gray', linestyle='--', linewidth=0.9, alpha=0.8)
            axes[i].set_ylim(0.0, 1.0)
            axes[i].set_ylabel(col)
            axes[i].set_title(f"Feature: {col}", fontsize=10, loc='left')

        axes[-1].set_xlabel('Date')
        for ax_ in axes:
            ax_.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    mode_suffix = 'full' if used_feats is None else 'drop_' + '_'.join([f for f in FEATURE_ORDER if f not in used_feats])
    out_png = EVAL_DIR / f"academic_price_features_{stock_id}_{mode_suffix}.png"
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
    if not show_fig:
        plt.close(fig)
    print(f"[PLOT] 學術風格圖輸出完成: {out_png}")
    return str(out_png)


def re_DES(x):
    return x


# =============================================================================
# 主程式
# =============================================================================
span = 1
long_d = 1
short_d = 1
short_to_long = 0
long_to_short = 0
threshold = 0.50
period = ['2019-12-31']
eval_start = '2021-12-31'
eval_end = None  # None → 動態取 DES 最後一天


def _parse_drop(drop_str):
    if drop_str is None or str(drop_str).strip() == '':
        return None
    drop = [x.strip() for x in str(drop_str).split(',') if x.strip()]
    invalid = [d for d in drop if d not in FEATURE_ORDER]
    if invalid:
        raise ValueError(f"未知特徵：{invalid}；可選：{FEATURE_ORDER}")
    if len(drop) >= len(FEATURE_ORDER):
        raise ValueError("不能 drop 全部特徵，至少要保留 1 個")
    used = [f for f in FEATURE_ORDER if f not in drop]
    return used


def main():
    while True:
        ticker = input("Please input US ticker (輸入 0 離開): ").strip().upper()
        if ticker == '0' or ticker == '':
            print("結束程式。")
            break

        print(f"可用特徵：{', '.join(FEATURE_ORDER)}")
        drop_in = input("要 drop 的特徵 (逗號分隔；直接 Enter = 全部使用): ").strip()
        try:
            used_feats = _parse_drop(drop_in)
        except Exception as e:
            print(f"[ERROR] drop 特徵輸入錯誤: {e}")
            continue

        cusum_in = input("啟用 CUSUM filter? (1=是 / 2=否) [1]: ").strip()
        use_cusum_filter = True if cusum_in == '' else cusum_in in ('1', 'y', 'Y', 'yes', 'true')

        retrain_in = input("強制重訓 DES/RF? (1=是 / 2=否) [2]: ").strip()
        force_retrain = False if retrain_in == '' else retrain_in in ('1', 'y', 'Y', 'yes', 'true')

        shap_in = input("啟用 SHAP explainer? (1=是 / 2=否) [2]: ").strip()
        use_shap = False if shap_in == '' else shap_in in ('1', 'y', 'Y', 'yes', 'true')

        force_shap_recompute = False
        shap_background_k = 50
        shap_chunk_size = 20
        shap_nsamples = 'auto'
        shap_waterfall_idx = -1

        if use_shap:
            shap_force_in = input("強制重算 SHAP? (1=是 / 2=否) [2]: ").strip()
            force_shap_recompute = False if shap_force_in == '' else shap_force_in in ('1', 'y', 'Y', 'yes', 'true')
            bg_in = input("SHAP 背景 kmeans 中心數 [50]: ").strip()
            try:
                shap_background_k = int(bg_in) if bg_in else 50
            except ValueError:
                shap_background_k = 50
            chunk_in = input("SHAP chunk size [20]: ").strip()
            try:
                shap_chunk_size = int(chunk_in) if chunk_in else 20
            except ValueError:
                shap_chunk_size = 20
            ns_in = input("SHAP nsamples [auto]: ").strip()
            if ns_in == '':
                shap_nsamples = 'auto'
            else:
                try:
                    shap_nsamples = int(ns_in)
                except ValueError:
                    shap_nsamples = 'auto'
            wf_in = input("waterfall 樣本索引 (-1 = 最後一筆) [-1]: ").strip()
            try:
                shap_waterfall_idx = int(wf_in) if wf_in else -1
            except ValueError:
                shap_waterfall_idx = -1

        thr_in = input(f"DES 信號門檻 (0.50~0.95) [{threshold}]: ").strip()
        try:
            _thr = float(thr_in) if thr_in else threshold
        except ValueError:
            _thr = threshold

        _used_display = used_feats if used_feats is not None else FEATURE_ORDER
        _drop_display = [f for f in FEATURE_ORDER if f not in _used_display]
        _mode = 'ABLATION' if used_feats is not None else 'FULL'
        print(f"[FEATURES] mode={_mode} | used={_used_display} | drop={_drop_display}")
        print(f"[CFG] {ticker}: CUSUM filter = {'ON' if use_cusum_filter else 'OFF'}, "
              f"DES threshold = {_thr}, force_retrain = {'ON' if force_retrain else 'OFF'}")
        print(f"[CFG][SHAP] enable={'ON' if use_shap else 'OFF'}, "
              f"force_recompute={'ON' if force_shap_recompute else 'OFF'}, "
              f"bg_k={shap_background_k}, chunk={shap_chunk_size}, "
              f"nsamples={shap_nsamples}, waterfall_idx={shap_waterfall_idx}")

        try:
            if use_shap:
                AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name, explain_context = update_DES(
                    [ticker], train_end, used_feats=used_feats,
                    force_retrain=force_retrain, return_explain_context=True,
                )
            else:
                AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name = update_DES(
                    [ticker], train_end, used_feats=used_feats,
                    force_retrain=force_retrain, return_explain_context=False,
                )
                explain_context = None

            stock_id_display = ticker

            des_last_day = pd.to_datetime(AGG_DES.index.max()).strftime('%Y-%m-%d')
            _eval_end = eval_end if eval_end else des_last_day
            print(f"[DATE] DES last day = {des_last_day}, eval_end = {_eval_end}")

            stock_price = stock_price.loc[eval_start:_eval_end].copy()
            if stock_price.empty:
                raise RuntimeError(f"{stock_id_display}: 指定期間 {eval_start}~{_eval_end} 無股價資料")

            AGG_DES = AGG_DES[~AGG_DES.index.duplicated(keep='last')]
            AGG_DES = AGG_DES.ewm(span=span, adjust=False).mean()
            AGG_DES = AGG_DES.reindex(stock_price.index).ffill().bfill()
            print(f"[DATE] AGG_DES range = {AGG_DES.index.min().date()} ~ {AGG_DES.index.max().date()}")

            AGG_RF = AGG_RF[~AGG_RF.index.duplicated(keep='last')]
            AGG_RF = AGG_RF.ewm(span=span, adjust=False).mean()
            AGG_RF = AGG_RF.reindex(stock_price.index).ffill().bfill()
            print(f"[DATE] stock_price range = {stock_price.index.min().date()} ~ {stock_price.index.max().date()}")

            # CUSUM 方向 filter (cumSum_prob_12/cusum_{ticker}.csv)
            cusum_sign_path = CUSUM_DIR_SIGN / f"cusum_{stock_id}.csv"
            cumSum = pd.read_csv(cusum_sign_path, index_col=0, parse_dates=True, header=None)
            cumSum.columns = ['cumSum']
            cumSum = cumSum[period[0]:]
            cumSum = cumSum.reindex(stock_price.index).bfill()
            cumSum = cumSum.loc[eval_start:_eval_end]
            print(f"[DATE] cumSum non-null max = {cumSum.dropna().index.max().date() if not cumSum.dropna().empty else 'None'}")

            # CUSUM 機率混合 (cumSum_prob_6/cusum_{ticker}.csv)
            cusum_prob_path = CUSUM_DIR_PROB / f"cusum_{stock_id}.csv"
            cumSum_prob = pd.read_csv(cusum_prob_path, index_col=0, parse_dates=True, header=None).squeeze("columns")
            cumSum_prob = cumSum_prob[period[0]:]
            cumSum_prob = cumSum_prob.reindex(stock_price.index).ffill()
            cumSum_prob = cumSum_prob.loc[eval_start:_eval_end]
            print(f"[DATE] cumSum_prob non-null max = {cumSum_prob.dropna().index.max().date() if not cumSum_prob.dropna().empty else 'None'}")

            # 混合 DES 機率與 CUSUM 機率 (0.6 / 0.4)
            AGG_DES_adj = AGG_DES * 0.6
            cumSum_prob_adj = cumSum_prob * 0.4
            AGG_DES_P = AGG_DES_adj.add(cumSum_prob_adj)
            AGG_DES_P = AGG_DES_P.bfill()

            data = {'AGG_DES': AGG_DES_P.values, 'cumSum': cumSum.values.flatten()}
            AGG_DES_temp = pd.DataFrame(data=data, index=cumSum.index)
            AGG_DES_S = AGG_DES_temp.apply(re_DES, axis=1)['AGG_DES']

            (acc_buy_DES, acc_sell_DES, cumAsset_DES, cumStock_DES,
             sig_buy_DES, sig_sell_DES, buy_action_DES, sell_action_DES, df_DES_S) = plot_performance(
                long_d, short_d, short_to_long, long_to_short, _thr,
                AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum,
                stock_id_display, stock_name, X_all, stock_price, stock_id,
                use_cusum_filter=use_cusum_filter,
            )

            df_buy  = df_DES_S[df_DES_S['buy_action']  == 1]['asset']
            df_sell = df_DES_S[df_DES_S['sell_action'] == 1]['asset']
            if len(df_buy) == len(df_sell):
                gain = df_sell.values - df_buy.values
            else:
                df_sell = pd.concat([df_sell, df_DES_S.iloc[[-1]]['asset']])
                gain = df_sell.values - df_buy.values

            profit = gain[gain > 0]
            loss   = gain[gain < 0]
            transaction = len(df_sell)
            win = len(profit)
            win_rate = (win / len(df_sell)) if transaction != 0 else np.inf
            avg_win  = profit.mean() if len(profit) > 0 else 0.0
            avg_loss = loss.mean()   if len(loss)   > 0 else 0.0
            win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf

            print('交易次數: ', transaction)
            print('獲利次數: ', win)
            print('勝率: {:14.2f}'.format(win_rate))
            print('總獲利: {:12.2f}'.format(np.sum(profit)))
            print('平均獲利: {:10.2f}'.format(avg_win))
            print('總損失:  {:11.2f}'.format(np.sum(loss)))
            print('平均損失: {:10.2f}'.format(avg_loss))
            print('盈虧比: {:12.2f}'.format(win_loss_ratio))
            W_L = pd.DataFrame({
                '日期': df_DES_S.index[-1],
                '股票代號': [stock_id_display],
                '股票名稱': [stock_name],
                '交易次數': [transaction],
                '獲利次數': [win],
                '總獲利':   [np.sum(profit)],
                '平均獲利': [avg_win],
                '總損失':   [np.sum(loss)],
                '平均損失': [avg_loss],
                '盈虧比':   [win_loss_ratio],
            })
            W_L.set_index('日期', inplace=True)

            if use_shap:
                if explain_context is None or explain_context.get('model') is None:
                    print('[SHAP][WARN] 找不到可用 DES 模型，略過 SHAP 計算。')
                else:
                    shap_tag = build_shap_tag(stock_id, period[0], used_feats)
                    local_csv  = Path(SHAP_OUTPUT_DIR) / f"local_{shap_tag}.csv"
                    global_csv = Path(SHAP_OUTPUT_DIR) / f"global_{shap_tag}.csv"
                    x_test = explain_context.get('X_test')
                    if x_test is None or len(x_test) == 0:
                        print('[SHAP][WARN] 測試區間 X_test 為空，略過 SHAP 計算。')
                    else:
                        if (not force_shap_recompute) and local_csv.exists() and global_csv.exists():
                            local_shap_df = pd.read_csv(local_csv, index_col=0, parse_dates=True)
                            global_df = pd.read_csv(global_csv)
                            print(f"[SHAP] 使用既有 cache: {local_csv}")
                            save_shap_summary_plot(shap_tag, x_test, local_shap_df)
                        else:
                            print('[SHAP] 建立 KernelExplainer 並計算測試區間 SHAP...')
                            explainer = build_kernel_explainer(
                                explain_context['model'],
                                explain_context['X_train'],
                                background_k=shap_background_k,
                            )
                            local_shap_df = compute_shap_local(
                                explainer, x_test,
                                chunk_size=shap_chunk_size,
                                nsamples=shap_nsamples,
                            )
                            global_df = compute_shap_global(local_shap_df)
                            save_shap_artifacts(
                                tag=shap_tag, explainer=explainer,
                                X_target=x_test, local_shap_df=local_shap_df,
                                global_df=global_df, waterfall_row=shap_waterfall_idx,
                            )
                        print('[SHAP] Global importance Top 6:')
                        print(global_df.head(6).to_string(index=False))

            if show_fig:
                plt.show()
        except Exception as e:
            print(f"[ERROR] 處理 {ticker} 時發生錯誤: {e}")
        finally:
            plt.close('all')


if __name__ == "__main__":
    main()
