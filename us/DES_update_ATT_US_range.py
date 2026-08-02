# =============================================================================
# DES_update_ATT_US_range.py
# US (Dow 30) DES backtest with configurable window, feature ablation, and CLI mode.
# Adapted from DES_update_ATT-sentiment_range.py to match the US paths / conventions
# used in DES_update_ATT_US.py (4 facets / per-share / zero fee / 1M USD).
#
# Default window: 2024-01-01 ~ 2026-03-31
#
# Cache-path rules:
#   FULL mode (no --drop)     -> reuse DES_update_ATT_US.py cache:
#       DES_model_US/        RF_model_US/
#       model_pred_DES_US/   model_pred_RF_US/
#       (no retraining; just re-runs the backtest over the requested window)
#   ABLATION mode (--drop set)-> writes to ablation paths:
#       DES_model_US_ablation/  RF_model_US_ablation/  (shares base model pkl)
#       model_pred_DES_US_ablation_range/  (range-specific predictions)
#
# Usage:
#   python "DES_update_ATT_US_range.py"
#       -> interactive: ticker / drop features / window
#   python "DES_update_ATT_US_range.py" --ticker AAPL
#       -> AAPL, all features, default window
#   python "DES_update_ATT_US_range.py" --ticker AAPL --drop macro
#       -> AAPL, drop macro, default window (ablation)
#   python "DES_update_ATT_US_range.py" --ticker AAPL --start 2024-01-01 --end 2026-03-31 --drop macro,moment
#       -> AAPL, drop macro+moment, custom window
#   python "DES_update_ATT_US_range.py" --ticker AAPL --force-retrain
#       -> ignore existing pred/pkl cache, force retrain DES/RF
# =============================================================================

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
    plt.rcParams['font.sans-serif'] = picked + ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


_configure_cjk_font()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from deslib.des.knora_e import KNORAE
import joblib
import warnings
import sys
import glob
from pathlib import Path
warnings.filterwarnings("ignore")

# deslib 0.3.7 vs sklearn>=1.7 compatibility
import sklearn.base
if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
    sklearn.base.BaseEstimator._validate_data = lambda self, *args, **kwargs: _sklearn_validate_data(self, *args, **kwargs)


# =============================================================================
# Workspace paths
# =============================================================================
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from feature._us_data import load_price_frames  # noqa: E402

EXPERIMENT_ROOT = _THIS_DIR / 'experiment'
FEATURE_ROOT    = _THIS_DIR / 'feature'

# FULL mode cache (shared with DES_update_ATT_US.py)
DES_MODEL_DIR = _THIS_DIR / 'DES_model_US'
PRED_DES_DIR  = _THIS_DIR / 'model_pred_DES_US'
RF_MODEL_DIR  = _THIS_DIR / 'RF_model_US'
PRED_RF_DIR   = _THIS_DIR / 'model_pred_RF_US'

# ABLATION paths
DES_MODEL_DIR_ABL  = _THIS_DIR / 'DES_model_US_ablation'              # model pkl shared with base
RF_MODEL_DIR_ABL   = _THIS_DIR / 'RF_model_US_ablation'
PRED_DIR_ABL_RANGE = _THIS_DIR / 'model_pred_DES_US_ablation_range'   # range-specific predictions

ENSEMBLE_DIR = _THIS_DIR / 'model_output_US'
EVAL_DIR     = _THIS_DIR / 'evaluation'

CUSUM_DIR_SIGN = _THIS_DIR / 'cumSum_prob_12'
CUSUM_DIR_PROB = _THIS_DIR / 'cumSum_prob_6'

for _p in (DES_MODEL_DIR, PRED_DES_DIR, RF_MODEL_DIR, PRED_RF_DIR,
           DES_MODEL_DIR_ABL, RF_MODEL_DIR_ABL, PRED_DIR_ABL_RANGE,
           ENSEMBLE_DIR, EVAL_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Global parameters
# =============================================================================
train_start = '2007-08-01'
train_end   = '2025-12-31'
test_start  = '2026-01-01'

# US 4 aspects (fixed order; drives ablation letters)
FEATURE_ORDER = ['fundamental', 'tech_trend', 'moment', 'macro']
sub_cats = FEATURE_ORDER.copy()

# US market convention
BUY_FEE         = 0.0
SELL_FEE        = 0.0
INITIAL_CAPITAL = 1_000_000.0  # USD

# Default price-load start and backtest window
DEFAULT_PRICE_START = '2021-12-31'
DEFAULT_DATE_START  = '2024-01-01'
DEFAULT_DATE_END    = '2026-03-31'

# Signal parameters
span = 1
long_d = 1
short_d = 1
short_to_long = 0
long_to_short = 0
threshold = 0.50
period = ['2019-12-31']

show_fig = True
save_fig = True

# Dow 30 company names (display only)
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


def compute_letters(used_feats):
    """Concatenate the first 2 letters of each active feature in FEATURE_ORDER order
    (e.g. all-selected -> 'futemoma'; drop macro -> 'futemo')."""
    ordered = [f for f in FEATURE_ORDER if f in used_feats]
    return ''.join(f[:2] for f in ordered)


# =============================================================================
# US price loader (yfinance + cache)
# =============================================================================
_PRICE_CACHE: dict = {}


def get_stock_price(ticker: str) -> pd.DataFrame:
    if ticker in _PRICE_CACHE:
        return _PRICE_CACHE[ticker]
    frames = load_price_frames([ticker])
    if ticker not in frames['Close'].columns:
        raise RuntimeError(f"{ticker}: failed to fetch prices from yfinance")
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
# Backtest (per-share, zero-fee, 1M USD)
# =============================================================================
def plot_backtest(stock_id, stock_name, y_pred, stock_price,
                  long, short, short_to_long, long_to_short,
                  threshold, clf, period, cumSum, prob):
    AGG_DES1 = (y_pred > threshold).astype(int)
    df = pd.DataFrame()

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
    cash   = pd.Series(0.0, index=AGG_DES1.index); cash.iloc[0] = INITIAL_CAPITAL
    shares = pd.Series(0.0, index=AGG_DES1.index)
    asset  = pd.Series(0.0, index=AGG_DES1.index)
    cost   = pd.Series(0.0, index=AGG_DES1.index)
    asset.iloc[0] = cash.iloc[0] + shares.iloc[0] * stock_price.iloc[0, 3] - cost.iloc[0]
    acc_buy = 0
    acc_sell = 0

    if prob == 0:
        for i in range(1, len(sig_buy)):
            raw_v = cumSum.iloc[i]
            cusum_v = float(np.asarray(raw_v).flatten()[0]) if hasattr(raw_v, 'values') else float(raw_v)
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0 and cusum_v > 0:
                shares.iloc[i] = cash.iloc[i-1] // stock_price.iloc[i, 0]
                cost.iloc[i] = shares.iloc[i] * stock_price.iloc[i, 0] * BUY_FEE
                cash.iloc[i] = cash.iloc[i-1] - cost.iloc[i] - shares.iloc[i] * stock_price.iloc[i, 0]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_buy += 1; buy_action.iloc[i] = 1; continue
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0 and cusum_v < 0:
                cost.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] * SELL_FEE
                cash.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] - cost.iloc[i] + cash.iloc[i-1]
                shares.iloc[i] = 0
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_sell += 1; sell_action.iloc[i] = 1; continue
            else:
                cash.iloc[i] = cash.iloc[i-1]
                shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
    else:
        for i in range(1, len(sig_buy)):
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0:
                shares.iloc[i] = cash.iloc[i-1] // stock_price.iloc[i, 0]
                cost.iloc[i] = shares.iloc[i] * stock_price.iloc[i, 0] * BUY_FEE
                cash.iloc[i] = cash.iloc[i-1] - cost.iloc[i] - shares.iloc[i] * stock_price.iloc[i, 0]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_buy += 1; buy_action.iloc[i] = 1; continue
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0:
                cost.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] * SELL_FEE
                cash.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i, 0] - cost.iloc[i] + cash.iloc[i-1]
                shares.iloc[i] = 0
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i, 3]
                acc_sell += 1; sell_action.iloc[i] = 1; continue
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
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(cumAsset * 100, label='Model Return', linewidth=3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f%%'))
    plt.plot(cumStock * 100, label='Stock Return')
    plt.title(
        f"{stock_id} {stock_name} ({clf})\nStock={cumStock.iloc[-1]*100:.2f}% vs Model={cumAsset.iloc[-1]*100:.2f}%",
        fontsize=16,
    )
    plt.legend(loc='upper left')
    plt.tight_layout()

    for i, action_b in enumerate(buy_action.index):
        s = datetime.strftime(datetime.date(action_b), '%Y-%m-%d')
        ax.annotate(f'BUY_{i+1}\n{s}', xy=(action_b, cumAsset[s] * 100),
                    xytext=(action_b, cumAsset[s] * 100 - np.max(cumAsset) * 10),
                    arrowprops=dict(color='red', arrowstyle="->"), color='red', weight="bold")
    for i, action_s in enumerate(sell_action.index):
        s = datetime.strftime(datetime.date(action_s), '%Y-%m-%d')
        ax.annotate(f'SELL_{i+1}\n{s}', xy=(action_s, cumAsset[s] * 100),
                    xytext=(action_s, cumAsset[s] * 100 + np.max(cumAsset) * 8),
                    arrowprops=dict(color='darkgreen', arrowstyle="->"), color='darkgreen', weight="bold")
    bbox = dict(boxstyle='round', fc='0.8', pad=1)
    if len(period) > 1:
        for i, DES_update in enumerate(period[1:]):
            mask = cumAsset[cumAsset.index <= DES_update]
            if mask.empty:
                continue
            d = mask.index[-1].date()
            s = datetime.strftime(d, '%Y-%m-%d')
            ax.annotate(f'UPDATE ENSEMBLE_{i+1}\n{DES_update}', xy=(d, cumAsset[s] * 100),
                        xytext=(d, cumAsset[s] * 100 + np.max(cumAsset) * 8),
                        bbox=bbox, arrowprops=dict(color='black', arrowstyle="->"),
                        color='black', weight="bold")

    if save_fig:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(EVAL_DIR / f"backtest_{stock_id}_L{long}S{short}.png", facecolor='white')
    if not show_fig:
        plt.close(fig)

    return (acc_buy, acc_sell, cumAsset, cumStock,
            sig_buy, sig_sell, buy_action, sell_action, df)


def findBestRF(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(200, 2000, num=10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)] + [None]
    random_state = [int(x) for x in np.linspace(0, 5000, num=700)]
    random_grid = {
        'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False], 'random_state': random_state,
    }
    unique, counts = np.unique(y_train, return_counts=True)
    counts = (1 / counts) * len(y_train)
    class_weights = dict(zip(unique, counts))
    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight=class_weights),
        param_distributions=random_grid, n_iter=30, cv=5, verbose=0, n_jobs=-1,
    )
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_


# =============================================================================
# update_DES: train / load
# =============================================================================
def update_DES(tickers, train_end, price_start=DEFAULT_PRICE_START, price_end=None,
               used_feats=None, letters=None, force_retrain=False):
    """Train / load DES and emit predictions.

    If either used_feats or letters is None    -> FULL mode (reuse DES_update_ATT_US.py cache).
    If both are provided                       -> ABLATION mode (writes to PRED_DIR_ABL_RANGE).
    """
    is_ablation = (used_feats is not None) and (letters is not None)
    feats = used_feats if is_ablation else sub_cats

    for stock_id in tickers:
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
            raise RuntimeError(f"{stock_id}: no ATT prediction files available (experiment/ATT_*_{stock_id})")

        X_all.index.name = 'Date'
        X_all.columns = present_feats
        X_all.dropna(how='all', inplace=True)
        X_all = X_all.ffill().bfill().fillna(0.5)
        X_all.index = pd.to_datetime(X_all.index)
        X_all = X_all.astype('float64')

        y_all = pd.read_csv(FEATURE_ROOT / f"fundamental_{stock_id}.csv",
                            index_col=0, parse_dates=True)['y_20']
        y_all = y_all.reindex(X_all.index)

        stock_price = get_stock_price(stock_id).loc[price_start:].copy()
        stock_price.index.name = 'Date'
        if price_end is not None:
            stock_price = stock_price.loc[:price_end]

        stock_name_v = DOW30_NAME.get(stock_id, stock_id)

        AGG_DES = pd.Series(dtype=float)
        AGG_RF  = pd.Series(dtype=float)
        for update in period:
            if is_ablation:
                pred_des_path = PRED_DIR_ABL_RANGE / f"DES_{stock_id}_{letters}.csv"
                pred_rf_path  = PRED_DIR_ABL_RANGE / f"RF_{stock_id}_{letters}.csv"
                des_pkl       = DES_MODEL_DIR_ABL  / f"DES_{stock_id}_{letters}.pkl"
                rf_pkl        = RF_MODEL_DIR_ABL   / f"RF_{stock_id}_{letters}.pkl"
            else:
                pred_des_path = PRED_DES_DIR  / f"DES_pred_{stock_id}_{update}.csv"
                pred_rf_path  = PRED_RF_DIR   / f"RF_pred_{stock_id}_{update}.csv"
                des_pkl       = DES_MODEL_DIR / f"DES_{stock_id}_{update}.pkl"
                rf_pkl        = RF_MODEL_DIR  / f"RF_{stock_id}_{update}.pkl"

            if (not force_retrain) and pred_des_path.exists() and pred_rf_path.exists():
                AGG_temp = pd.read_csv(pred_des_path, index_col=0, parse_dates=True).squeeze("columns")
                RF_temp  = pd.read_csv(pred_rf_path,  index_col=0, parse_dates=True).squeeze("columns")
                AGG_DES = pd.concat([AGG_DES, AGG_temp])
                AGG_RF  = pd.concat([AGG_RF,  RF_temp])
            else:
                X_train = X_all.loc[train_start:train_end]
                y_train = y_all.loc[train_start:train_end]
                if (not force_retrain) and des_pkl.exists() and rf_pkl.exists():
                    base_classifier = joblib.load(rf_pkl)
                    model = joblib.load(des_pkl)
                else:
                    base_classifier = findBestRF(X_train, y_train)
                    model = KNORAE(pool_classifiers=base_classifier, k=10, DFP=True)
                    model.fit(X_train, y_train)
                    joblib.dump(base_classifier, rf_pkl)
                    joblib.dump(model, des_pkl)
                testY_base = pd.Series(base_classifier.predict_proba(X_all)[:, 1], index=X_all.index)
                testY_base.to_csv(pred_rf_path, index=True)
                testY_ensemble = pd.Series(model.predict_proba(X_all)[:, 1], index=X_all.index)
                testY_ensemble.to_csv(pred_des_path, index=True, encoding='utf-8')
                AGG_DES = pd.concat([AGG_DES, testY_ensemble])
                AGG_RF  = pd.concat([AGG_RF,  testY_base])
    return AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name_v


# =============================================================================
# Multi-panel plot (price + DES_original/prob/smooth + each facet)
# =============================================================================
def plot_performance(long, short, short_to_long, long_to_short, threshold,
                     AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum, stock_name_v,
                     use_cusum_filter=True):
    _prob_flag = 0 if use_cusum_filter else 1
    res = plot_backtest(stock_id, stock_name_v, AGG_DES_S, stock_price,
                        long, short, short_to_long, long_to_short, threshold,
                        AGG_DES_S.__class__.__name__, period, cumSum, _prob_flag)
    (acc_buy_DES_S, acc_sell_DES_S, cumAsset_DES_S, cumStock_DES_S,
     sig_buy_DES_S, sig_sell_DES_S, buy_action_DES_S, sell_action_DES_S, df_DES_S) = res

    n_features = X_all.shape[1]
    n_panels = 4 + n_features
    fig, ax = plt.subplots(n_panels, figsize=(25, max(12, 1.5 * n_panels)))
    x = np.arange(stock_price.shape[0])
    ticks = [dt.strftime('%Y/%m/%d') for dt in stock_price.index]
    space = max(int(len(ticks) / 10), 1)
    ticks = [t if i % space == 0 or i == len(ticks) - 1 else '' for i, t in enumerate(ticks)]

    ax[0].plot(x, stock_price['Close'].values)
    ax[0].set_xticks(x); ax[0].set_xticklabels(ticks)
    ax[0].set_title(f'Dynamic_Ensemble_Section_{stock_id} {stock_name_v}', fontsize=16)

    ax[1].bar(x, AGG_DES.values,
              color=['red' if v > threshold else 'gray' if v == threshold else 'green' for v in AGG_DES.values])
    ax[1].set_xticks(x); ax[1].set_xticklabels(ticks); ax[1].set_ylabel('DES_original')

    ax[2].bar(x, AGG_DES_P.values,
              color=['red' if v > threshold else 'gray' if v == threshold else 'green' for v in AGG_DES_P.values])
    ax[2].set_xticks(x); ax[2].set_xticklabels(ticks); ax[2].set_ylabel('DES_prob')

    ax[3].bar(x, AGG_DES_S.values,
              color=['red' if v > threshold else 'gray' if v == threshold else 'green' for v in AGG_DES_S.values])
    ax[3].set_xticks(x); ax[3].set_xticklabels(ticks); ax[3].set_ylabel('DES_smooth')

    testX = X_all.reindex(stock_price.index)
    for i, col in enumerate(testX.columns.tolist()):
        vals = testX[col].values
        colors = ['lightgray' if pd.isna(v) else ('red' if v > threshold else 'green') for v in vals]
        ax[i + 4].bar(x, vals, color=colors)
        ax[i + 4].set_xticks(x); ax[i + 4].set_xticklabels(ticks); ax[i + 4].set_ylabel(f"{col}")

    # Mark buy/sell signals on ax[0] and ax[3]
    buy_sig  = [i for i, xd in enumerate(stock_price.index) if xd in buy_action_DES_S.index]
    sell_sig = [j for j, xd in enumerate(stock_price.index) if xd in sell_action_DES_S.index]
    ymin0, ymax0 = ax[0].get_ylim()
    for i, sig_b in enumerate(buy_sig):
        ax[0].axvline(sig_b, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax[3].axvline(sig_b, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax[0].text(sig_b, ymax0, f'B{i+1}', va='bottom', ha='center',
                   color='red', fontsize=10, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='red', boxstyle='round', pad=0.2))
        ax[3].text(sig_b, 1.02, f'B{i+1}', va='bottom', ha='center',
                   color='red', fontsize=10, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='red', boxstyle='round', pad=0.2))
    for j, sig_s in enumerate(sell_sig):
        ax[0].axvline(sig_s, color='darkgreen', linestyle='--', linewidth=0.8, alpha=0.6)
        ax[3].axvline(sig_s, color='darkgreen', linestyle='--', linewidth=0.8, alpha=0.6)
        ax[0].text(sig_s, ymin0, f'S{j+1}', va='top', ha='center',
                   color='darkgreen', fontsize=10, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='darkgreen', boxstyle='round', pad=0.2))
        ax[3].text(sig_s, 1.02, f'S{j+1}', va='bottom', ha='center',
                   color='darkgreen', fontsize=10, weight='bold',
                   bbox=dict(facecolor='white', edgecolor='darkgreen', boxstyle='round', pad=0.2))

    plt.tight_layout()
    if save_fig:
        ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(ENSEMBLE_DIR / f"ensemble_{stock_id}.png", facecolor='white')
    if not show_fig:
        plt.close(fig)
    return res


def re_DES(x):
    return x


# =============================================================================
# Single-ticker driver
# =============================================================================
def run_one(ticker, date_start=None, date_end=None, used_feats=None, letters=None,
            use_cusum_filter=True, des_threshold=None, force_retrain=False):
    global stock_price, stock_id, X_all, stock_name
    tickers = [ticker]
    _thr = des_threshold if des_threshold is not None else threshold

    AGG_DES, AGG_RF, stock_price_v, stock_id_v, X_all_v, stock_name_v = update_DES(
        tickers, train_end,
        price_start=DEFAULT_PRICE_START,
        price_end=date_end,
        used_feats=used_feats,
        letters=letters,
        force_retrain=force_retrain,
    )
    if date_start is not None:
        stock_price_v = stock_price_v.loc[date_start:]
    if stock_price_v.empty:
        raise RuntimeError(f"{ticker}: no prices for the requested window {date_start}~{date_end}")

    stock_price = stock_price_v
    stock_id = stock_id_v
    X_all = X_all_v
    stock_name = stock_name_v

    AGG_DES = AGG_DES[~AGG_DES.index.duplicated(keep='last')]
    AGG_DES = AGG_DES.ewm(span=span, adjust=False).mean().reindex(stock_price.index).ffill().bfill()

    AGG_RF = AGG_RF[~AGG_RF.index.duplicated(keep='last')]
    AGG_RF = AGG_RF.ewm(span=span, adjust=False).mean().reindex(stock_price.index).ffill().bfill()

    # CUSUM directional filter
    cumSum = pd.read_csv(CUSUM_DIR_SIGN / f"cusum_{stock_id}.csv",
                         index_col=0, parse_dates=True, header=None)
    cumSum.columns = ['cumSum']
    cumSum = cumSum.loc[period[0]:].reindex(stock_price.index).bfill()

    # CUSUM probability blend
    cumSum_prob = pd.read_csv(CUSUM_DIR_PROB / f"cusum_{stock_id}.csv",
                              index_col=0, parse_dates=True, header=None).squeeze("columns")
    cumSum_prob = cumSum_prob.loc[period[0]:].reindex(stock_price.index).ffill()

    AGG_DES_P = (AGG_DES * 0.6).add(cumSum_prob * 0.4).bfill()
    AGG_DES_temp = pd.DataFrame({'AGG_DES': AGG_DES_P.values, 'cumSum': cumSum.values.flatten()},
                                index=cumSum.index)
    AGG_DES_S = AGG_DES_temp.apply(re_DES, axis=1)['AGG_DES']

    res = plot_performance(long_d, short_d, short_to_long, long_to_short, _thr,
                           AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum, stock_name,
                           use_cusum_filter=use_cusum_filter)
    df_DES_S = res[-1]

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
    win_rate = (win / transaction) if transaction != 0 else np.inf
    avg_win  = profit.mean() if len(profit) else 0.0
    avg_loss = loss.mean()   if len(loss)   else 0.0
    win_loss_ratio = (avg_win / abs(avg_loss)) if (len(loss) and avg_loss != 0) else np.inf

    rng_str = f"{date_start or stock_price.index[0].date()} ~ {date_end or stock_price.index[-1].date()}"
    print(f"\n=== {stock_id} {stock_name}  window {rng_str}  CUSUM={'ON' if use_cusum_filter else 'OFF'}  DES_T={_thr} ===")
    print('trades:        ', transaction)
    print('winning trades:', win)
    print('win rate:      {:14.2f}'.format(win_rate))
    print('total profit:  {:12.2f}'.format(np.sum(profit)))
    print('avg win:       {:10.2f}'.format(avg_win))
    print('total loss:    {:11.2f}'.format(np.sum(loss)))
    print('avg loss:      {:10.2f}'.format(avg_loss))
    print('win/loss ratio:{:12.2f}'.format(win_loss_ratio))


def _parse_drop(drop_str):
    """Empty string -> FULL mode (None, None), reuses the base US cache.
    Non-empty drop -> ABLATION mode (used_feats, letters)."""
    if drop_str is None or str(drop_str).strip() == '':
        return None, None
    drop = [x.strip() for x in str(drop_str).split(',') if x.strip()]
    invalid = [d for d in drop if d not in FEATURE_ORDER]
    if invalid:
        raise ValueError(f"unknown feature: {invalid}; choose from {FEATURE_ORDER}")
    if len(drop) >= len(FEATURE_ORDER):
        raise ValueError(f"cannot drop all {len(FEATURE_ORDER)} features; at least one must be kept")
    used = [f for f in FEATURE_ORDER if f not in drop]
    return used, compute_letters(used)


def main():
    parser = argparse.ArgumentParser(description='US DES backtest (configurable ticker, window, and feature ablation)')
    parser.add_argument('--ticker', type=str, default=None, help='US ticker (e.g. AAPL); if omitted, enter interactive mode')
    parser.add_argument('--start', type=str, default=DEFAULT_DATE_START,
                        help=f'backtest start date YYYY-MM-DD (default {DEFAULT_DATE_START})')
    parser.add_argument('--end', type=str, default=DEFAULT_DATE_END,
                        help=f'backtest end date YYYY-MM-DD (default {DEFAULT_DATE_END})')
    parser.add_argument('--drop', type=str, default='',
                        help=f'features to drop, comma-separated; empty = use all. Choices: {",".join(FEATURE_ORDER)}')
    parser.add_argument('--cusum', type=str, default='on', choices=['on', 'off', '1', '2'],
                        help='enable CUSUM filter: on/1 = yes (default), off/2 = no')
    parser.add_argument('--threshold', '--des-threshold', dest='threshold',
                        type=float, default=threshold,
                        help=f'DES signal threshold (0.50~0.95, default {threshold})')
    parser.add_argument('--force-retrain', action='store_true',
                        help='ignore existing pred/pkl cache; force retrain DES/RF and overwrite')
    parser.add_argument('--no-show', action='store_true', help='do not open the plot window')
    args = parser.parse_args()

    use_cusum_filter = args.cusum in ('on', '1')
    des_threshold = args.threshold

    global show_fig
    if args.no_show:
        show_fig = False

    if args.ticker:
        ticker = args.ticker.strip().upper()
        try:
            used_feats, letters = _parse_drop(args.drop)
            _used_display = used_feats if used_feats is not None else FEATURE_ORDER
            _mode = 'FULL (reuse base US cache)' if used_feats is None else 'ABLATION'
            print(f"[FEATURES] mode={_mode} | used={_used_display} | drop={[f for f in FEATURE_ORDER if f not in _used_display]} | letters={letters}")
            print(f"[RANGE]    {args.start} ~ {args.end}")
            print(f"[CFG]      CUSUM filter = {'ON' if use_cusum_filter else 'OFF'}, DES threshold = {des_threshold}, force_retrain = {'ON' if args.force_retrain else 'OFF'}")
            run_one(ticker, args.start, args.end, used_feats, letters,
                    use_cusum_filter=use_cusum_filter, des_threshold=des_threshold,
                    force_retrain=args.force_retrain)
        except Exception as e:
            print(f"[ERROR] failure while processing {ticker}: {e}")
        finally:
            if show_fig:
                plt.show()
            plt.close('all')
        return

    # Interactive mode
    while True:
        ticker = input("Please input US ticker (0 to quit): ").strip().upper()
        if ticker == '0' or ticker == '':
            print("Exit.")
            break
        try:
            print(f"Available features: {', '.join(FEATURE_ORDER)}")
            drop_in = input("Features to drop (comma-separated; Enter = use all): ").strip()
            used_feats, letters = _parse_drop(drop_in)

            start_in = input(f"Start date [{args.start}]: ").strip() or args.start
            end_in   = input(f"End date [{args.end}]: ").strip() or args.end

            cusum_in = input("Enable CUSUM filter? (1=yes / 2=no) [1]: ").strip()
            use_cusum_filter_i = True if cusum_in == '' else cusum_in in ('1', 'y', 'Y', 'yes', 'true')

            thr_in = input(f"DES signal threshold (0.50~0.95) [{des_threshold}]: ").strip()
            try:
                des_threshold_i = float(thr_in) if thr_in else des_threshold
            except ValueError:
                print(f"[WARN] cannot parse '{thr_in}'; using {des_threshold}")
                des_threshold_i = des_threshold

            retrain_in = input("Force retrain DES/RF? (1=yes / 2=no) [2]: ").strip()
            force_retrain_i = False if retrain_in == '' else retrain_in in ('1', 'y', 'Y', 'yes', 'true')

            _used_display = used_feats if used_feats is not None else FEATURE_ORDER
            _mode = 'FULL (reuse base US cache)' if used_feats is None else 'ABLATION'
            print(f"[FEATURES] mode={_mode} | used={_used_display} | drop={[f for f in FEATURE_ORDER if f not in _used_display]} | letters={letters}")
            print(f"[RANGE]    {start_in} ~ {end_in}")
            print(f"[CFG]      CUSUM filter = {'ON' if use_cusum_filter_i else 'OFF'}, DES threshold = {des_threshold_i}, force_retrain = {'ON' if force_retrain_i else 'OFF'}")
            run_one(ticker, start_in, end_in, used_feats, letters,
                    use_cusum_filter=use_cusum_filter_i, des_threshold=des_threshold_i,
                    force_retrain=force_retrain_i)
            if show_fig:
                plt.show()
        except Exception as e:
            print(f"[ERROR] failure while processing {ticker}: {e}")
        finally:
            plt.close('all')


if __name__ == '__main__':
    main()
