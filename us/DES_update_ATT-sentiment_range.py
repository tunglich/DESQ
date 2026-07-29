# =============================================================================
# DES_update_ATT-sentiment_range.py
# 在 DES_update_ATT-sentiment.py 基礎上新增「指定區間」、「指定 ticker」
# 以及「特徵 Ablation」功能（參考 SKILL_DES_ablation.md）。
#
# 預設區間：2024-01-01 ~ 2026-03-31
# Ablation 輸出資料夾：D:/model_pred_DES_ablation_range/DES_{id}_{letters}.csv
#
# 用法：
#   python "DES_update_ATT-sentiment_range.py"
#       → 互動式：輸入 ticker / drop 特徵 / 區間
#   python "DES_update_ATT-sentiment_range.py" --ticker 2330
#       → 跑 2330，全部特徵、預設區間
#   python "DES_update_ATT-sentiment_range.py" --ticker 2330 --drop sentiment
#       → 跑 2330，drop sentiment、預設區間
#   python "DES_update_ATT-sentiment_range.py" --ticker 2330 --start 2024-01-01 --end 2026-03-31 --drop sentiment,macro
#       → 跑 2330，drop sentiment+macro，限定回測區間
#   python "DES_update_ATT-sentiment_range.py" --ticker 2330 --force-retrain
#       → 忽略既有 pred/pkl cache，強制重訓 DES/RF 並覆寫輸出
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
        print("[DES][WARN] 找不到中文字型，圖表中文將顯示為方框。")
        picked = ['DejaVu Sans']
    plt.rcParams['font.sans-serif'] = picked + ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

_configure_cjk_font()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from deslib.des.knora_e import KNORAE
import joblib, warnings, os, sys, glob
warnings.filterwarnings("ignore")

# deslib 0.3.7 vs sklearn>=1.7 相容性
import sklearn.base
if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
    sklearn.base.BaseEstimator._validate_data = lambda self, *args, **kwargs: _sklearn_validate_data(self, *args, **kwargs)

if sys.platform.startswith('linux'):
    _D = '/mnt/d'
    _FINLAB_ROOT = '/mnt/c/Users/tungl/finlab'
else:
    _D = 'D:'
    _FINLAB_ROOT = r'C:/Users/tungl/finlab'

# 全域參數
train_start = '2007-08-01'
train_end = '2024-12-31'
test_start = '2025-01-01'
sub_cats = ['fundamental', 'tech_trend', 'moment', 'trade', 'macro', 'sentiment']
out_dir   = f'{_FINLAB_ROOT}/cumSum'
out_dir_P = f'{_FINLAB_ROOT}/cumSum_prob_6'

show_fig = True
save_fig = True

# 預設股價載入起點（與原版相同）
DEFAULT_PRICE_START = '2021-12-31'

# === Ablation 相關設定 (參考 SKILL_DES_ablation.md) ===
# letters 串接的固定特徵順序
FEATURE_ORDER = ['fundamental', 'trade', 'tech_trend', 'moment', 'sentiment', 'macro']
# 預設回測區間
DEFAULT_DATE_START = '2024-01-01'
DEFAULT_DATE_END   = '2026-03-31'
# Ablation 專屬輸出資料夾
PRED_DIR_ABL_RANGE = f'{_D}/model_pred_DES_ablation_range'
DES_MODEL_DIR_ABL  = f'{_D}/DES_model_ablation'
RF_MODEL_DIR_ABL   = f'{_D}/RF_model_ablation'
for _p in (PRED_DIR_ABL_RANGE, DES_MODEL_DIR_ABL, RF_MODEL_DIR_ABL):
    os.makedirs(_p, exist_ok=True)

# 信號參數（與原版一致）
span = 1
long_d = 1
short_d = 1
short_to_long = 0
long_to_short = 0
threshold = 0.50
period = ['2019-12-31']

# 由 CLI 設定（在 main 內覆寫）
CLI_DATE_START = None
CLI_DATE_END = None


def compute_letters(used_feats):
    """依 FEATURE_ORDER 串接每個使用中特徵的前 2 字母。
    範例：全選 -> 'futrtemosema'；drop sentiment -> 'futrtemoma'。
    """
    ordered = [f for f in FEATURE_ORDER if f in used_feats]
    return ''.join(f[:2] for f in ordered)


def price_data(filename):
    df = pd.read_csv(f'{_D}/CmoneyFactor/{filename}.csv', index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='last')]
    return df


def plot_backtest(stock_id, y_pred, stock_price, long, short, short_to_long, long_to_short,
                  threshold, clf, period, cumSum, prob):
    AGG_DES1 = (y_pred > threshold).astype(int)
    df = pd.DataFrame()

    if AGG_DES1.iloc[0] == 0:
        sig_buy = []
        for i in range(len(AGG_DES1)):
            pat = [0]*short_to_long + [1]*long
            pat1 = [0,1] + [1]*long
            pat2 = [1,0] + [1]*long
            if (i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat) or
                i >= (len(pat1)-1) and np.array_equal(AGG_DES1[i-(len(pat1)-1):i+1].values, pat1) or
                i >= (len(pat2)-1) and np.array_equal(AGG_DES1[i-(len(pat2)-1):i+1].values, pat2)):
                sig_buy.append(1)
            else:
                sig_buy.append(0)
    else:
        sig_buy = [1]
        for i in range(1, len(AGG_DES1)):
            pat = [0]*short_to_long + [1]*long
            pat1 = [0,1] + [1]*long
            pat2 = [1,0] + [1]*long
            if (i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat) or
                i >= (len(pat1)-1) and np.array_equal(AGG_DES1[i-(len(pat1)-1):i+1].values, pat1) or
                i >= (len(pat2)-1) and np.array_equal(AGG_DES1[i-(len(pat2)-1):i+1].values, pat2)):
                sig_buy.append(1)
            else:
                sig_buy.append(0)

    sig_sell = []
    for i in range(len(AGG_DES1)):
        pat = [1]*long_to_short + [0]*short
        if i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat):
            sig_sell.append(-1)
        else:
            sig_sell.append(0)

    sig_buy = pd.Series(sig_buy, index=AGG_DES1.index)
    sig_sell = pd.Series(sig_sell, index=AGG_DES1.index)
    buy_action = pd.Series(0, index=AGG_DES1.index)
    sell_action = pd.Series(0, index=AGG_DES1.index)
    cash = pd.Series(0.0, index=AGG_DES1.index); cash.iloc[0] = 50000000
    shares = pd.Series(0.0, index=AGG_DES1.index)
    asset = pd.Series(0.0, index=AGG_DES1.index)
    cost = pd.Series(0.0, index=AGG_DES1.index)
    asset.iloc[0] = cash.iloc[0] + shares.iloc[0]*stock_price.iloc[0,3] - cost.iloc[0]
    acc_buy = 0; acc_sell = 0

    if prob == 0:
        for i in range(1, len(sig_buy)):
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0 and cumSum.iloc[i].values > 0:
                shares.iloc[i] = (cash.iloc[i-1]) // (stock_price.iloc[i,0]*1000)
                cost.iloc[i] = shares.iloc[i]*stock_price.iloc[i,0]*0.001425*1000
                cash.iloc[i] = -(cash.iloc[i-1]-cost.iloc[i]-shares.iloc[i]*stock_price.iloc[i,0]*1000)
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000
                acc_buy += 1; buy_action.iloc[i] = 1; continue
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0 and cumSum.iloc[i].values < 0:
                cost.iloc[i] = (shares.iloc[i-1]*stock_price.iloc[i,0])*0.004425*1000
                shares.iloc[i] = 0
                cash.iloc[i] = shares.iloc[i-1]*stock_price.iloc[i,0]*1000 - cost.iloc[i] + cash.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000
                acc_sell += 1; sell_action.iloc[i] = 1; continue
            else:
                cash.iloc[i] = cash.iloc[i-1]; shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000
    else:
        for i in range(1, len(sig_buy)):
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0:
                shares.iloc[i] = (cash.iloc[i-1]) // (stock_price.iloc[i,0]*1000)
                cost.iloc[i] = shares.iloc[i]*stock_price.iloc[i,0]*0.001425*1000
                cash.iloc[i] = -(cash.iloc[i-1]-cost.iloc[i]-shares.iloc[i]*stock_price.iloc[i,0]*1000)
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000
                acc_buy += 1; buy_action.iloc[i] = 1; continue
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0:
                cost.iloc[i] = (shares.iloc[i-1]*stock_price.iloc[i,0])*0.004425*1000
                shares.iloc[i] = 0
                cash.iloc[i] = shares.iloc[i-1]*stock_price.iloc[i,0]*1000 - cost.iloc[i] + cash.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000
                acc_sell += 1; sell_action.iloc[i] = 1; continue
            else:
                cash.iloc[i] = cash.iloc[i-1]; shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000

    ret = asset / asset.shift(1) - 1
    ret_stock = stock_price['Close'] / stock_price['Close'].shift(1) - 1
    cumAsset = np.cumprod(1+ret) - 1
    cumStock = np.cumprod(1+ret_stock) - 1
    buy_action = buy_action.where(buy_action == 1).dropna()
    sell_action = sell_action.where(sell_action == 1).dropna()

    df['cash'] = round(cash,2); df['shares'] = round(shares,2); df['cost'] = round(cost,2)
    df['buy_action'] = buy_action; df['sell_action'] = sell_action
    df['close'] = stock_price['Close']; df['asset'] = round(asset,2)

    from matplotlib.ticker import FormatStrFormatter
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,1,1)
    plt.plot(cumAsset*100, label='Model Return', linewidth=3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f%%'))
    plt.plot(cumStock*100, label='Stock Return')
    plt.title(f"{stock_id} {stock_name} ({clf})\nStock={cumStock.iloc[-1]*100:.2f}% vs Model={cumAsset.iloc[-1]*100:.2f}%", fontsize=16)
    plt.legend(loc='upper left'); plt.tight_layout()

    for i, action_b in enumerate(buy_action.index):
        s = datetime.strftime(datetime.date(action_b), '%Y-%m-%d')
        ax.annotate(f'BUY_{i+1}\n{s}', xy=(action_b, cumAsset[s]*100),
                    xytext=(action_b, cumAsset[s]*100 - np.max(cumAsset)*10),
                    arrowprops=dict(color='red', arrowstyle="->"), color='red', weight="bold")
    for i, action_s in enumerate(sell_action.index):
        s = datetime.strftime(datetime.date(action_s), '%Y-%m-%d')
        ax.annotate(f'SELL_{i+1}\n{s}', xy=(action_s, cumAsset[s]*100),
                    xytext=(action_s, cumAsset[s]*100 + np.max(cumAsset)*8),
                    arrowprops=dict(color='darkgreen', arrowstyle="->"), color='darkgreen', weight="bold")
    bbox = dict(boxstyle='round', fc='0.8', pad=1)
    if len(period) > 1:
        for i, DES_update in enumerate(period[1:]):
            mask = cumAsset[cumAsset.index <= DES_update]
            if mask.empty: continue
            d = mask.index[-1].date(); s = datetime.strftime(d, '%Y-%m-%d')
            ax.annotate(f'UPDATE ENSEMBLE_{i+1}\n{DES_update}', xy=(d, cumAsset[s]*100),
                        xytext=(d, cumAsset[s]*100 + np.max(cumAsset)*8),
                        bbox=bbox, arrowprops=dict(color='black', arrowstyle="->"),
                        color='black', weight="bold")

    if save_fig:
        os.makedirs('evaluation', exist_ok=True)
        plt.savefig(f"evaluation/backtest_{stock_id}_L{long}S{short}.png", facecolor='white')
    if not show_fig:
        plt.close(fig)

    return acc_buy, acc_sell, cumAsset, cumStock, sig_buy, sig_sell, buy_action, sell_action, df


def findBestRF(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(200, 2000, num=10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)] + [None]
    random_state = [int(x) for x in np.linspace(0, 5000, num=700)]
    random_grid = {
        'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
        'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,4],
        'bootstrap': [True, False], 'random_state': random_state,
    }
    unique, counts = np.unique(y_train, return_counts=True)
    counts = (1/counts) * len(y_train)
    class_weights = dict(zip(unique, counts))
    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight=class_weights),
        param_distributions=random_grid, n_iter=30, cv=5, verbose=0, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_
0

def update_DES(tickers, train_end, price_start=DEFAULT_PRICE_START, price_end=None,
               used_feats=None, letters=None, force_retrain=False):
    """訓練/載入 DES 並輸出預測。

    used_feats / letters 任一為 None 時 -> 全特徵模式（與原版路徑相同）。
    兩者都提供時 -> Ablation 模式，預測寫到 PRED_DIR_ABL_RANGE，
    模型寫到 DES_MODEL_DIR_ABL / RF_MODEL_DIR_ABL。
    """
    is_ablation = (used_feats is not None) and (letters is not None)
    feats = used_feats if is_ablation else sub_cats

    for stock_id in tickers:
        X_all = pd.DataFrame()
        present_feats = []
        for cat in feats:
            df_all = []
            file_path = glob.glob(f"{_D}/experiments_df_test/ATT_{cat}_{stock_id}/experiment_result_*.csv")
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
            raise RuntimeError(f"{stock_id}: 無任何 ATT 預測檔可讀取")

        X_all.index.name = 'Date'
        X_all.columns = present_feats
        X_all.dropna(how='all', inplace=True)
        X_all = X_all.ffill().bfill().fillna(0.5)
        X_all.index = pd.to_datetime(X_all.index)
        X_all = X_all.astype('float64')

        y_all = pd.read_csv(f"{_D}/Feature_new/fundamental_{stock_id}.csv",
                            index_col=0, parse_dates=True)['y_20']
        y_all = y_all.reindex(X_all.index)

        # 股價載入：價格起點固定為 price_start（預設 2021-12-31，與原版一致），
        # 若指定 price_end，截到該日為止
        df_open  = price_data('Open').loc[price_start:, stock_id]
        df_close = price_data('Close').loc[price_start:, stock_id]
        df_volume= price_data('Volume').loc[price_start:, stock_id]
        df_high  = price_data('High').loc[price_start:, stock_id]
        df_low   = price_data('Low').loc[price_start:, stock_id]
        stock_price = pd.DataFrame({'Open':df_open, 'High':df_high, 'Low':df_low,
                                    'Close':df_close, 'Volume':df_volume})
        stock_price.index.name = 'Date'
        if price_end is not None:
            stock_price = stock_price.loc[:price_end]

        name_temp = pd.read_csv(f'{_D}/CmoneyFactor/Stock_name.csv')
        stock_name_v = name_temp[stock_id][0] if stock_id in name_temp.columns else stock_id

        AGG_DES = pd.Series(); AGG_RF = pd.Series()
        for update in period:
            if is_ablation:
                pred_des_path = f"{PRED_DIR_ABL_RANGE}/DES_{stock_id}_{letters}.csv"
                pred_rf_path  = f"{PRED_DIR_ABL_RANGE}/RF_{stock_id}_{letters}.csv"
                des_pkl       = f"{DES_MODEL_DIR_ABL}/DES_{stock_id}_{letters}.pkl"
                rf_pkl        = f"{RF_MODEL_DIR_ABL}/RF_{stock_id}_{letters}.pkl"
            else:
                pred_des_path = f"{_D}/model_pred_DES_test/DES_pred_{stock_id}_{update}.csv"
                pred_rf_path  = f"{_D}/model_pred_RF_test/RF_pred_{stock_id}_{update}.csv"
                des_pkl       = f"{_D}/DES_model_test/DES_{stock_id}_{update}.pkl"
                rf_pkl        = f"{_D}/RF_model_test/RF_{stock_id}_{update}.pkl"

            if (not force_retrain) and os.path.exists(pred_des_path) and os.path.exists(pred_rf_path):
                AGG_temp = pd.read_csv(pred_des_path, index_col=0, parse_dates=True).squeeze("columns")
                RF_temp  = pd.read_csv(pred_rf_path,  index_col=0, parse_dates=True).squeeze("columns")
                AGG_DES = pd.concat([AGG_DES, AGG_temp])
                AGG_RF  = pd.concat([AGG_RF,  RF_temp])
            else:
                X_train = X_all.loc[train_start:train_end]
                y_train = y_all.loc[train_start:train_end]
                if (not force_retrain) and os.path.exists(des_pkl) and os.path.exists(rf_pkl):
                    base_classifier = joblib.load(rf_pkl)
                    model = joblib.load(des_pkl)
                else:
                    base_classifier = findBestRF(X_train, y_train)
                    model = KNORAE(pool_classifiers=base_classifier, k=10, DFP=True)
                    model.fit(X_train, y_train)
                    joblib.dump(base_classifier, rf_pkl)
                    joblib.dump(model, des_pkl)
                testY_base = pd.Series(base_classifier.predict_proba(X_all)[:,1], index=X_all.index)
                testY_base.to_csv(pred_rf_path, index=True)
                testY_ensemble = pd.Series(model.predict_proba(X_all)[:,1], index=X_all.index)
                testY_ensemble.to_csv(pred_des_path, index=True, encoding='utf-8')
                AGG_DES = pd.concat([AGG_DES, testY_ensemble])
                AGG_RF  = pd.concat([AGG_RF,  testY_base])
    return AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name_v


def plot_performance(long, short, short_to_long, long_to_short, threshold,
                     AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum, stock_name_v,
                     use_cusum_filter=True):
    # 先跑回測，取得買/賣動作（要標到 ensemble 圖上）
    _prob_flag = 0 if use_cusum_filter else 1
    res = plot_backtest(stock_id, AGG_DES_S, stock_price, long, short, short_to_long, long_to_short,
                        threshold, AGG_DES_S.__class__.__name__, period, cumSum, _prob_flag)
    (acc_buy_DES_S, acc_sell_DES_S, cumAsset_DES_S, cumStock_DES_S, sig_buy_DES_S, sig_sell_DES_S,
     buy_action_DES_S, sell_action_DES_S, df_DES_S) = res

    fig, ax = plt.subplots(10, figsize=(25, 15))
    x = np.arange(stock_price.shape[0])
    ticks = [dt.strftime('%Y/%m/%d') for dt in stock_price.index]
    space = max(int(len(ticks)/10), 1)
    ticks = [tick_ if i % space == 0 or i == len(ticks)-1 else '' for i, tick_ in enumerate(ticks)]

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

    for i, col in enumerate(X_all.columns.tolist()):
        testX = X_all.reindex(stock_price.index)
        vals = testX[col].values
        colors = ['lightgray' if pd.isna(v) else ('red' if v > threshold else 'green') for v in vals]
        ax[i+4].bar(x, vals, color=colors)
        ax[i+4].set_xticks(x); ax[i+4].set_xticklabels(ticks); ax[i+4].set_ylabel(f"{col}")

    # 在 ax[0] 股價圖與 ax[3] DES_smooth 上標出買賣點：用垂直線 + 上方文字方塊
    buy_sig = [i for i, xd in enumerate(stock_price.index) if xd in buy_action_DES_S.index]
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
        os.makedirs(f"{_D}/model_output", exist_ok=True)
        fig.savefig(f"{_D}/model_output/ensemble_{stock_id}.png", facecolor='white')
    if not show_fig:
        plt.close(fig)
    return res


def re_DES(x):
    return x


def run_one(ticker, date_start=None, date_end=None, used_feats=None, letters=None,
            use_cusum_filter=True, des_threshold=None, force_retrain=False):
    """單檔執行回測。date_start/date_end 為 None 時即套用預設行為。
    used_feats / letters 提供時則走 Ablation 路徑。
    use_cusum_filter: True=使用 CUSUM 過濾 (預設), False=純信號交易。
    des_threshold: DES 信號門檻，None 時使用全域預設 threshold。
    """
    global stock_price, stock_id, X_all, stock_name  # plot_* 內部以全域取用
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
    # 若指定起點，截掉前段
    if date_start is not None:
        stock_price_v = stock_price_v.loc[date_start:]
    if stock_price_v.empty:
        raise RuntimeError(f"stock_price 在指定區間 {date_start}~{date_end} 為空")

    stock_price = stock_price_v
    stock_id = stock_id_v
    X_all = X_all_v
    stock_name = stock_name_v

    AGG_DES = AGG_DES[~AGG_DES.index.duplicated(keep='last')]
    AGG_DES = AGG_DES.ewm(span=span, adjust=False).mean().reindex(stock_price.index).ffill().bfill()

    AGG_RF = AGG_RF[~AGG_RF.index.duplicated(keep='last')]
    AGG_RF = AGG_RF.ewm(span=span, adjust=False).mean().reindex(stock_price.index).ffill().bfill()

    cumSum = pd.read_csv(f'{out_dir}/cusum_{stock_id}.csv', index_col=0, parse_dates=True)
    cumSum = cumSum.loc[period[0]:].reindex(stock_price.index).bfill()

    cumSum_prob = pd.read_csv(f'{out_dir_P}/cumsum_prob_{stock_id}.csv',
                              index_col=0, parse_dates=True).squeeze("columns")
    cumSum_prob = cumSum_prob.loc[period[0]:].reindex(stock_price.index).ffill()

    AGG_DES_P = (AGG_DES*0.6).add(cumSum_prob*0.4).bfill()
    AGG_DES_temp = pd.DataFrame({'AGG_DES': AGG_DES_P.values, 'cumSum': cumSum.values.flatten()},
                                index=cumSum.index)
    AGG_DES_S = AGG_DES_temp.apply(re_DES, axis=1)['AGG_DES']

    res = plot_performance(long_d, short_d, short_to_long, long_to_short, _thr,
                           AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum, stock_name,
                           use_cusum_filter=use_cusum_filter)
    df_DES_S = res[-1]

    df_buy = df_DES_S[df_DES_S['buy_action'] == 1]['asset']
    df_sell = df_DES_S[df_DES_S['sell_action'] == 1]['asset']
    if len(df_buy) == len(df_sell):
        gain = df_sell.values - df_buy.values
    else:
        df_sell = pd.concat([df_sell, df_DES_S.iloc[[-1]]['asset']])
        gain = df_sell.values - df_buy.values

    profit = gain[gain > 0]; loss = gain[gain < 0]
    transaction = len(df_sell); win = len(profit)
    win_rate = (win / transaction) if transaction != 0 else np.inf
    avg_win = profit.mean() if len(profit) else 0.0
    avg_loss = loss.mean() if len(loss) else 0.0
    win_loss_ratio = (avg_win / abs(avg_loss)) if (len(loss) and avg_loss != 0) else np.inf

    rng_str = f"{date_start or stock_price.index[0].date()} ~ {date_end or stock_price.index[-1].date()}"
    print(f"\n=== {stock_id} {stock_name}  區間 {rng_str}  CUSUM={'ON' if use_cusum_filter else 'OFF'}  DES_T={_thr} ===")
    print('交易次數: ', transaction)
    print('獲利次數: ', win)
    print('勝率: {:14.2f}'.format(win_rate))
    print('總獲利: {:12.2f}'.format(np.sum(profit)))
    print('平均獲利: {:10.2f}'.format(avg_win))
    print('總損失:  {:11.2f}'.format(np.sum(loss)))
    print('平均損失: {:10.2f}'.format(avg_loss))
    print('盈虧比: {:12.2f}'.format(win_loss_ratio))


def _parse_drop(drop_str):
    """解析 drop 字串 -> (used_feats, letters)。空字串 = 全選, 此時回傳 (None, None)
    讓 update_DES 走原版 (DES_update_ATT-sentiment.py) 的 cache 路徑, 避免重訓 DES。
    只有指定 drop 時才進 Ablation 模式 (寫到 *_ablation_range 路徑)。"""
    if drop_str is None or str(drop_str).strip() == '':
        # 全特徵：回傳 None，走原版 cache，不重訓
        return None, None
    drop = [x.strip() for x in str(drop_str).split(',') if x.strip()]
    invalid = [d for d in drop if d not in FEATURE_ORDER]
    if invalid:
        raise ValueError(f"未知特徵：{invalid}；可選：{FEATURE_ORDER}")
    if len(drop) >= 6:
        raise ValueError("不能 drop 全部 6 個特徵，至少要保留 1 個")
    used = [f for f in FEATURE_ORDER if f not in drop]
    return used, compute_letters(used)


def main():
    parser = argparse.ArgumentParser(description='DES 回測（可指定 ticker、區間與特徵 ablation）')
    parser.add_argument('--ticker', type=str, default=None, help='股票代號，未指定則進入互動模式')
    parser.add_argument('--start', type=str, default=DEFAULT_DATE_START,
                        help=f'回測起始日 YYYY-MM-DD（預設 {DEFAULT_DATE_START}）')
    parser.add_argument('--end', type=str, default=DEFAULT_DATE_END,
                        help=f'回測結束日 YYYY-MM-DD（預設 {DEFAULT_DATE_END}）')
    parser.add_argument('--drop', type=str, default='',
                        help='要 drop 的特徵，逗號分隔（例：sentiment,macro）；空白 = 全部使用')
    parser.add_argument('--cusum', type=str, default='on', choices=['on', 'off', '1', '2'],
                        help='是否啟用 CUSUM filter：on/1 = 是 (預設), off/2 = 否')
    parser.add_argument('--threshold', '--des-threshold', dest='threshold',
                        type=float, default=threshold,
                        help=f'DES 信號門檻 (0.50~0.95，預設 {threshold})')
    parser.add_argument('--force-retrain', action='store_true',
                        help='忽略既有 pred/pkl cache，強制重訓 DES/RF 並覆寫')
    parser.add_argument('--no-show', action='store_true', help='不彈出視窗')
    args = parser.parse_args()

    use_cusum_filter = args.cusum in ('on', '1')
    des_threshold = args.threshold

    global show_fig
    if args.no_show:
        show_fig = False

    if args.ticker:
        try:
            used_feats, letters = _parse_drop(args.drop)
            _used_display = used_feats if used_feats is not None else FEATURE_ORDER
            _mode = 'FULL (reuse original DES cache)' if used_feats is None else 'ABLATION'
            print(f"[FEATURES] mode={_mode} | used={_used_display} | drop={[f for f in FEATURE_ORDER if f not in _used_display]} | letters={letters}")
            print(f"[RANGE]    {args.start} ~ {args.end}")
            print(f"[CFG]      CUSUM filter = {'ON' if use_cusum_filter else 'OFF'}, DES threshold = {des_threshold}, force_retrain = {'ON' if args.force_retrain else 'OFF'}")
            run_one(args.ticker.strip(), args.start, args.end, used_feats, letters,
                    use_cusum_filter=use_cusum_filter, des_threshold=des_threshold,
                    force_retrain=args.force_retrain)
        except Exception as e:
            print(f"[ERROR] 處理 {args.ticker} 時發生錯誤: {e}")
        finally:
            if show_fig:
                plt.show()
            plt.close('all')
        return

    # 互動模式：逐次詢問 ticker / drop / 區間
    while True:
        ticker = input("Please input stock_id (輸入 0 離開): ").strip()
        if ticker == '0' or ticker == '':
            print("結束程式。"); break
        try:
            print("可用特徵：fundamental, trade, tech_trend, moment, sentiment, macro")
            drop_in = input("要 drop 的特徵 (逗號分隔；直接 Enter = 全部使用): ").strip()
            used_feats, letters = _parse_drop(drop_in)

            start_in = input(f"回測起始日 [預設 {args.start}]: ").strip() or args.start
            end_in   = input(f"回測結束日 [預設 {args.end}]: ").strip() or args.end

            # CUSUM filter 選項（預設 ON）
            cusum_in = input("啟用 CUSUM filter? (1=是 / 2=否) [1]: ").strip()
            if cusum_in == '':
                use_cusum_filter_i = True
            else:
                use_cusum_filter_i = cusum_in in ('1', 'y', 'Y', 'yes', 'true')

            # DES 信號門檻
            thr_in = input(f"DES 信號門檻 (0.50~0.95) [{des_threshold}]: ").strip()
            try:
                des_threshold_i = float(thr_in) if thr_in else des_threshold
            except ValueError:
                print(f"[WARN] 無法解析 '{thr_in}'，改用 {des_threshold}")
                des_threshold_i = des_threshold

            retrain_in = input("強制重訓 DES/RF? (1=是 / 2=否) [2]: ").strip()
            if retrain_in == '':
                force_retrain_i = False
            else:
                force_retrain_i = retrain_in in ('1', 'y', 'Y', 'yes', 'true')

            _used_display = used_feats if used_feats is not None else FEATURE_ORDER
            _mode = 'FULL (reuse original DES cache)' if used_feats is None else 'ABLATION'
            print(f"[FEATURES] mode={_mode} | used={_used_display} | drop={[f for f in FEATURE_ORDER if f not in _used_display]} | letters={letters}")
            print(f"[RANGE]    {start_in} ~ {end_in}")
            print(f"[CFG]      CUSUM filter = {'ON' if use_cusum_filter_i else 'OFF'}, DES threshold = {des_threshold_i}, force_retrain = {'ON' if force_retrain_i else 'OFF'}")
            run_one(ticker, start_in, end_in, used_feats, letters,
                    use_cusum_filter=use_cusum_filter_i, des_threshold=des_threshold_i,
                    force_retrain=force_retrain_i)
            if show_fig:
                plt.show()
        except Exception as e:
            print(f"[ERROR] 處理 {ticker} 時發生錯誤: {e}")
        finally:
            plt.close('all')


if __name__ == '__main__':
    main()
