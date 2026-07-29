# =============================================================================
# DES_update_ATT-sentiment.py
# 功能：使用 Dynamic Ensemble Selection (DES) 方法，結合多面向 ATT 模型預測結果
#       與 CUSUM 統計量，產生股票買賣信號並進行回測與績效繪圖。
#
# ⚠️ 本檔為「互動式全區間」版本，僅在需要首次訓練 DES 模型 / 建立 cache 時使用。
#    若只是要「看某檔在某段時間的回測結果」，請改用：
#        DES_update_ATT-sentiment_range.py  (預設 workflow，見 SKILL_DES_range.md)
#    它直接讀取本檔產生的 cache（D:/model_pred_DES_test/DES_pred_*.csv），
#    支援 --ticker / --start / --end / --cusum / --no-show，且不會重訓 DES。
#
# 流程概述：
#   1. 讀取各面向（fundamental, tech_trend, moment, trade, macro, sentiment）的
#      ATT 模型實驗預測結果，合併為特徵矩陣 X_all
#   2. 使用 RandomForest 作為基礎分類器，透過 RandomizedSearchCV 調參
#   3. 以 KNORAE (Dynamic Ensemble Selection) 作為集成模型進行預測
#   4. 結合 CUSUM 異常偵測統計量，過濾買賣時機
#   5. 模擬回測交易並繪製績效圖表
# =============================================================================

# --- 資料處理與數值計算 ---
import pandas as pd
import numpy as np
from datetime import datetime

# --- 繪圖設定 ---
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 互動式後端，可彈出視窗顯示圖形
import matplotlib.pyplot as plt

# --- 中文字型設定（模組載入時即套用，所有繪圖函式皆生效）---
def _configure_cjk_font():
    """依平台挑選可用 CJK 字型；找不到時印出提示而非靜默失敗。"""
    from matplotlib import font_manager
    candidates = [
        'Microsoft JhengHei',   # Windows 繁中
        'Microsoft YaHei',      # Windows 簡中
        'PingFang TC',          # macOS 繁中
        'Heiti TC',             # macOS
        'Noto Sans CJK TC',     # Linux 常見（fonts-noto-cjk）
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',     # TTC collection 在 matplotlib 常以 JP 名稱暴露
        'Noto Serif CJK TC',
        'Noto Serif CJK JP',
        'WenQuanYi Micro Hei',  # Linux fallback
        'WenQuanYi Zen Hei',
        'AR PL UMing TW',
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [name for name in candidates if name in installed]
    if not picked:
        # 最後再掃一次：任何名稱含 "CJK" 的字型都拿來用
        picked = sorted({f.name for f in font_manager.fontManager.ttflist if 'CJK' in f.name})
    if not picked:
        print("[DES][WARN] 找不到中文字型，圖表中文將顯示為方框。")
        print("           WSL/Ubuntu 可安裝：sudo apt install fonts-noto-cjk")
        print("           安裝後建議執行：fc-cache -f  並刪除 ~/.cache/matplotlib")
        picked = ['DejaVu Sans']
    return picked

_CJK_FONT_STACK = _configure_cjk_font()


def _apply_cjk_rcparams():
    """在 style context 內重設 CJK 相關字型，避免被 seaborn/serif 設定覆蓋。"""
    # 放最前面，後接 DejaVu Sans 當 ASCII fallback
    plt.rcParams['font.sans-serif'] = _CJK_FONT_STACK + ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


_apply_cjk_rcparams()

# --- 機器學習：RandomForest 基礎分類器與調參 ---
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# --- DESlib：Dynamic Ensemble Selection 動態集成選擇方法 ---
# 提供多種 DES 策略，本程式主要使用 KNORAE
from deslib.des.knora_u import KNORAU      # KNORA-Eliminate 的無權重版本
from deslib.des.knora_e import KNORAE      # KNORA-Eliminate：依鄰近區域選擇最佳分類器子集
from deslib.des.meta_des import METADES    # 以 meta-learning 選擇分類器
from deslib.des.des_clustering import DESClustering  # 基於聚類的 DES
from deslib.des.des_p import DESP          # 基於後驗機率的 DES
from deslib.des.des_knn import DESKNN      # 基於 KNN 的 DES
from deslib.des.knop import KNOP           # KNORA-Probability
from deslib.des.des_mi import DESMI        # 基於互信息的 DES

# --- 其他工具 ---
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import joblib      # 模型序列化存取（儲存/載入 .pkl）
import warnings
import os
import sys
import glob
from pathlib import Path
warnings.filterwarnings("ignore")

# =============================================================================
# deslib 0.3.7 相容性修補：scikit-learn >= 1.7 移除了 _validate_data 方法
# =============================================================================
import sklearn.base
if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
    sklearn.base.BaseEstimator._validate_data = lambda self, *args, **kwargs: _sklearn_validate_data(self, *args, **kwargs)

# =============================================================================
# 跨平台路徑前綴：WSL 下 D:/ 掛載於 /mnt/d/，Windows 下直接使用 D:/
# =============================================================================
if sys.platform.startswith('linux'):
    _D = '/mnt/d'
else:
    _D = 'D:'

# =============================================================================
# 全域參數設定
# =============================================================================

train_start = '2007-08-01'    # 訓練資料起始日期
train_end = '2025-12-31'      # 訓練資料結束日期
test_start = '2026-01-01'     # 測試資料起始日期
now = datetime.strftime(datetime.now(), '%Y-%m-%d')  # 當前日期字串

# ATT 模型的六大面向分類（每個面向各python 自有多次實驗結果）
FEATURE_ORDER = ['fundamental', 'trade', 'tech_trend', 'moment', 'sentiment', 'macro']
sub_cats = FEATURE_ORDER.copy()

# drop feature (ablation) 專用快取，避免和全特徵共用同一路徑
PRED_DIR_ABL = f'{_D}/model_pred_DES_ablation'
DES_MODEL_DIR_ABL = f'{_D}/DES_model_ablation'
RF_MODEL_DIR_ABL = f'{_D}/RF_model_ablation'
for _p in (PRED_DIR_ABL, DES_MODEL_DIR_ABL, RF_MODEL_DIR_ABL):
    os.makedirs(_p, exist_ok=True)

SHAP_OUTPUT_DIR = f'{_D}/model_output/shap'
SHAP_FIG_DPI = 300
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

def compute_letters(used_feats):
    """依固定順序將使用中特徵轉成短碼，作為 ablation cache key。"""
    ordered = [f for f in FEATURE_ORDER if f in used_feats]
    return ''.join(f[:2] for f in ordered)


def _safe_slug(value):
    """將參數字串轉成檔名友善格式。"""
    return str(value).replace(' ', '_').replace(':', '-').replace('/', '-').replace('\\', '-')


def build_shap_tag(stock_id, period_key, used_feats):
    """組合 SHAP 輸出 tag，確保 full/ablation 不互蓋。"""
    mode = 'full' if used_feats is None else f"abl_{compute_letters(used_feats)}"
    return f"{stock_id}_{_safe_slug(period_key)}_{mode}"


def _get_shap_module():
    try:
        import shap
        return shap
    except Exception as e:
        print(f"[SHAP][WARN] shap 無法載入，略過 explainability：{e}")
        return None


def build_kernel_explainer(des_model, X_train, background_k=50):
    """建立 KNORAE 專用 KernelExplainer（class-1 probability）。"""
    shap = _get_shap_module()
    if shap is None:
        return None
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train 為空，無法建立 SHAP 背景資料")

    bg_k = max(1, min(int(background_k), len(X_train)))
    # kmeans 可把背景資料壓縮成代表點，兼顧準確度與速度。
    background = shap.kmeans(np.asarray(X_train, dtype='float64'), bg_k)
    explainer = shap.KernelExplainer(
        lambda x: des_model.predict_proba(np.asarray(x, dtype='float64'))[:, 1],
        background
    )
    return explainer


def compute_shap_local(explainer, X_target, chunk_size=20, nsamples='auto'):
    """分批計算 local SHAP，回傳與 X_target 對齊的 DataFrame。"""
    if explainer is None:
        return None
    if X_target is None or len(X_target) == 0:
        raise ValueError("X_target 為空，無法計算 SHAP")

    total = len(X_target)
    local_blocks = []
    iterator = range(0, total, max(1, int(chunk_size)))
    for start in iterator:
        end = min(start + max(1, int(chunk_size)), total)
        batch = X_target.iloc[start:end]
        shap_vals = explainer.shap_values(np.asarray(batch, dtype='float64'), nsamples=nsamples)
        shap_arr = np.asarray(shap_vals, dtype='float64')
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)
        local_blocks.append(shap_arr)

    local_matrix = np.vstack(local_blocks)
    local_df = pd.DataFrame(local_matrix, index=X_target.index, columns=X_target.columns)
    return local_df


def compute_shap_global(local_shap_df):
    """以 mean(abs(SHAP)) 計算全域重要性。"""
    if local_shap_df is None or local_shap_df.empty:
        raise ValueError("local_shap_df 為空，無法計算 global importance")
    g = np.abs(local_shap_df).mean(axis=0).sort_values(ascending=False)
    global_df = pd.DataFrame({'feature': g.index, 'mean_abs_shap': g.values})
    return global_df


def save_shap_artifacts(tag, explainer, X_target, local_shap_df, global_df, waterfall_row=0):
    """儲存 local/global CSV 與 summary/beeswarm、waterfall、force 圖。"""
    shap = _get_shap_module()
    if shap is None:
        return

    base = Path(SHAP_OUTPUT_DIR)
    base.mkdir(parents=True, exist_ok=True)

    local_csv = base / f"local_{tag}.csv"
    global_csv = base / f"global_{tag}.csv"
    summary_png = base / f"summary_{tag}.png"
    waterfall_png = base / f"waterfall_{tag}.png"
    force_png = base / f"force_{tag}.png"

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
    # Some CJK fonts do not include U+2212; normalize labels to ASCII '-' to avoid tofu boxes.
    wf_ax = plt.gca()
    dash_variants = ['−', '‐', '‑', '‒', '–', '—', '﹣', '－']
    for txt in wf_ax.texts:
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
    print(f"[SHAP] 輸出完成: {summary_png}")
    print(f"[SHAP] 輸出完成: {waterfall_png}")
    print(f"[SHAP] 輸出完成: {force_png}")


def save_shap_summary_plot(tag, X_target, local_shap_df):
    """輸出純 SHAP beeswarm（dot）圖，不與既有回測圖共用畫布。"""
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
    shap.summary_plot(
        shap_aligned.values,
        x_aligned,
        plot_type='dot',
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(summary_png, facecolor='white', dpi=SHAP_FIG_DPI)
    plt.close(fig)
    print(f"[SHAP] 輸出完成: {summary_png}")

def price_data(filename):
    """讀取 CMoney 股價資料（Open/High/Low/Close/Volume）。
    
    Args:
        filename: 檔名（不含副檔名），如 'Open', 'Close' 等
    Returns:
        DataFrame，index 為日期，columns 為各股票代號
    """
    df = pd.read_csv(f'{_D}/CmoneyFactor/{filename}.csv', index_col = 0, parse_dates = True)
    df = df[~df.index.duplicated(keep='last')]  # 移除重複日期，保留最新資料
    return df

def plot_backtest(stock_id, stock_name, y_pred, stock_price, long, short, short_to_long, long_to_short, threshold, clf, period, cumSum, prob):
    """回測模擬函式：根據模型預測信號模擬交易，計算累積報酬並繪製回測圖。
    
    Args:
        stock_id:       股票代號
        y_pred:         模型預測的機率值 (Series)
        stock_price:    股價 DataFrame (Open/High/Low/Close/Volume)
        long:           買入信號需連續看多的天數
        short:          賣出信號需連續看空的天數
        short_to_long:  從空轉多需要的過渡天數
        long_to_short:  從多轉空需要的過渡天數
        threshold:      預測機率 > threshold 視為看多 (預設 0.5)
        clf:            分類器名稱（用於圖表標題）
        period:         模型更新的時間節點列表
        cumSum:         CUSUM 統計量（用於過濾買賣時機）
        prob:           0=使用 CUSUM 過濾, 其他=不使用 CUSUM 過濾
    Returns:
        acc_buy, acc_sell:      買入/賣出次數
        cumAsset, cumStock:     模型累積報酬 vs 股票累積報酬
        sig_buy, sig_sell:      買入/賣出信號序列
        buy_action, sell_action: 實際執行的買入/賣出動作
        df:                     交易明細 DataFrame
    """
    # 將預測機率轉為二元信號：> threshold 為 1 (看多), 否則為 0 (看空)
    AGG_DES1 = (y_pred > threshold).astype(int)
    df = pd.DataFrame()
    
    # --- 產生買入信號 ---
    # 買入條件：信號序列匹配特定模式（連續看多 long 天，或從空轉多的過渡模式）
    # pat:  標準模式 — 先經過 short_to_long 天看空，再連續 long 天看多
    # pat1: 快速反轉模式1 — [0,1] + 連續 long 天看多
    # pat2: 快速反轉模式2 — [1,0] + 連續 long 天看多
    if AGG_DES1.iloc[0] == 0:  # 若第一天為看空，從空倉開始
        sig_buy = []
        for i in range(len(AGG_DES1)):
            pat = [0] * short_to_long+ [1] * long
            pat1 = [0,1] + [1] * long
            pat2 = [1,0] + [1] * long
            if (i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat) or 
               i >= (len(pat1)-1) and np.array_equal(AGG_DES1[i-(len(pat1)-1):i+1].values, pat1) or
                i >= (len(pat2)-1) and np.array_equal(AGG_DES1[i-(len(pat2)-1):i+1].values, pat2)):
                sig_buy.append(1)
            else:
                sig_buy.append(0)
    else:  # 若第一天為看多，從滿倉開始
        sig_buy = [1]
        for i in range(1, len(AGG_DES1)):
            pat = [0] * short_to_long + [1] * long
            pat1 = [0,1] + [1] * long
            pat2 = [1,0] + [1] * long
            if (i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat) or
                i >= (len(pat1)-1) and np.array_equal(AGG_DES1[i-(len(pat1)-1):i+1].values, pat1) or
                i >= (len(pat2)-1) and np.array_equal(AGG_DES1[i-(len(pat2)-1):i+1].values, pat2)):
                sig_buy.append(1)
            else:
                sig_buy.append(0) 
        
    # --- 產生賣出信號 ---
    # 賣出條件：信號序列匹配特定看空模式
    # pat: 先連續 long_to_short 天看多，再連續 short 天看空
    sig_sell = []
    for i in range(len(AGG_DES1)):
        pat = [1] * long_to_short + [0] * short
        pat1 = [0,1] + [0] * short
        if i >= (len(pat)-1) and np.array_equal(AGG_DES1[i-(len(pat)-1):i+1].values, pat):
            sig_sell.append(-1)  # -1 表示賣出信號
        else:
            sig_sell.append(0)


    # --- 初始化交易模擬變數 ---
    sig_buy = pd.Series(sig_buy, index=AGG_DES1.index)    # 買入信號序列
    sig_sell = pd.Series(sig_sell, index=AGG_DES1.index)   # 賣出信號序列

    buy_action = pd.Series(0, index=AGG_DES1.index)   # 實際買入動作記錄
    sell_action = pd.Series(0, index=AGG_DES1.index)  # 實際賣出動作記錄
    
    cash = pd.Series(0.0, index=AGG_DES1.index)       # 現金部位
    cash.iloc[0] = 50000000                             # 初始資金 5000 萬元
    shares = pd.Series(0.0, index=AGG_DES1.index)      # 持股張數
    asset = pd.Series(0.0, index=AGG_DES1.index)       # 總資產
    cost = pd.Series(0.0, index=AGG_DES1.index)        # 交易成本（手續費+稅）
    asset.iloc[0] = cash.iloc[0] + shares.iloc[0] * stock_price.iloc[0,3] - cost.iloc[0]  # 初始總資產

    acc_buy = 0    # 累計買入次數
    acc_sell = 0   # 累計賣出次數
    
    # =================================================================
    # 交易模擬遏輯
    # prob == 0: 使用 CUSUM 過濾——僅在 CUSUM > 0 (趨勢向上) 時買入，
    #            CUSUM < 0 (趨勢向下) 時賣出
    # prob != 0: 純粹依據模型信號買賣，不使用 CUSUM 過濾
    #
    # 交易計算：
    #   買入手續費: 0.1425% (券商手續費)
    #   賣出手續費: 0.4425% (券商手續費 + 證交稅 0.3%)
    #   以開盤價成交，以收盤價計算帳面總資產
    # =================================================================
    if prob == 0:  # 使用 CUSUM 過濾的交易模式
    
        for i in range(1, len(sig_buy)):

            # 買入條件：前一日有買入信號 + 當前空倉 + CUSUM > 0 (趨勢向上)
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0 and cumSum.iloc[i].values > 0:  
                shares.iloc[i] = (cash.iloc[i-1]) // (stock_price.iloc[i,0]*1000)         # 以開盤價計算可買張數
                cost.iloc[i] = shares.iloc[i]*stock_price.iloc[i,0]*0.001425*1000         # 買入手續費 0.1425%
                cash.iloc[i] = -(cash.iloc[i-1]-cost.iloc[i]-shares.iloc[i]*stock_price.iloc[i,0]*1000)  # 更新現金
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000       # 以收盤價計算總資產

                acc_buy = acc_buy + 1
                buy_action.iloc[i] = 1
                continue

            # 賣出條件：前一日有賣出信號 + 當前有持股 + CUSUM < 0 (趨勢向下)
            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0 and cumSum.iloc[i].values < 0:
                cost.iloc[i] = (shares.iloc[i-1] * stock_price.iloc[i,0])*0.004425*1000  # 賣出手續費+稅 0.4425%
                shares.iloc[i] = 0  # 清倉
                cash.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i,0]*1000 - cost.iloc[i] + cash.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i,3]*1000
                acc_sell = acc_sell + 1
                sell_action.iloc[i]=1
                continue

            else:  # 無交易，維持現狀
                cash.iloc[i] = cash.iloc[i-1]
                shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] =  cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i,3]*1000
    else:  # 不使用 CUSUM 過濾的交易模式（純信號交易）
        
        for i in range(1, len(sig_buy)):
        
            if sig_buy.iloc[i-1] == 1 and shares.iloc[i-1] == 0:  # 有買入信號且空倉  
                shares.iloc[i] = (cash.iloc[i-1]) // (stock_price.iloc[i,0]*1000)
                cost.iloc[i] = shares.iloc[i]*stock_price.iloc[i,0]*0.001425*1000
                cash.iloc[i] = -(cash.iloc[i-1]-cost.iloc[i]-shares.iloc[i]*stock_price.iloc[i,0]*1000)
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i]*stock_price.iloc[i,3]*1000

                acc_buy = acc_buy + 1
                buy_action.iloc[i] = 1
                continue

            elif sig_sell.iloc[i-1] == -1 and shares.iloc[i-1] != 0:
                cost.iloc[i] = (shares.iloc[i-1] * stock_price.iloc[i,0])*0.004425*1000
                shares.iloc[i] = 0
                cash.iloc[i] = shares.iloc[i-1] * stock_price.iloc[i,0]*1000 - cost.iloc[i] + cash.iloc[i-1]
                asset.iloc[i] = cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i,3]*1000
                acc_sell = acc_sell + 1
                sell_action.iloc[i]=1
                continue

            else:
                cash.iloc[i] = cash.iloc[i-1]
                shares.iloc[i] = shares.iloc[i-1]
                asset.iloc[i] =  cash.iloc[i] + shares.iloc[i] * stock_price.iloc[i,3]*1000
    
    # --- 計算累積報酬 ---
    ret = asset / asset.shift(1) - 1                            # 模型每日報酬率
    ret_stock = stock_price['Close'] / stock_price['Close'].shift(1) - 1  # 股票每日報酬率
    cumAsset = np.cumprod(1+ret) - 1                            # 模型累積報酬
    cumStock = np.cumprod(1+ret_stock) - 1                      # 股票累積報酬（Buy & Hold）
    
    # 只保留實際執行的買賣動作
    buy_action = buy_action.where(buy_action == 1).dropna()
    sell_action = sell_action.where(sell_action == 1).dropna()
    
    # 建立交易明細 DataFrame
    df['cash'] = round(cash,2)
    df['shares'] = round(shares,2)
    df['cost'] = round(cost,2)
    df['buy_action'] = buy_action
    df['sell_action'] = sell_action
    df['close'] = stock_price['Close']
    df['asset'] = round(asset,2)

    # --- 繪製回測績效圖（學術論文風格）---
    from matplotlib.ticker import FormatStrFormatter

    with plt.style.context('seaborn-v0_8-whitegrid'):
        _apply_cjk_rcparams()
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(cumAsset * 100, label='Model Return', linewidth=2.3, color='#C44E52')
        ax.plot(cumStock * 100, label='Stock Return', linewidth=2.0, color='black', alpha=0.85)
        # y 軸維持百分比數值尺度，但不顯示 '%' 符號
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
        ax.set_ylabel('Cumulative Return (%)', fontsize=13)
        ax.set_xlabel('Date', fontsize=13)
        ax.set_title(
            f"{stock_id} {stock_name} ({clf})\nStock={cumStock.iloc[-1]*100:.2f}% vs Model={cumAsset.iloc[-1]*100:.2f}%",
            fontsize=15,
            fontweight='bold',
        )
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)

        # 以簡潔 marker 標示買賣點，避免大箭頭遮蔽線圖
        buy_idx = buy_action.index.intersection(cumAsset.index)
        sell_idx = sell_action.index.intersection(cumAsset.index)
        if len(buy_idx) > 0:
            ax.scatter(
                buy_idx,
                cumAsset.loc[buy_idx] * 100,
                marker='^',
                s=95,
                color='#D62728',
                edgecolor='white',
                linewidth=0.6,
                label='Buy',
                zorder=4,
            )
        if len(sell_idx) > 0:
            ax.scatter(
                sell_idx,
                cumAsset.loc[sell_idx] * 100,
                marker='v',
                s=95,
                color='#2E8B57',
                edgecolor='white',
                linewidth=0.6,
                label='Sell',
                zorder=4,
            )

        # 若有多次模型更新，使用淡灰虛線標示更新時點
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
        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc='upper center',
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
            fontsize=11,
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    
    if save_fig:  # 存檔回測圖
        plt.savefig(f"evaluation/backtest_{stock_id}_L{long}S{short}.png", facecolor='white')
    if not show_fig:  # 若不顯示則關閉圖形
        plt.close(fig)
        


    return acc_buy, acc_sell, cumAsset, cumStock, sig_buy, sig_sell, buy_action, sell_action, df


# =============================================================================
# findBestRF: 透過 RandomizedSearchCV 搜尋最佳 RandomForest 超參數
# 使用 class_weight 處理類別不平衡問題，n_iter=30 限制搜尋次數以避免過擬合
# =============================================================================
def findBestRF(X_train, y_train):
    """用 RandomizedSearchCV 搜尋最佳 RandomForest 參數。
    
    Args:
        X_train: 訓練特徵 DataFrame
        y_train: 訓練標籤 Series
    Returns:
        最佳參數的 RandomForestClassifier 實例
    """
    # 定義超參數搜尋空間
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]  # 樹的數量
    max_features = ['auto', 'sqrt', 'log2']  # 每棵樹的特徵選擇方式
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] + [None]  # 樹的最大深度
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_state = [int(x) for x in np.linspace(start=0, stop= 5000, num = 700)]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'random_state':random_state}   
    # 計算類別權重以處理不平衡資料集
    unique, counts = np.unique(y_train, return_counts=True)
    counts = (1 / counts)*len(y_train)  # 反比例權重
    class_weights = dict(zip(unique, counts))
    
    # RandomizedSearchCV: 30 次隨機搜尋，5-fold 交叉驗證
    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight = class_weights), 
        param_distributions=random_grid, 
        n_iter=30,             # 搜尋次數（過多容易過擬合）
        cv=5,                  # 5-fold 交叉驗證
        verbose=0,  
        n_jobs=-1)
    rf_random.fit(X_train, y_train)
    #rf_random.best_estimator_
    return rf_random.best_estimator_


# =============================================================================
# update_DES: 主要模型訓練/載入函式
# 對每檔股票：
#   1. 讀取 6 個面向的 ATT 模型實驗預測結果，合併為特徵矩陣 X_all
#   2. 讀取股價資料與標籤 (y_20: 20日後股價漲跌方向)
#   3. 對每個 period 時間節點：
#      - 若已有儲存的預測結果 CSV，直接讀取
#      - 若已有儲存的模型 PKL，載入模型
#      - 否則重新訓練 RF + KNORAE 並儲存
#   4. 輸出 DES 與 RF 的預測機率序列
# =============================================================================

def update_DES(tickers, train_end, used_feats=None, force_retrain=False, return_explain_context=False):
    # used_feats: list of features to keep, or None for all
    is_ablation = used_feats is not None
    feats = used_feats if is_ablation else sub_cats
    letters = compute_letters(feats) if is_ablation else None
    explain_context = None
    for stock_id in tickers:
        # --- Step 1: 讀取各面向 ATT 模型的實驗預測結果 ---
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
            except:
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

        y_all = pd.read_csv(f"{_D}/Feature_new/fundamental_{stock_id}.csv", index_col=0, parse_dates=True)['y_20']
        y_all = y_all.reindex(X_all.index)

        start_date = '2021-12-31'
        end_date = datetime.strftime(X_all.index[-1],'%Y-%m-%d')
        df_open = price_data('Open').loc[start_date:,stock_id]
        df_close = price_data('Close').loc[start_date:,stock_id]
        df_volume = price_data('Volume').loc[start_date:,stock_id]
        df_high = price_data('High').loc[start_date:,stock_id]
        df_low = price_data('Low').loc[start_date:,stock_id]
        stock_price = pd.DataFrame({'Open':df_open, 'High':df_high, 'Low':df_low, 'Close':df_close, 'Volume':df_volume})
        stock_price.index.name = 'Date'

        name_temp = pd.read_csv(f'{_D}/CmoneyFactor/Stock_name.csv')
        stock_name = name_temp[stock_id][0]

        AGG_DES = pd.Series()
        AGG_RF = pd.Series()
        X_train = X_all.loc[train_start:train_end]
        X_test = X_all.loc[train_end:]
        y_train = y_all.loc[train_start:train_end]
        y_test = y_all.loc[train_end:]

        last_model = None
        last_base_classifier = None
        last_paths = {}

        for update in period:
            if is_ablation:
                pred_des_path = f"{PRED_DIR_ABL}/DES_{stock_id}_{letters}.csv"
                pred_rf_path = f"{PRED_DIR_ABL}/RF_{stock_id}_{letters}.csv"
                des_model_path = f"{DES_MODEL_DIR_ABL}/DES_{stock_id}_{letters}.pkl"
                rf_model_path = f"{RF_MODEL_DIR_ABL}/RF_{stock_id}_{letters}.pkl"
            else:
                pred_des_path = f"{_D}/model_pred_DES_test/DES_pred_{stock_id}_{update}.csv"
                pred_rf_path = f"{_D}/model_pred_RF_test/RF_pred_{stock_id}_{update}.csv"
                des_model_path = f"{_D}/DES_model_test/DES_{stock_id}_{update}.pkl"
                rf_model_path = f"{_D}/RF_model_test/RF_{stock_id}_{update}.pkl"

            if (not force_retrain) and os.path.exists(pred_des_path) and os.path.exists(pred_rf_path):
                AGG_temp = pd.read_csv(pred_des_path, index_col = 0,parse_dates=True).squeeze("columns")
                RF_temp = pd.read_csv(pred_rf_path, index_col = 0, parse_dates=True).squeeze("columns")
                AGG_DES = pd.concat([AGG_DES, AGG_temp])
                AGG_RF = pd.concat([AGG_RF, RF_temp])
                if return_explain_context and os.path.exists(des_model_path) and os.path.exists(rf_model_path):
                    last_base_classifier = joblib.load(rf_model_path)
                    last_model = joblib.load(des_model_path)
                del AGG_temp, RF_temp
            else:
                if (not force_retrain) and os.path.exists(des_model_path) and os.path.exists(rf_model_path):
                    base_classifier = joblib.load(rf_model_path)
                    model = joblib.load(des_model_path)
                else:
                    base_classifier = findBestRF(X_train, y_train)
                    model = KNORAE(pool_classifiers=base_classifier, k=10, DFP=True)
                    model.fit(X_train, y_train)
                    joblib.dump(base_classifier, rf_model_path)
                    joblib.dump(model, des_model_path)
                testY_base = pd.Series(base_classifier.predict_proba(X_all)[:, 1], index=X_all.index)   
                testY_base.to_csv(pred_rf_path, index=True)
                testY_ensemble = pd.Series(model.predict_proba(X_all)[:, 1], index=X_all.index)
                testY_ensemble.to_csv(pred_des_path, index=True, encoding='utf-8')
                AGG_DES = pd.concat([AGG_DES, testY_ensemble])
                AGG_RF = pd.concat([AGG_RF, testY_base])
                if return_explain_context:
                    last_base_classifier = base_classifier
                    last_model = model
                else:
                    del base_classifier, model
                del testY_base, testY_ensemble

            last_paths = {
                'pred_des_path': pred_des_path,
                'pred_rf_path': pred_rf_path,
                'des_model_path': des_model_path,
                'rf_model_path': rf_model_path,
                'period_key': update,
            }

        if return_explain_context:
            explain_context = {
                'model': last_model,
                'base_classifier': last_base_classifier,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'paths': last_paths,
            }

    if return_explain_context:
        return AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name, explain_context
    return AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name

show_fig = True   # 是否彈出視窗顯示圖形
save_fig = True   # 是否儲存圖形到檔案

# =============================================================================
# plot_performance: 繪製信號綜覽圖（10 個子圖）
# 包含：股價走勢、DES 原始信號、DES+CUSUM 機率信號、DES 平滑信號、
#       以及 6 個面向的個別信號
# =============================================================================
def plot_performance(long, short, short_to_long, long_to_short, threshold, AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum, stock_id_display, stock_name, use_cusum_filter=True):
    """繪製信號綜覽圖，顯示各面向信號並標註買賣點。
    
    Args:
        long/short/short_to_long/long_to_short: 信號模式參數
        threshold:  買入門檻值
        AGG_DES_S:  經 CUSUM 過濾後的 DES 信號（用於回測）
        AGG_DES:    DES 原始信號
        AGG_DES_P:  DES + CUSUM_prob 混合信號
        period:     模型更新時間節點列表
        cumSum:     CUSUM 統計量
        stock_name: 股票名稱
        use_cusum_filter: 是否啟用 CUSUM 過濾 (prob=0 啟用 / prob=1 關閉)
    """
    testX = X_all.reindex(stock_price.index)
    n_features = len(testX.columns.tolist())
    # 依需求僅保留：股價 + DES_output + 各面向
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
        ax[0].set_title(f'Dynamic Ensemble Overview | {stock_id_display} {stock_name}', fontsize=22, fontweight='bold')
        ax[0].set_ylabel('Price', fontsize=15, labelpad=10)

        color_long = '#C44E52'
        color_short = '#2E8B57'
        color_neutral = '#B0B0B0'

        ax[1].bar(x, AGG_DES_S.values, color=[color_long if v > threshold else color_neutral if v == threshold else color_short for v in AGG_DES_S.values], width=0.85)
        ax[1].axhline(threshold, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
        ax[1].set_ylabel('DES_output', fontsize=15, labelpad=10)

        # 固定 Price / DES_output 的 y-label 座標，確保視覺上完全對齊
        ax[0].yaxis.set_label_coords(-0.04, 0.5)
        ax[1].yaxis.set_label_coords(-0.04, 0.5)

        for i, col in enumerate(testX.columns.tolist()):
            vals = testX[col].values
            colors = [
                '#D9D9D9' if pd.isna(v) else (color_long if v > threshold else color_short)
                for v in vals
            ]
            ax[i + 2].bar(x, vals, color=colors, width=0.85)
            ax[i + 2].axhline(threshold, color='black', linestyle='--', linewidth=0.9, alpha=0.75)
            ax[i + 2].set_ylabel('')
            ax[i + 2].set_title(f"{col}", fontsize=19, loc='center', pad=6)

        for a in ax:
            a.set_xlim(0, len(x) - 1)
            a.margins(x=0)
            a.tick_params(axis='y', labelsize=14)
            a.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)

        # 只在最下方顯示日期，避免互相遮蓋；右側留白避免最後日期被裁切
        for a in ax[:-1]:
            a.tick_params(axis='x', labelbottom=False)
        ax[-1].set_xticks(quarter_positions)
        ax[-1].set_xticklabels(quarter_labels, rotation=20, ha='right', fontsize=15)
    

    '''acc_buy_DES, acc_sell_DES, cumAsset_DES, cumStock_DES, sig_buy_DES, sig_sell_DES, buy_action_DES, sell_action_DES, df_DES\
    = plot_backtest(stock_id, AGG_DES, stock_price, long,short, short_to_long, long_to_short, threshold, AGG_DES_P.__class__.__name__, period, cumSum, 1)'''     # Sell signal set to 10 in most cases
    # 執行回測：prob=0 啟用 CUSUM 過濾；prob=1 關閉過濾
    _prob_flag = 0 if use_cusum_filter else 1
    acc_buy_DES_S, acc_sell_DES_S, cumAsset_DES_S, cumStock_DES_S, sig_buy_DES_S, sig_sell_DES_S, buy_action_DES_S, sell_action_DES_S, df_DES_S\
    = plot_backtest(stock_id_display, stock_name, AGG_DES_S, stock_price, long, short, short_to_long, long_to_short, threshold, AGG_DES_S.__class__.__name__, period, cumSum, _prob_flag)
    
    # 在 DES_output 圖標註買賣點
    

    '''buy_sig = [i for i, x in enumerate(stock_price.index) if x in [y for y in buy_action_DES.index]]
    sell_sig = [j for j, w in enumerate(stock_price.index) if w in [z for z in sell_action_DES.index]]

    for i, sig_b in enumerate(buy_sig):
        ax[1].text(sig_b, 0.05, f'BUY_{i+1}', va='bottom', transform=ax[1].transData,ha = 'center',\
                   bbox=dict(facecolor='white',edgecolor = 'red', boxstyle = 'round'), fontdict=dict(fontsize = 14, color = 'red'), )
    for j, sig_s in enumerate(sell_sig):
        ax[1].text(sig_s, 0.35, f'SELL_{j+1}', va='bottom', transform=ax[1].transData,ha = 'center',\
                   bbox=dict(facecolor='white',edgecolor = 'darkgreen', boxstyle = 'round'), fontdict=dict(fontsize = 14, color = 'darkgreen') )'''
        
    buy_sig_DES_S = [i for i, x in enumerate(stock_price.index) if x in [y for y in buy_action_DES_S.index]]
    sell_sig_DES_S = [j for j, w in enumerate(stock_price.index) if w in [z for z in sell_action_DES_S.index]]

    # 在股價圖與 DES_output 圖標示買賣點
    if len(buy_sig_DES_S) > 0:
        buy_close = stock_price['Close'].reindex(buy_action_DES_S.index).values
        ax[0].scatter(buy_sig_DES_S, buy_close, marker='^', s=150, color='#D62728', edgecolor='white', linewidth=0.8, label='Buy')
    if len(sell_sig_DES_S) > 0:
        sell_close = stock_price['Close'].reindex(sell_action_DES_S.index).values
        ax[0].scatter(sell_sig_DES_S, sell_close, marker='v', s=150, color=color_short, edgecolor='white', linewidth=0.8, label='Sell')
    if len(buy_sig_DES_S) > 0 or len(sell_sig_DES_S) > 0:
        ax[0].legend(loc='upper left', frameon=True, fontsize=18, markerscale=1.4, borderpad=0.5, labelspacing=0.4)

    if len(buy_sig_DES_S) > 0:
        y_buy = AGG_DES_S.reindex(buy_action_DES_S.index).values
        ax[1].scatter(buy_sig_DES_S, y_buy, marker='^', s=90, color='#D62728', edgecolor='white', linewidth=0.7)
    if len(sell_sig_DES_S) > 0:
        y_sell = AGG_DES_S.reindex(sell_action_DES_S.index).values
        ax[1].scatter(sell_sig_DES_S, y_sell, marker='v', s=90, color=color_short, edgecolor='white', linewidth=0.7)
    
    for i, sig_b in enumerate(buy_sig_DES_S):
        ax[1].text(sig_b, 0.05, f'BUY_{i+1}', va='bottom', transform=ax[1].transData, ha='center',\
                   bbox=dict(facecolor='white', edgecolor='#D62728', boxstyle='round,pad=0.2'), fontdict=dict(fontsize=12, color='#D62728'))
    for j, sig_s in enumerate(sell_sig_DES_S):
        ax[1].text(sig_s, 0.35, f'SELL_{j+1}', va='bottom', transform=ax[1].transData, ha='center',\
                   bbox=dict(facecolor='white', edgecolor=color_short, boxstyle='round,pad=0.2'), fontdict=dict(fontsize=12, color=color_short))

    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.11, hspace=0.35)
    if save_fig:
        plt.savefig(f"{_D}/model_output/ensemble_{stock_id}.png", facecolor='white', dpi=300)
    if not show_fig:
        plt.close(fig)
    

    return acc_buy_DES_S, acc_sell_DES_S, cumAsset_DES_S, cumStock_DES_S, sig_buy_DES_S, sig_sell_DES_S, buy_action_DES_S, sell_action_DES_S, df_DES_S


def plot_academic_price_features(stock_id, stock_name, stock_price, X_all, threshold, used_feats=None):
    """繪製學術論文風格圖：股價 + 各面向信號（可依 used_feats 篩選）。"""
    # 依固定順序顯示，確保不同股票/實驗圖面一致
    selected_cols = [f for f in FEATURE_ORDER if f in X_all.columns]
    if used_feats is not None:
        selected_cols = [f for f in selected_cols if f in used_feats]

    if len(selected_cols) == 0:
        print('[PLOT][WARN] 無可用面向可繪製，略過學術風格圖。')
        return None

    n_panels = 1 + len(selected_cols)
    fig_height = max(9, 2.1 * n_panels)

    # 用淺底網格與高對比配色，適合論文印刷與投影片展示
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(14, fig_height),
            sharex=True,
            constrained_layout=True,
        )

        if n_panels == 1:
            axes = [axes]

        fig.suptitle(
            f"{stock_id} {stock_name} | Price and Feature Signals (Academic Style)",
            fontsize=14,
            fontweight='bold',
            y=1.01,
        )

        # 全域字型與線條細節
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
        for ax in axes:
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

    os.makedirs('evaluation', exist_ok=True)
    mode_suffix = 'full' if used_feats is None else 'drop_' + '_'.join([f for f in FEATURE_ORDER if f not in used_feats])
    out_png = f"evaluation/academic_price_features_{stock_id}_{mode_suffix}.png"
    fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')

    if not show_fig:
        plt.close(fig)

    print(f"[PLOT] 學術風格圖輸出完成: {out_png}")
    return out_png

# CUSUM 檔案放在上一層 finlab/ 目錄（由其他流程維護更新）
if sys.platform.startswith('linux'):
    _FINLAB_ROOT = '/mnt/c/Users/tungl/finlab'
else:
    _FINLAB_ROOT = r'C:/Users/tungl/finlab'
out_dir = f'{_FINLAB_ROOT}/cumSum'           # CUSUM 統計量資料夾
out_dir_P = f'{_FINLAB_ROOT}/cumSum_prob_6'  # CUSUM 機率資料夾

def re_DES(x):
    """可選的 CUSUM 過濾函式（目前停用）。
    若啟用，當 DES 信號與 CUSUM 方向矛盾時，將 DES 設為中性值 0.5。
    """
    #if (x['AGG_DES']-0.5) * x['cumSum'] < 0:
        #x['AGG_DES'] = 0.5
    return x

# =============================================================================
# 主程式執行區塊
# =============================================================================
# --- 信號參數設定（所有股票共用）---
span = 1              # EWM 平滑跨度（span=1 表示不平滑）
long = 1              # 買入信號需連續看多的天數
short = 1             # 賣出信號需連續看空的天數
short_to_long = 0     # 從空轉多過渡天數
long_to_short = 0     # 從多轉空過渡天數
threshold = 0.50      # 機率 > threshold 視為看多信號
period = ['2019-12-31']  # 模型訓練起始節點（可加入多個時間點進行溻動更新）
eval_start = '2021-12-31'  # 回測/繪圖起始日期
# 回測/繪圖結束日期；None 代表每次依該檔 DES 訊號最後一天動態決定
eval_end = None

def _parse_drop(drop_str):
    """解析 drop 字串 -> used_feats (list)。空字串 = 全選，回傳 None。"""
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


def _normalize_ticker_input(ticker_raw):
    """支援 2330 / 2330.TT / 2330.TW；回傳資料代號與顯示代號。"""
    t = str(ticker_raw).strip().upper()
    if t.endswith('.TT') or t.endswith('.TW'):
        data_id = t.split('.')[0]
        display_id = t
    else:
        data_id = t
        display_id = f"{data_id}.TT"
    return data_id, display_id


def _to_english_stock_name(stock_id, stock_name):
    """指定股票名稱英文化（目前優先處理 2330）。"""
    alias = {
        '2330': 'TSMC',
        '2454': 'MediaTek',
    }
    return alias.get(str(stock_id), stock_name)

while True:
    ticker = input("Please input stock_id (輸入 0 離開): ").strip()
    if ticker == '0' or ticker == '':
        print("結束程式。")
        break
    data_ticker, display_ticker = _normalize_ticker_input(ticker)

    # drop 特徵互動式選擇
    print("可用特徵：fundamental, trade, tech_trend, moment, sentiment, macro")
    drop_in = input("要 drop 的特徵 (逗號分隔；直接 Enter = 全部使用): ").strip()
    try:
        used_feats = _parse_drop(drop_in)
    except Exception as e:
        print(f"[ERROR] drop 特徵輸入錯誤: {e}")
        continue

    # 啟用 CUSUM filter
    cusum_in = input("啟用 CUSUM filter? (1=是 / 2=否) [1]: ").strip()
    if cusum_in == '':
        use_cusum_filter = True
    else:
        use_cusum_filter = cusum_in in ('1', 'y', 'Y', 'yes', 'true')

    # 強制重訓（忽略既有 cache）
    retrain_in = input("強制重訓 DES/RF? (1=是 / 2=否) [2]: ").strip()
    if retrain_in == '':
        force_retrain = False
    else:
        force_retrain = retrain_in in ('1', 'y', 'Y', 'yes', 'true')

    shap_in = input("啟用 SHAP explainer? (1=是 / 2=否) [2]: ").strip()
    if shap_in == '':
        use_shap = False
    else:
        use_shap = shap_in in ('1', 'y', 'Y', 'yes', 'true')

    # SHAP 預設值：若未啟用 SHAP，直接沿用這些值並略過後續互動問題
    force_shap_recompute = False
    shap_background_k = 50
    shap_chunk_size = 20
    shap_nsamples = 'auto'
    shap_waterfall_idx = -1

    if use_shap:
        shap_force_in = input("強制重算 SHAP（忽略既有 SHAP cache）? (1=是 / 2=否) [2]: ").strip()
        if shap_force_in == '':
            force_shap_recompute = False
        else:
            force_shap_recompute = shap_force_in in ('1', 'y', 'Y', 'yes', 'true')

        bg_in = input("SHAP 背景 kmeans 中心數 [50]: ").strip()
        try:
            shap_background_k = int(bg_in) if bg_in else 50
        except ValueError:
            print(f"[WARN] 無法解析 '{bg_in}'，改用預設 50")
            shap_background_k = 50

        chunk_in = input("SHAP 計算 chunk size [20]: ").strip()
        try:
            shap_chunk_size = int(chunk_in) if chunk_in else 20
        except ValueError:
            print(f"[WARN] 無法解析 '{chunk_in}'，改用預設 20")
            shap_chunk_size = 20

        ns_in = input("SHAP nsamples [auto]: ").strip()
        if ns_in == '':
            shap_nsamples = 'auto'
        else:
            try:
                shap_nsamples = int(ns_in)
            except ValueError:
                print(f"[WARN] 無法解析 '{ns_in}'，改用 auto")
                shap_nsamples = 'auto'

        wf_in = input("waterfall 樣本索引（可用負值，-1=最後一筆）[-1]: ").strip()
        try:
            shap_waterfall_idx = int(wf_in) if wf_in else -1
        except ValueError:
            print(f"[WARN] 無法解析 '{wf_in}'，改用預設 -1")
            shap_waterfall_idx = -1

    thr_in = input(f"DES 信號門檻 (0.50~0.95) [{threshold}]: ").strip()
    try:
        _thr = float(thr_in) if thr_in else threshold
    except ValueError:
        print(f"[WARN] 無法解析 '{thr_in}'，改用預設 {threshold}")
        _thr = threshold

    # 預設不輸出 academic 圖；僅在明確要求時才輸出
    academic_in = input("輸出 academic 圖? (1=是 / 2=否) [2]: ").strip()
    if academic_in == '':
        export_academic = False
    else:
        export_academic = academic_in in ('1', 'y', 'Y', 'yes', 'true')

    _used_display = used_feats if used_feats is not None else FEATURE_ORDER
    _drop_display = [f for f in FEATURE_ORDER if f not in _used_display]
    _mode = 'ABLATION' if used_feats is not None else 'FULL'
    print(f"[FEATURES] mode={_mode} | used={_used_display} | drop={_drop_display}")
    print(f"[CFG] {display_ticker}: CUSUM filter = {'ON' if use_cusum_filter else 'OFF'}, DES threshold = {_thr}, force_retrain = {'ON' if force_retrain else 'OFF'}")
    print(
        f"[CFG][SHAP] enable={'ON' if use_shap else 'OFF'}, force_recompute={'ON' if force_shap_recompute else 'OFF'}, "
        f"bg_k={shap_background_k}, chunk={shap_chunk_size}, nsamples={shap_nsamples}, waterfall_idx={shap_waterfall_idx}"
    )
    print(f"[CFG][PLOT] academic = {'ON' if export_academic else 'OFF'}")

    try:
        if use_shap:
            AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name, explain_context = update_DES(
                [data_ticker],
                train_end,
                used_feats=used_feats,
                force_retrain=force_retrain,
                return_explain_context=True,
            )
        else:
            AGG_DES, AGG_RF, stock_price, stock_id, X_all, stock_name = update_DES(
                [data_ticker],
                train_end,
                used_feats=used_feats,
                force_retrain=force_retrain,
                return_explain_context=False,
            )
            explain_context = None

        stock_id_display = display_ticker
        stock_name = _to_english_stock_name(stock_id, stock_name)

        # 動態回測終點：預設取 DES 訊號的最後一天，若有手動指定 eval_end 則以手動值為準。
        des_last_day = pd.to_datetime(AGG_DES.index.max()).strftime('%Y-%m-%d')
        _eval_end = eval_end if eval_end else des_last_day
        print(f"[DATE] DES last day = {des_last_day}, eval_end setting = {eval_end}, effective end = {_eval_end}")

        # 只保留指定測試期間，避免圖表與回測混入過早資料
        stock_price = stock_price.loc[eval_start:_eval_end].copy()
        if stock_price.empty:
            raise RuntimeError(f"{stock_id_display}: 指定測試期間 {eval_start}~{_eval_end} 無股價資料")

        AGG_DES = AGG_DES[~AGG_DES.index.duplicated(keep='last')]
        AGG_DES = AGG_DES.ewm(span=span, adjust=False).mean()
        AGG_DES = AGG_DES.reindex(stock_price.index)
        AGG_DES = AGG_DES.ffill()
        AGG_DES = AGG_DES.bfill()
        print(f"[DATE] AGG_DES range = {AGG_DES.index.min().date()} ~ {AGG_DES.index.max().date()}")

        AGG_RF = AGG_RF[~AGG_RF.index.duplicated(keep='last')]
        AGG_RF = AGG_RF.ewm(span=span, adjust=False).mean()
        AGG_RF = AGG_RF.reindex(stock_price.index)
        AGG_RF = AGG_RF.ffill()
        AGG_RF = AGG_RF.bfill()
        print(f"[DATE] stock_price range = {stock_price.index.min().date()} ~ {stock_price.index.max().date()}")

        cumSum = pd.read_csv(f'{out_dir}/cusum_{stock_id}.csv', index_col=0, parse_dates=True)
        cumSum = cumSum[period[0]:]
        cumSum = cumSum.reindex(stock_price.index)
        cumSum = cumSum.loc[eval_start:_eval_end]
        cumSum = cumSum.bfill()
        print(f"[DATE] cumSum non-null max = {cumSum.dropna().index.max().date() if not cumSum.dropna().empty else 'None'}")

        cumSum_prob = pd.read_csv(f'{out_dir_P}/cumsum_prob_{stock_id}.csv', index_col=0, parse_dates=True).squeeze("columns")
        cumSum_prob = cumSum_prob[period[0]:]
        cumSum_prob = cumSum_prob.reindex(stock_price.index)
        cumSum_prob = cumSum_prob.loc[eval_start:_eval_end]
        cumSum_prob = cumSum_prob.ffill()
        print(f"[DATE] cumSum_prob non-null max = {cumSum_prob.dropna().index.max().date() if not cumSum_prob.dropna().empty else 'None'}")

        AGG_DES_adj = AGG_DES * 0.6
        cumSum_prob_adj = cumSum_prob * 0.4
        AGG_DES_P = AGG_DES_adj.add(cumSum_prob_adj)
        AGG_DES_P = AGG_DES_P.bfill()

        data = {'AGG_DES': AGG_DES_P.values, 'cumSum': cumSum.values.flatten()}
        AGG_DES_temp = pd.DataFrame(data=data, index=cumSum.index)
        AGG_DES_S = AGG_DES_temp.apply(re_DES, axis=1)
        AGG_DES_S = AGG_DES_S['AGG_DES']

        acc_buy_DES, acc_sell_DES, cumAsset_DES, cumStock_DES, sig_buy_DES, sig_sell_DES, buy_action_DES, sell_action_DES, df_DES_S\
            = plot_performance(long, short, short_to_long, long_to_short, _thr, AGG_DES_S, AGG_DES, AGG_DES_P, period, cumSum, stock_id_display, stock_name, use_cusum_filter=use_cusum_filter)

        # 額外輸出：學術論文風格（僅在使用者明確要求時輸出）
        if export_academic:
            plot_academic_price_features(
                stock_id=stock_id,
                stock_name=stock_name,
                stock_price=stock_price,
                X_all=X_all,
                threshold=_thr,
                used_feats=used_feats,
            )

        df_buy = df_DES_S[df_DES_S['buy_action'] == 1]['asset']
        df_sell = df_DES_S[df_DES_S['sell_action'] == 1]['asset']
        if len(df_buy) == len(df_sell):
            gain = df_sell.values - df_buy.values
        else:
            df_sell = pd.concat([df_sell, df_DES_S.iloc[[-1]]['asset']])
            gain = df_sell.values - df_buy.values

        profit = gain[gain > 0]
        loss = gain[gain < 0]
        transaction = len(df_sell)
        win = len(profit)
        if transaction != 0:
            win_rate = win / len(df_sell)
        else:
            win_rate = np.inf
        avg_win = profit.mean()
        avg_loss = loss.mean()
        win_loss_ratio = (avg_win / abs(avg_loss))

        print('交易次數: ', transaction)
        print('獲利次數: ', win)
        print('勝率: {:14.2f}'.format(win_rate))
        print('總獲利: {:12.2f}'.format(np.sum(profit)))
        print('平均獲利: {:10.2f}'.format(avg_win))
        print('總損失:  {:11.2f}'.format(np.sum(loss)))
        print('平均損失: {:10.2f}'.format(avg_loss))
        print('盈虧比: {:12.2f}'.format(win_loss_ratio))
        W_L = pd.DataFrame({'日期': df_DES_S.index[-1], '股票代號': [stock_id_display], '股票名稱': [stock_name], '交易次數': [transaction],
                            '獲利次數': [win], '總獲利': [np.sum(profit)], '平均獲利': [avg_win],
                            '總損失': [np.sum(loss)], '平均損失': [avg_loss], '盈虧比': [win_loss_ratio]})
        W_L.set_index('日期', inplace=True)

        if use_shap:
            if explain_context is None or explain_context.get('model') is None:
                print('[SHAP][WARN] 找不到可用 DES 模型，略過 SHAP 計算。')
            else:
                shap_tag = build_shap_tag(stock_id, period[0], used_feats)
                local_csv = Path(SHAP_OUTPUT_DIR) / f"local_{shap_tag}.csv"
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
                        print('[SHAP] 建立 KernelExplainer 並計算整個測試區間 SHAP，可能需要一些時間...')
                        explainer = build_kernel_explainer(
                            explain_context['model'],
                            explain_context['X_train'],
                            background_k=shap_background_k,
                        )
                        local_shap_df = compute_shap_local(
                            explainer,
                            x_test,
                            chunk_size=shap_chunk_size,
                            nsamples=shap_nsamples,
                        )
                        global_df = compute_shap_global(local_shap_df)
                        save_shap_artifacts(
                            tag=shap_tag,
                            explainer=explainer,
                            X_target=x_test,
                            local_shap_df=local_shap_df,
                            global_df=global_df,
                            waterfall_row=shap_waterfall_idx,
                        )

                    print('[SHAP] Global importance Top 6:')
                    print(global_df.head(6).to_string(index=False))

        if show_fig:
            plt.show()
    except Exception as e:
        print(f"[ERROR] 處理 {ticker} 時發生錯誤: {e}")
    finally:
        plt.close('all')