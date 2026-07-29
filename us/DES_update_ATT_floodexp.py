"""DES_update_ATT_floodexp.py

在 flooding 消融實驗 (none / static / dynamic) 的 ATT 預測之上訓練 DES (KNORAE)，
並做 DES 自己的 train / val / test 三段切分（依日期，與 ATT 對齊）。

讀取:
    D:/experiment_flood/<mode>/ATT_<aspect>_<stock>/
        experiment_result_train_*.csv   # ATT 訓練段預測（每 repeat 一檔）
        experiment_result_val_*.csv     # ATT 驗證段預測
        experiment_result_test_*.csv    # ATT 測試段預測

流程（每個 FLOOD_MODE）:
    1. 對每個面向 (6 aspects)，將所有 repeats 的預測取均值得到單一機率序列
    2. 將 6 個面向堆疊為欄位 -> X_all (DataFrame, columns=aspects, index=Date)
    3. y 從 D:/Feature_new/fundamental_<stock>.csv['y_20'] reindex 到 X_all
    4. 切分:
         DES train : index <= TRAIN_END        (預設 2024-09-02)
         DES val   : VAL_START..VAL_END        (預設 2024-09-03..2025-12-31)
         DES test  : index >= TEST_START       (預設 2026-01-01)
    5. RandomizedSearchCV 訓練 RandomForest -> 包 KNORAE
    6. 對 val/test 計算 ROC-AUC, PR-AUC, F1, accuracy
    7. 存預測 CSV、metrics JSON、模型 PKL

最後輸出三模式比較圖到 D:/evaluation_plot/_compare/des_compare_<stock>.png

可設環境變數覆寫上述路徑與日期（見檔頭 docstring）。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

# deslib 0.3.7 vs sklearn>=1.7 相容性（同 DES_update_ATT-sentiment.py）
try:
    import sklearn.utils.validation as _skv
    if not hasattr(_skv, '_check_pos_label_consistency'):
        from sklearn.metrics._classification import _check_pos_label_consistency as _c
        _skv._check_pos_label_consistency = _c
except Exception:  # noqa: BLE001
    pass

# sklearn 1.6+ 移除 BaseEstimator._validate_data（搬成獨立函式 validate_data）。
# deslib.base.fit 仍呼叫 self._validate_data(...)，這裡掛一個 shim 還原舊介面。
try:
    from sklearn.base import BaseEstimator as _BE
    if not hasattr(_BE, '_validate_data'):
        from sklearn.utils.validation import validate_data as _vd

        def _validate_data_shim(self, *args, **kwargs):
            return _vd(self, *args, **kwargs)

        _BE._validate_data = _validate_data_shim
except Exception:  # noqa: BLE001
    pass

from deslib.des.knora_e import KNORAE  # noqa: E402

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

matplotlib.use('Agg')

# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
STOCK_ID = os.getenv('STOCK_ID', '2330')
FLOOD_MODES = [m.strip() for m in os.getenv('FLOOD_MODES', 'none,static,dynamic').split(',') if m.strip()]
ASPECTS = [a.strip() for a in os.getenv(
    'ASPECTS', 'fundamental,trade,tech_trend,moment,sentiment,macro'
).split(',') if a.strip()]
EXP_ROOT = Path(os.getenv('EXP_ROOT', 'D:/experiment_flood'))
OUTPUT_ROOT = Path(os.getenv('OUTPUT_ROOT', 'D:/DES_flood'))
CMP_DIR = Path(os.getenv('CMP_DIR', 'D:/evaluation_plot/_compare'))
DATA_ROOT = Path(os.getenv('DATA_ROOT', 'D:/Feature_new'))

TRAIN_END = pd.Timestamp(os.getenv('TRAIN_END', '2024-09-02'))
VAL_START = pd.Timestamp(os.getenv('VAL_START', '2024-09-03'))
VAL_END = pd.Timestamp(os.getenv('VAL_END', '2025-12-31'))
TEST_START = pd.Timestamp(os.getenv('TEST_START', '2026-01-01'))

RF_ITER = int(os.getenv('RF_ITER', '30'))
RF_CV = int(os.getenv('RF_CV', '5'))
KNORAE_K = int(os.getenv('KNORAE_K', '10'))
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))

MODE_COLORS = {'none': '#4C72B0', 'static': '#DD8452', 'dynamic': '#55A868'}

# ----------------------------------------------------------------------
# 輔助函式
# ----------------------------------------------------------------------
def _read_pred_csv(path: str) -> pd.Series:
    """讀單一 experiment_result_<period>_<r>.csv (兩欄: date, prob)。"""
    s = pd.read_csv(path, index_col=0, parse_dates=True, header=None).squeeze('columns')
    s = s[~s.index.duplicated(keep='last')]
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors='coerce').dropna()
    return s


def _gather_aspect_preds(mode: str, aspect: str, stock_id: str) -> pd.Series | None:
    """讀某 mode + aspect 全部 repeats 的 train+val+test 預測，跨 repeat 取均值。"""
    base = EXP_ROOT / mode / f'ATT_{aspect}_{stock_id}'
    if not base.exists():
        return None
    series_list: list[pd.Series] = []
    for period in ('train', 'val', 'test'):
        files = sorted(glob.glob(str(base / f'experiment_result_{period}_*.csv')))
        if not files:
            continue
        per_repeat: list[pd.Series] = []
        for f in files:
            try:
                per_repeat.append(_read_pred_csv(f))
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] failed to read {f}: {e}")
        if not per_repeat:
            continue
        avg = pd.DataFrame(per_repeat).T.mean(axis=1)
        series_list.append(avg)
    if not series_list:
        return None
    merged = pd.concat(series_list).sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]
    return merged


def build_X_all(mode: str, stock_id: str) -> tuple[pd.DataFrame, list[str]]:
    """合併 6 個面向的平均預測為 X_all。回傳 (DataFrame, present_aspects)。"""
    cols: dict[str, pd.Series] = {}
    for aspect in ASPECTS:
        s = _gather_aspect_preds(mode, aspect, stock_id)
        if s is None or s.empty:
            print(f"  [SKIP] {mode}/{aspect}: no prediction files")
            continue
        cols[aspect] = s
    if not cols:
        raise RuntimeError(f"mode={mode}: 沒有任何面向有預測檔可用")
    X = pd.DataFrame(cols)
    X = X.dropna(how='all').ffill().bfill().fillna(0.5).astype('float64')
    return X, list(cols.keys())


def load_y(stock_id: str) -> pd.Series:
    """讀 y_20 標籤。"""
    fp = DATA_ROOT / f'fundamental_{stock_id}.csv'
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    if 'y_20' not in df.columns:
        raise KeyError(f"'y_20' 不在 {fp}")
    y = df['y_20']
    y.index = pd.to_datetime(y.index)
    return y


def fit_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """用 RandomizedSearchCV 找 RF 最佳參數（class_weight 平衡）。"""
    n_estimators = [int(x) for x in np.linspace(200, 2000, 10)]
    max_features = ['sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, 11)] + [None]
    param_dist = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=RF_ITER,
        cv=RF_CV,
        scoring='average_precision',  # PR-AUC，呼應 ATT 階段目標
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def compute_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = THRESHOLD) -> dict:
    """val/test 上的指標。"""
    mask = y_true.notna() & y_prob.notna()
    yt = y_true.loc[mask].astype(int).to_numpy()
    yp = y_prob.loc[mask].to_numpy()
    if len(yt) == 0:
        return {'n': 0}
    yhat = (yp >= threshold).astype(int)
    out = {
        'n': int(len(yt)),
        'pos_rate': float(yt.mean()),
        'pred_pos_rate': float(yhat.mean()),
        'accuracy': float(accuracy_score(yt, yhat)),
        'f1': float(f1_score(yt, yhat, zero_division=0)),
    }
    if len(np.unique(yt)) > 1:
        out['roc_auc'] = float(roc_auc_score(yt, yp))
        out['pr_auc'] = float(average_precision_score(yt, yp))
    else:
        out['roc_auc'] = float('nan')
        out['pr_auc'] = float('nan')
    return out


# ----------------------------------------------------------------------
# 單一 mode 流程
# ----------------------------------------------------------------------
def run_one_mode(mode: str, y_all: pd.Series, stock_id: str) -> dict:
    print(f"\n========== FLOOD_MODE={mode} ==========")
    X_all, present = build_X_all(mode, stock_id)
    print(f"  X_all shape={X_all.shape}, aspects={present}")
    print(f"  date range: {X_all.index.min().date()} -> {X_all.index.max().date()}")

    # 對齊 y
    y = y_all.reindex(X_all.index)

    # 切分
    train_mask = X_all.index <= TRAIN_END
    val_mask = (X_all.index >= VAL_START) & (X_all.index <= VAL_END)
    test_mask = X_all.index >= TEST_START
    X_tr, y_tr = X_all.loc[train_mask], y.loc[train_mask]
    X_va, y_va = X_all.loc[val_mask], y.loc[val_mask]
    X_te, y_te = X_all.loc[test_mask], y.loc[test_mask]
    # 訓練資料不能有 NaN 標籤
    train_keep = y_tr.notna()
    X_tr = X_tr.loc[train_keep]
    y_tr = y_tr.loc[train_keep].astype(int)
    print(f"  splits  train n={len(X_tr)}  val n={len(X_va)}  test n={len(X_te)}")
    if len(X_tr) < 30:
        raise RuntimeError(f"{mode}: DES 訓練樣本過少 (n={len(X_tr)})")
    if len(np.unique(y_tr)) < 2:
        raise RuntimeError(f"{mode}: DES 訓練標籤只有一個類別")

    # RF
    print("  fitting RF (RandomizedSearchCV)...")
    rf = fit_rf(X_tr, y_tr)
    print(f"  RF best params: {rf.get_params()}")

    # KNORAE 需要 base classifier 已 fit; 使用 RF 的 estimators_ 作為 pool
    # 注意: deslib 0.3.7 + KNORAE 期望 pool_classifiers 為一個已 fit 的分類器（會自動拆出 estimators_）
    print(f"  fitting KNORAE (k={KNORAE_K})...")
    des = KNORAE(pool_classifiers=rf, k=KNORAE_K, DFP=True)
    des.fit(X_tr.to_numpy(), y_tr.to_numpy())

    # 預測（全範圍以便輸出時序圖；同時取出 val / test 子集）
    p_all_rf = pd.Series(rf.predict_proba(X_all.to_numpy())[:, 1], index=X_all.index)
    p_all_des = pd.Series(des.predict_proba(X_all.to_numpy())[:, 1], index=X_all.index)

    # 指標
    metrics = {
        'mode': mode,
        'val_des': compute_metrics(y.loc[val_mask], p_all_des.loc[val_mask]),
        'val_rf': compute_metrics(y.loc[val_mask], p_all_rf.loc[val_mask]),
        'test_des': compute_metrics(y.loc[test_mask], p_all_des.loc[test_mask]),
        'test_rf': compute_metrics(y.loc[test_mask], p_all_rf.loc[test_mask]),
        'aspects_used': present,
        'split': {
            'train_end': str(TRAIN_END.date()),
            'val_start': str(VAL_START.date()),
            'val_end': str(VAL_END.date()),
            'test_start': str(TEST_START.date()),
            'n_train': int(len(X_tr)),
            'n_val': int(len(X_va)),
            'n_test': int(len(X_te)),
        },
    }
    print(f"  [METRICS] val_des  PR-AUC={metrics['val_des'].get('pr_auc'):.3f}  "
          f"ROC-AUC={metrics['val_des'].get('roc_auc'):.3f}  "
          f"F1={metrics['val_des'].get('f1'):.3f}  acc={metrics['val_des'].get('accuracy'):.3f}")
    print(f"  [METRICS] test_des PR-AUC={metrics['test_des'].get('pr_auc'):.3f}  "
          f"ROC-AUC={metrics['test_des'].get('roc_auc'):.3f}  "
          f"F1={metrics['test_des'].get('f1'):.3f}  acc={metrics['test_des'].get('accuracy'):.3f}")

    # 輸出
    out_dir = OUTPUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all_des.to_csv(out_dir / f'des_pred_{stock_id}.csv', header=False)
    p_all_rf.to_csv(out_dir / f'rf_pred_{stock_id}.csv', header=False)
    joblib.dump(des, out_dir / f'des_model_{stock_id}.pkl')
    joblib.dump(rf, out_dir / f'rf_model_{stock_id}.pkl')
    with open(out_dir / f'metrics_{stock_id}.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OUT] {out_dir}/")
    return metrics


# ----------------------------------------------------------------------
# 比較圖
# ----------------------------------------------------------------------
def plot_compare(all_metrics: list[dict], stock_id: str) -> None:
    """三模式 × {val, test} × {PR-AUC, ROC-AUC, F1, accuracy} bar chart。"""
    CMP_DIR.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ['pr_auc', 'roc_auc', 'f1', 'accuracy']
    periods = ['val', 'test']
    fig, axes = plt.subplots(2, len(metrics_to_plot),
                             figsize=(3.4 * len(metrics_to_plot), 6.4), sharey='row')
    modes_present = [m['mode'] for m in all_metrics]
    x = np.arange(len(modes_present))
    width = 0.55

    for row, period in enumerate(periods):
        for col, metric in enumerate(metrics_to_plot):
            ax = axes[row, col]
            vals = []
            for m in all_metrics:
                v = m[f'{period}_des'].get(metric)
                vals.append(np.nan if v is None else v)
            colors = [MODE_COLORS.get(mm, '#888') for mm in modes_present]
            bars = ax.bar(x, vals, width, color=colors, edgecolor='black', linewidth=0.6)
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(modes_present, fontsize=9)
            ax.set_ylim(0.0, max(1.0, np.nanmax(vals) * 1.15 if vals else 1.0))
            ax.axhline(0.5, color='gray', linewidth=0.7, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col == 0:
                ax.set_ylabel(f'{period.upper()}\n{metric}', fontsize=10)
            else:
                ax.set_ylabel(metric, fontsize=10)
            if row == 0:
                ax.set_title(metric, fontsize=11)

    fig.suptitle(f'DES on Flooding ATT outputs — {stock_id} (3 modes × val/test)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = CMP_DIR / f'des_compare_{stock_id}.png'
    out_pdf = CMP_DIR / f'des_compare_{stock_id}.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[CMP] {out_png}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] STOCK_ID={STOCK_ID}  modes={FLOOD_MODES}")
    print(f"[CONFIG] EXP_ROOT={EXP_ROOT}  OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"[CONFIG] splits: train<= {TRAIN_END.date()}  "
          f"val={VAL_START.date()}..{VAL_END.date()}  test>= {TEST_START.date()}")

    y_all = load_y(STOCK_ID)

    all_metrics: list[dict] = []
    for mode in FLOOD_MODES:
        if not (EXP_ROOT / mode).exists():
            print(f"[SKIP] mode={mode}: {EXP_ROOT / mode} not exist")
            continue
        try:
            m = run_one_mode(mode, y_all, STOCK_ID)
            all_metrics.append(m)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] mode={mode}: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        plot_compare(all_metrics, STOCK_ID)
        with open(OUTPUT_ROOT / f'metrics_summary_{STOCK_ID}.json', 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY] {OUTPUT_ROOT}/metrics_summary_{STOCK_ID}.json")
    else:
        print("[WARN] no mode produced metrics")
    return 0


if __name__ == '__main__':
    sys.exit(main())
