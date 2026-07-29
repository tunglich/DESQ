"""Combined_output.py

Combined-5 features 一條龍輸出：
  1. 從 EXPERIMENT_ROOT/<FLOOD_MODE>/ATT_combined5_<stock>/ 偵測 Dflooding 保留的
     top-3 repeats（看 experiment_<r>.keras 殘留）
  2. 讀那 3 個 repeats 的 train/val/test 預測 CSV，跨 repeat 取平均 → 最終 prob 序列
  3. 對齊 y_20 計算 val / test 指標（ROC-AUC, PR-AUC, F1, accuracy）
  4. 簡易回測：long-only signal=(prob>=thr) shift(1) → daily return = pos * close.pct_change()
     跟 buy-and-hold 比 equity curve / Sharpe / max drawdown
  5. 輸出 CSV / metrics JSON / plots

執行範例:
    STOCK_ID=2330 FLOOD_MODE=dynamic \
    EXPERIMENT_OUTPUT_DIR=D:/experiment_flood_combined_v2 \
    OUTPUT_ROOT=D:/Combined_output \
    TRAIN_END_DATE=2023-12-31 TEST_START_DATE=2024-01-01 TEST_END_DATE=2026-06-03 \
    python Combined_output.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
matplotlib.use('Agg')


def platform_path(path_str: str) -> str:
    """Windows `D:/...` -> Linux/WSL `/mnt/d/...`（其它系統原樣回傳）。"""
    if os.name != 'nt' and len(path_str) >= 2 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path_str


# ----------------------------------------------------------------------
# 設定（env-driven）
# ----------------------------------------------------------------------
STOCK_ID = os.getenv('STOCK_ID', '2330')
FLOOD_MODE = os.getenv('FLOOD_MODE', 'dynamic').strip().lower()
ASPECT = os.getenv('COMBINED_ASPECT', 'combined5')

EXPERIMENT_OUTPUT_DIR = Path(platform_path(os.getenv('EXPERIMENT_OUTPUT_DIR', 'D:/experiment_flood')))
EXP_DIR = EXPERIMENT_OUTPUT_DIR / FLOOD_MODE / f'ATT_{ASPECT}_{STOCK_ID}'

OUTPUT_ROOT = Path(platform_path(os.getenv('OUTPUT_ROOT', 'D:/Combined_output')))
OUT_DIR = OUTPUT_ROOT / f'{STOCK_ID}_{ASPECT}_{FLOOD_MODE}'
PLOT_DIR = OUT_DIR / 'plots'

DATA_ROOT = Path(platform_path(os.getenv('DATA_ROOT', 'D:/Feature_new')))
LABEL_FILE = DATA_ROOT / f'fundamental_{STOCK_ID}.csv'
LABEL_COL = os.getenv('LABEL_COL', 'y_20')
PRICE_FILE = Path(platform_path(os.getenv('PRICE_FILE', str(DATA_ROOT / f'tech_trend_{STOCK_ID}.csv'))))
PRICE_COL = os.getenv('PRICE_COL', 'close')

# 切分日期：保持與 ATT 訓練端一致
TRAIN_END_DATE = pd.Timestamp(os.getenv('TRAIN_END_DATE', '2023-12-31'))
TEST_START_DATE = pd.Timestamp(os.getenv('TEST_START_DATE', '2024-01-01'))
TEST_END_DATE = pd.Timestamp(os.getenv('TEST_END_DATE', '2026-06-03'))

DEFAULT_THRESHOLD = float(os.getenv('DEFAULT_THRESHOLD', '0.5'))
# 風險假設
ANNUAL_TRADING_DAYS = int(os.getenv('ANNUAL_TRADING_DAYS', '252'))
RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.0'))


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def detect_surviving_repeats(exp_dir: Path) -> list[int]:
    """從 experiment_<r>.keras 推回 Dflooding 保留的 top-N repeat 編號。"""
    keras_files = sorted(exp_dir.glob('experiment_*.keras'))
    pat = re.compile(r'experiment_(\d+)\.keras$')
    repeats: list[int] = []
    for f in keras_files:
        m = pat.search(f.name)
        if m:
            repeats.append(int(m.group(1)))
    repeats.sort()
    return repeats


def _read_pred_csv(path: Path) -> pd.Series:
    """讀單一 experiment_result_<period>_<r>.csv (兩欄: date, prob, no header)。"""
    s = pd.read_csv(path, index_col=0, parse_dates=True, header=None).squeeze('columns')
    s = s[~s.index.duplicated(keep='last')]
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors='coerce').dropna()
    return s


def load_period_preds(exp_dir: Path, period: str, repeats: list[int]) -> pd.DataFrame:
    """讀指定 repeats 的某段 (train/val/test) 預測，回傳 DataFrame[date x repeat]。"""
    cols: dict[int, pd.Series] = {}
    for r in repeats:
        fp = exp_dir / f'experiment_result_{period}_{r}.csv'
        if not fp.exists():
            print(f"  [WARN] missing {fp.name}, skip repeat {r}")
            continue
        cols[r] = _read_pred_csv(fp)
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df


def load_labels() -> pd.Series:
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"label file not found: {LABEL_FILE}")
    df = pd.read_csv(LABEL_FILE, index_col=0, parse_dates=True)
    if LABEL_COL not in df.columns:
        raise KeyError(f"'{LABEL_COL}' not in {LABEL_FILE}; cols={list(df.columns)[:10]}...")
    y = pd.to_numeric(df[LABEL_COL], errors='coerce')
    y.index = pd.to_datetime(y.index)
    return y


def load_price() -> pd.Series:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"price file not found: {PRICE_FILE}")
    df = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not in {PRICE_FILE}; cols={list(df.columns)[:15]}...")
    p = pd.to_numeric(df[PRICE_COL], errors='coerce').dropna()
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


# ----------------------------------------------------------------------
# 指標 / 閾值
# ----------------------------------------------------------------------
def compute_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict:
    mask = y_true.notna() & y_prob.notna()
    yt = y_true.loc[mask].astype(int).to_numpy()
    yp = y_prob.loc[mask].to_numpy()
    if len(yt) == 0:
        return {'n': 0}
    yhat = (yp >= threshold).astype(int)
    out: dict = {
        'n': int(len(yt)),
        'pos_rate': float(yt.mean()),
        'pred_pos_rate': float(yhat.mean()),
        'threshold': float(threshold),
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


def best_f1_threshold(y_true: pd.Series, y_prob: pd.Series) -> tuple[float, float]:
    """在 val 上掃 PR 曲線找最大 F1 的閾值。"""
    mask = y_true.notna() & y_prob.notna()
    yt = y_true.loc[mask].astype(int).to_numpy()
    yp = y_prob.loc[mask].to_numpy()
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return DEFAULT_THRESHOLD, float('nan')
    prec, rec, thr = precision_recall_curve(yt, yp)
    # f1 對齊 thr 長度（precision_recall_curve 多回 1 個點）
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    f1s = f1s[:-1]
    if len(f1s) == 0:
        return DEFAULT_THRESHOLD, float('nan')
    idx = int(np.argmax(f1s))
    return float(thr[idx]), float(f1s[idx])


# ----------------------------------------------------------------------
# 回測
# ----------------------------------------------------------------------
def backtest(prob: pd.Series, price: pd.Series, threshold: float) -> dict:
    """Long-only：signal=(prob>=thr).shift(1)，daily return = signal * close.pct_change()。"""
    # 對齊到 price 索引
    df = pd.DataFrame({'prob': prob, 'price': price}).dropna()
    if df.empty:
        return {'n_days': 0}, pd.DataFrame()
    df['ret'] = df['price'].pct_change().fillna(0.0)
    df['signal'] = (df['prob'] >= threshold).astype(int).shift(1).fillna(0).astype(int)
    df['strategy_ret'] = df['signal'] * df['ret']
    df['bh_ret'] = df['ret']
    df['equity_strategy'] = (1.0 + df['strategy_ret']).cumprod()
    df['equity_bh'] = (1.0 + df['bh_ret']).cumprod()

    def _stats(r: pd.Series, eq: pd.Series) -> dict:
        if len(r) < 2:
            return {}
        mean_d = r.mean()
        std_d = r.std(ddof=1)
        ann_ret = (1.0 + mean_d) ** ANNUAL_TRADING_DAYS - 1.0
        ann_vol = std_d * np.sqrt(ANNUAL_TRADING_DAYS) if std_d > 0 else float('nan')
        sharpe = ((mean_d - RISK_FREE_RATE / ANNUAL_TRADING_DAYS) /
                  std_d * np.sqrt(ANNUAL_TRADING_DAYS)) if std_d > 0 else float('nan')
        roll_max = eq.cummax()
        dd = eq / roll_max - 1.0
        return {
            'total_return': float(eq.iloc[-1] - 1.0),
            'cagr': float(ann_ret),
            'ann_vol': float(ann_vol),
            'sharpe': float(sharpe),
            'max_drawdown': float(dd.min()),
        }

    out = {
        'n_days': int(len(df)),
        'date_start': df.index.min().strftime('%Y-%m-%d'),
        'date_end': df.index.max().strftime('%Y-%m-%d'),
        'threshold': float(threshold),
        'exposure': float(df['signal'].mean()),
        'strategy': _stats(df['strategy_ret'], df['equity_strategy']),
        'buy_and_hold': _stats(df['bh_ret'], df['equity_bh']),
    }
    return out, df


# ----------------------------------------------------------------------
# 畫圖
# ----------------------------------------------------------------------
def plot_prob_timeline(prob: pd.Series, prob_std: pd.Series, y: pd.Series,
                        thr_default: float, thr_best: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(prob.index, prob.values, color='#2E86AB', lw=1.0, label='ensemble prob (mean of top-3)')
    ax.fill_between(prob.index, (prob - prob_std).values, (prob + prob_std).values,
                    color='#2E86AB', alpha=0.15, label='±1 std across top-3')
    ax.axhline(thr_default, color='gray', ls='--', lw=0.8, label=f'thr={thr_default:.2f}')
    if not np.isnan(thr_best):
        ax.axhline(thr_best, color='#E63946', ls='--', lw=0.8, label=f'best-F1 thr={thr_best:.2f}')
    ax.axvspan(TRAIN_END_DATE, TEST_START_DATE, color='lightyellow', alpha=0.4, zorder=0)
    ax.axvline(TRAIN_END_DATE, color='black', ls=':', lw=0.8)
    ax.axvline(TEST_START_DATE, color='black', ls=':', lw=0.8)
    # y_true 散點 (1=red dot top, 0=gray dot bottom)
    y_aligned = y.reindex(prob.index)
    pos = y_aligned == 1
    neg = y_aligned == 0
    ax.scatter(prob.index[pos], np.full(pos.sum(), 1.02), s=2, c='#06A77D', alpha=0.5, label='y=1')
    ax.scatter(prob.index[neg], np.full(neg.sum(), -0.02), s=2, c='#9E9E9E', alpha=0.5, label='y=0')
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel('probability')
    ax.set_title(f'{STOCK_ID} {ASPECT} ({FLOOD_MODE}) — ensemble probability timeline')
    ax.legend(loc='upper left', fontsize=8, ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_equity(bt_df: pd.DataFrame, period_label: str, threshold: float, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(bt_df.index, bt_df['equity_strategy'], color='#2E86AB', lw=1.6,
             label=f'strategy (thr={threshold:.2f})')
    ax1.plot(bt_df.index, bt_df['equity_bh'], color='#888888', lw=1.2, ls='--', label='buy & hold')
    ax1.set_ylabel('equity (×)')
    ax1.set_title(f'{STOCK_ID} {ASPECT} ({FLOOD_MODE}) — {period_label} backtest')
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(alpha=0.3)
    dd = bt_df['equity_strategy'] / bt_df['equity_strategy'].cummax() - 1.0
    ax2.fill_between(bt_df.index, dd.values, 0, color='#E63946', alpha=0.35)
    ax2.set_ylabel('drawdown')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_metric_bars(metrics_val: dict, metrics_test_default: dict, metrics_test_best: dict,
                      out: Path) -> None:
    keys = ['roc_auc', 'pr_auc', 'f1', 'accuracy']
    groups = [('val', metrics_val), ('test @0.5', metrics_test_default), ('test @bestF1', metrics_test_best)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(keys))
    width = 0.26
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, (label, m) in enumerate(groups):
        vals = [m.get(k, float('nan')) for k in keys]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i], edgecolor='white')
        for j, v in enumerate(vals):
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                ax.text(x[j] + (i - 1) * width, v + 0.012, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{STOCK_ID} {ASPECT} ({FLOOD_MODE}) — metrics')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> int:
    print(f"[CFG] STOCK_ID={STOCK_ID}  ASPECT={ASPECT}  FLOOD_MODE={FLOOD_MODE}")
    print(f"[CFG] EXP_DIR={EXP_DIR}")
    print(f"[CFG] OUT_DIR={OUT_DIR}")
    print(f"[CFG] split  train/val_end={TRAIN_END_DATE.date()}  "
          f"test={TEST_START_DATE.date()}~{TEST_END_DATE.date()}")

    if not EXP_DIR.exists():
        print(f"[ERROR] experiment dir not found: {EXP_DIR}", file=sys.stderr)
        return 2

    repeats = detect_surviving_repeats(EXP_DIR)
    if not repeats:
        print(f"[ERROR] no surviving experiment_*.keras under {EXP_DIR}", file=sys.stderr)
        return 3
    print(f"[REPEATS] surviving top-{len(repeats)} repeats = {repeats}")

    # 讀三段預測，跨 repeat 平均
    parts: dict[str, pd.DataFrame] = {}
    for period in ('train', 'val', 'test'):
        df_p = load_period_preds(EXP_DIR, period, repeats)
        if df_p.empty:
            print(f"  [WARN] no {period} predictions loaded")
        parts[period] = df_p
        print(f"  [{period:>5}] shape={df_p.shape}  "
              f"range=[{df_p.index.min().date() if not df_p.empty else 'NA'} .. "
              f"{df_p.index.max().date() if not df_p.empty else 'NA'}]")

    # 整體機率序列 (拼接 train+val+test，重複日期取 test > val > train 順序覆寫)
    def _combine(parts_dict: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        all_df = []
        for period in ('train', 'val', 'test'):
            df_p = parts_dict[period]
            if df_p.empty:
                continue
            tmp = df_p.copy()
            tmp.columns = [f'{period}_r{r}' for r in tmp.columns]
            all_df.append(tmp)
        if not all_df:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
        wide = pd.concat(all_df, axis=1).sort_index()
        # 取平均與 std 時把 train/val/test 的同一 repeat 視為同樣機率（不會同日重疊）
        repeat_means: dict[int, pd.Series] = {}
        for r in repeats:
            cols = [c for c in wide.columns if c.endswith(f'_r{r}')]
            if not cols:
                continue
            repeat_means[r] = wide[cols].bfill(axis=1).iloc[:, 0]
        rdf = pd.DataFrame(repeat_means)
        prob_mean = rdf.mean(axis=1)
        prob_std = rdf.std(axis=1, ddof=0)
        return prob_mean, prob_std, rdf

    prob_mean, prob_std, repeat_df = _combine(parts)
    if prob_mean.empty:
        print("[ERROR] empty combined probability series", file=sys.stderr)
        return 4

    # 標籤 + 切分
    y = load_labels().reindex(prob_mean.index)

    val_prob = parts['val'].mean(axis=1) if not parts['val'].empty else pd.Series(dtype=float)
    test_prob = parts['test'].mean(axis=1) if not parts['test'].empty else pd.Series(dtype=float)
    val_y = y.reindex(val_prob.index)
    test_y = y.reindex(test_prob.index)

    # 閾值：固定 0.5 + 由 val 找 best F1
    thr_best, best_f1 = best_f1_threshold(val_y, val_prob)
    print(f"[THR] default={DEFAULT_THRESHOLD}  best-F1@val={thr_best:.3f} (F1={best_f1:.3f})")

    # 指標
    metrics = {
        'config': {
            'stock_id': STOCK_ID, 'aspect': ASPECT, 'flood_mode': FLOOD_MODE,
            'repeats_used': repeats,
            'train_val_end': str(TRAIN_END_DATE.date()),
            'test_start': str(TEST_START_DATE.date()),
            'test_end': str(TEST_END_DATE.date()),
        },
        'val_default': compute_metrics(val_y, val_prob, DEFAULT_THRESHOLD),
        'val_bestF1':  compute_metrics(val_y, val_prob, thr_best),
        'test_default': compute_metrics(test_y, test_prob, DEFAULT_THRESHOLD),
        'test_bestF1':  compute_metrics(test_y, test_prob, thr_best),
    }
    for k in ('val_default', 'val_bestF1', 'test_default', 'test_bestF1'):
        m = metrics[k]
        if m.get('n', 0) == 0:
            print(f"[METRIC] {k}: empty")
            continue
        print(f"[METRIC] {k:>12s}  n={m['n']:4d}  "
              f"ROC={m.get('roc_auc', float('nan')):.3f}  "
              f"PR={m.get('pr_auc', float('nan')):.3f}  "
              f"F1={m.get('f1', float('nan')):.3f}  "
              f"acc={m.get('accuracy', float('nan')):.3f}  "
              f"pos_rate={m.get('pos_rate', float('nan')):.3f}  "
              f"pred_pos={m.get('pred_pos_rate', float('nan')):.3f}")

    # 回測（test 段 + 全段 各畫一張）
    price = load_price()

    bt = {}
    test_prob_aligned = test_prob.copy()
    # 限制 test 段日期到 TEST_END_DATE
    test_prob_aligned = test_prob_aligned[(test_prob_aligned.index >= TEST_START_DATE) &
                                            (test_prob_aligned.index <= TEST_END_DATE)]
    for tag, thr in (('default', DEFAULT_THRESHOLD), ('bestF1', thr_best)):
        bt_stats, bt_df = backtest(test_prob_aligned, price, thr)
        bt[f'test_{tag}'] = bt_stats
        msg = bt_stats.get('strategy', {})
        print(f"[BT] test_{tag} thr={thr:.3f}  "
              f"strat_ret={msg.get('total_return', float('nan')):.3f}  "
              f"sharpe={msg.get('sharpe', float('nan')):.3f}  "
              f"maxDD={msg.get('max_drawdown', float('nan')):.3f}  "
              f"exposure={bt_stats.get('exposure', float('nan')):.3f}")
        # 存回測明細與圖
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        bt_df.to_csv(OUT_DIR / f'backtest_test_{tag}.csv')
        plot_equity(bt_df, f'test (thr={thr:.2f})', thr, PLOT_DIR / f'equity_test_{tag}.png')

    metrics['backtest'] = bt

    # 輸出 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'prob_mean': prob_mean,
        'prob_std': prob_std,
        'y_true': y,
    }).to_csv(OUT_DIR / 'predictions_all.csv', index_label='date')
    for period, df_p in parts.items():
        if df_p.empty:
            continue
        out_p = df_p.copy()
        out_p['prob_mean'] = df_p.mean(axis=1)
        out_p['prob_std'] = df_p.std(axis=1, ddof=0)
        out_p['y_true'] = y.reindex(df_p.index)
        out_p.to_csv(OUT_DIR / f'predictions_{period}.csv', index_label='date')

    # metrics JSON
    with open(OUT_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 圖
    plot_prob_timeline(prob_mean, prob_std, y, DEFAULT_THRESHOLD, thr_best,
                       PLOT_DIR / 'prob_timeline.png')
    plot_metric_bars(metrics['val_default'], metrics['test_default'], metrics['test_bestF1'],
                     PLOT_DIR / 'metric_bars.png')

    print(f"[DONE] outputs at {OUT_DIR}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
