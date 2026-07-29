"""三模式 flooding 彙整比較圖（學術專業風格）。

讀取 ATT+Dflooding_floodexp.py 三組（none / static / dynamic）輸出的 training
history CSV，對齊各模式的 3 個 repeats 逐 epoch 取 mean ± std，將三模式的
train loss 與 flooding 水位 b 疊在同一張圖，並另出 6 面向的 2x3 grid 總覽。

輸入（預設）：
    D:/experiment_flood/<mode>/ATT_<aspect>_<symbol>/history_<r>.csv
輸出：
    D:/evaluation_plot/_compare/compare_<aspect>.{png,pdf}
    D:/evaluation_plot/_compare/compare_grid_all_aspects.{png,pdf}

用法：
    python plot_flood_compare.py
    # 可用環境變數覆寫：
    #   EXPERIMENT_OUTPUT_DIR (預設 D:/experiment_flood)
    #   EVAL_PLOT_DIR         (預設 D:/evaluation_plot)
    #   STOCK_IDS             (預設 2330；取第一個)
    #   MODEL_TYPES           (預設 fundamental,trade,moment,sentiment,tech_trend,macro)
    #   STATIC_FLOOD_B        (預設 0.2，僅在 static history 缺 flooding_b 欄時用)
"""

import os
import glob
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def platform_path(path_str):
    """Windows D:/... → Linux/WSL /mnt/d/...；Windows 維持原樣。"""
    if os.name != 'nt' and len(path_str) >= 2 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path_str


EXPERIMENT_OUTPUT_DIR = platform_path(os.getenv('EXPERIMENT_OUTPUT_DIR', 'D:/experiment_flood'))
EVAL_PLOT_DIR = platform_path(os.getenv('EVAL_PLOT_DIR', 'D:/evaluation_plot'))
SYMBOL = [x.strip() for x in os.getenv('STOCK_IDS', '2330').split(',') if x.strip()][0]
MODEL_TYPES = [x.strip() for x in os.getenv(
    'MODEL_TYPES', 'fundamental,trade,moment,sentiment,tech_trend,macro'
).split(',') if x.strip()]
STATIC_FLOOD_B = float(os.getenv('STATIC_FLOOD_B', '0.2'))

MODES = ['none', 'static', 'dynamic']
MODE_COLORS = {'none': '#4C72B0', 'static': '#DD8452', 'dynamic': '#55A868'}
MODE_LABELS = {'none': 'No flooding', 'static': 'Static flooding', 'dynamic': 'Dynamic flooding'}

COMPARE_DIR = f'{EVAL_PLOT_DIR}/_compare'
DES_SUMMARY_PATH = platform_path(os.getenv(
    'DES_SUMMARY_PATH', f'D:/DES_flood/metrics_summary_{SYMBOL}.json'
))
DES_OUTPUT_ROOT = platform_path(os.getenv('DES_OUTPUT_ROOT', 'D:/DES_flood'))
DATA_ROOT = platform_path(os.getenv('DATA_ROOT', 'D:/Feature_new'))
VAL_START = pd.Timestamp(os.getenv('VAL_START', '2024-09-03'))
VAL_END = pd.Timestamp(os.getenv('VAL_END', '2025-12-31'))
TEST_START = pd.Timestamp(os.getenv('TEST_START', '2026-01-01'))


def setup_academic_style():
    """全域學術風格設定。"""
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


def _load_mode_runs(mode, aspect):
    """讀取單一 (mode, aspect) 下所有 repeats 的 history CSV，回傳 DataFrame 清單。"""
    pattern = f'{EXPERIMENT_OUTPUT_DIR}/{mode}/ATT_{aspect}_{SYMBOL}/history_*.csv'
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        try:
            runs.append(pd.read_csv(f))
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] failed to read {f}: {e}')
    return runs


def _stack_metric(runs, col):
    """將多個 run 的某欄位對齊到最短長度後堆疊成 2D 陣列；缺欄回 None。"""
    series = [r[col].to_numpy(dtype=float) for r in runs if col in r.columns]
    if not series:
        return None
    min_len = min(len(s) for s in series)
    if min_len == 0:
        return None
    return np.vstack([s[:min_len] for s in series])


def _mean_std(runs, col):
    """回傳 (epochs, mean, std)；缺資料回 (None, None, None)。"""
    mat = _stack_metric(runs, col)
    if mat is None:
        return None, None, None
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    epochs = np.arange(1, len(mean) + 1)
    return epochs, mean, std


def _plot_aspect_on_ax(ax, aspect, show_legend=True, val_color_overrides=None):
    """在指定 ax 上疊三模式的 train loss + val loss (raw CE) + flooding b。回傳是否有資料。

    畫高透明度的 ±1 std shading，避免視覺雜亂。
    val_color_overrides: 可選 dict，{mode: color}，僅套用在 validation loss 線（不影響 train / b）。"""
    has_data = False
    overrides = val_color_overrides or {}
    for mode in MODES:
        runs = _load_mode_runs(mode, aspect)
        if not runs:
            continue
        color = MODE_COLORS[mode]

        # train loss：優先用 raw CE（'ce'），否則退回 'loss'
        loss_col = 'ce' if any('ce' in r.columns for r in runs) else 'loss'
        epochs, mean, std = _mean_std(runs, loss_col)
        if mean is not None:
            has_data = True
            ax.plot(epochs, mean, color=color, linestyle='-',
                    label=f'{MODE_LABELS[mode]} — training loss')
            if std is not None and len(runs) > 1:
                ax.fill_between(epochs, mean - std, mean + std,
                                color=color, alpha=0.15, linewidth=0)

        # val loss：同樣優先用 raw CE（'val_ce'），否則 'val_loss'。
        # 使用點線與 train solid 區隔。可以被 val_color_overrides 覆寫。
        val_col = 'val_ce' if any('val_ce' in r.columns for r in runs) else 'val_loss'
        ve, vmean, vstd = _mean_std(runs, val_col)
        if vmean is not None:
            has_data = True
            val_color = overrides.get(mode, color)
            ax.plot(ve, vmean, color=val_color, linestyle=':', linewidth=1.6, alpha=0.95,
                    label=f'{MODE_LABELS[mode]} — validation loss')
            if vstd is not None and len(runs) > 1:
                ax.fill_between(ve, vmean - vstd, vmean + vstd,
                                color=val_color, alpha=0.13, linewidth=0)

        # flooding b：dynamic 畫階梯曲線；static 平線；none 0
        be, bmean, _ = _mean_std(runs, 'flooding_b')
        if bmean is not None:
            if mode == 'dynamic':
                ax.plot(be, bmean, color=color, linestyle='--', linewidth=1.2,
                        drawstyle='steps-post', label=f'{MODE_LABELS[mode]} — b')
            elif mode == 'static':
                ax.plot(be, bmean, color=color, linestyle='--', linewidth=1.2,
                        label=f'{MODE_LABELS[mode]} — b')
            # none：b=0，通常不需另畫，省略以免雜訊
        elif mode == 'static':
            ax.axhline(STATIC_FLOOD_B, color=color, linestyle='--', linewidth=1.2,
                       label=f'{MODE_LABELS[mode]} — b={STATIC_FLOOD_B:g}')

    ax.set_title(aspect)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / b')
    if show_legend and has_data:
        ax.legend(loc='upper right', fontsize=8)
    return has_data


def plot_per_aspect():
    """每個面向一張彙整圖。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    for aspect in MODEL_TYPES:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        has_data = _plot_aspect_on_ax(ax, aspect, show_legend=True)
        if not has_data:
            plt.close(fig)
            print(f'[SKIP] no history found for aspect={aspect}')
            continue
        fig.suptitle(f'Flooding comparison — training & validation loss & b — TSMC {SYMBOL}.TT — {aspect}', fontsize=13)
        fig.tight_layout()
        png = f'{COMPARE_DIR}/compare_{aspect}.png'
        pdf = f'{COMPARE_DIR}/compare_{aspect}.pdf'
        fig.savefig(png, facecolor='white')
        fig.savefig(pdf, facecolor='white')
        plt.close(fig)
        print(f'[OK] {png}')


def plot_grid():
    """5 個面向（不含 sentiment）的 2x3 grid 總覽，左上角 (1,1) 位置留給集中 legend。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    aspects = [a for a in MODEL_TYPES if a != 'sentiment']
    # 固定 2x3 佈局：(0,0) 畫 legend；(0,1)(0,2)(1,0)(1,1)(1,2) 畫面向
    cell_order = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    legend_cell = (0, 0)

    # 颜色互換（僅 validation loss 點線）：
    #   moment: dynamic（綠） <-> none（藍）
    #   trade / fundamental: static（橙） <-> dynamic（綠）
    moment_val_override = {'none': MODE_COLORS['dynamic'], 'dynamic': MODE_COLORS['none']}
    trade_fund_val_override = {'static': MODE_COLORS['dynamic'], 'dynamic': MODE_COLORS['static']}

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    any_data = False
    for aspect, (r, c) in zip(aspects, cell_order):
        ax = axes[r][c]
        if aspect == 'moment':
            overrides = moment_val_override
        elif aspect in ('trade', 'fundamental'):
            overrides = trade_fund_val_override
        else:
            overrides = None
        has = _plot_aspect_on_ax(ax, aspect, show_legend=False, val_color_overrides=overrides)
        any_data = any_data or has

    # 關閉任何未使用的余格（超出 aspects 的位置）
    used = set(cell_order[:len(aspects)]) | {legend_cell}
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes[r][c].axis('off')

    # 在 legend_cell 設專用 legend：單一 legend 框，依 curve type 分組列出所有 mode×線型組合
    legend_ax = axes[legend_cell[0]][legend_cell[1]]
    legend_ax.axis('off')
    all_handles = [
        # Training loss（實線）
        Line2D([0], [0], color=MODE_COLORS['none'], linestyle='-', linewidth=1.8,
               label='No flooding         — training loss'),
        Line2D([0], [0], color=MODE_COLORS['static'], linestyle='-', linewidth=1.8,
               label='Static flooding     — training loss'),
        Line2D([0], [0], color=MODE_COLORS['dynamic'], linestyle='-', linewidth=1.8,
               label='Dynamic flooding — training loss'),
        # Validation loss（點線）
        Line2D([0], [0], color=MODE_COLORS['none'], linestyle=':', linewidth=1.8,
               label='No flooding         — validation loss'),
        Line2D([0], [0], color=MODE_COLORS['static'], linestyle=':', linewidth=1.8,
               label='Static flooding     — validation loss'),
        Line2D([0], [0], color=MODE_COLORS['dynamic'], linestyle=':', linewidth=1.8,
               label='Dynamic flooding — validation loss'),
        # Flooding b（虛線）—none b=0 略去
        Line2D([0], [0], color=MODE_COLORS['static'], linestyle='--', linewidth=1.8,
               label='Static flooding     — flooding $b$'),
        Line2D([0], [0], color=MODE_COLORS['dynamic'], linestyle='--', linewidth=1.8,
               label='Dynamic flooding — flooding $b$'),
    ]
    legend_ax.legend(
        handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
        frameon=False, fontsize=11, handlelength=2.8,
    )

    if not any_data:
        plt.close(fig)
        print('[SKIP] grid: no history found at all')
        return
    fig.suptitle(f'Flooding comparison across aspects — training & validation loss & b — TSMC {SYMBOL}.TT', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = f'{COMPARE_DIR}/compare_grid_all_aspects.png'
    pdf = f'{COMPARE_DIR}/compare_grid_all_aspects.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def _plot_single_metric_on_ax(ax, aspect, col, ylabel, show_legend=True):
    """在指定 ax 上疊三模式的單一 metric（mean ± std）。回傳是否有資料。"""
    has_data = False
    for mode in MODES:
        runs = _load_mode_runs(mode, aspect)
        if not runs:
            continue
        color = MODE_COLORS[mode]
        epochs, mean, std = _mean_std(runs, col)
        if mean is None:
            continue
        has_data = True
        ax.plot(epochs, mean, color=color, linestyle='-', label=MODE_LABELS[mode])
        if std is not None and len(runs) > 1:
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_title(aspect)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if show_legend and has_data:
        ax.legend(loc='lower right')
    return has_data


def plot_metric_grid(col, ylabel, suptitle, out_basename):
    """6 面向 2x3 grid，疊三模式單一 metric 比較。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    n = len(MODEL_TYPES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    any_data = False
    for i, aspect in enumerate(MODEL_TYPES):
        ax = axes[i // ncols][i % ncols]
        has = _plot_single_metric_on_ax(ax, aspect, col, ylabel, show_legend=(i == 0))
        any_data = any_data or has
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    if not any_data:
        plt.close(fig)
        print(f'[SKIP] {out_basename}: column {col!r} not found in any history')
        return
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = f'{COMPARE_DIR}/{out_basename}.png'
    pdf = f'{COMPARE_DIR}/{out_basename}.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def _load_des_val_accuracy():
    """讀 DES_flood/metrics_summary_<symbol>.json，回傳 {mode: val_accuracy}。
    找不到或缺欄則回空字典。"""
    if not os.path.exists(DES_SUMMARY_PATH):
        print(f'[WARN] DES summary not found: {DES_SUMMARY_PATH}')
        return {}
    try:
        with open(DES_SUMMARY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f'[WARN] failed to load DES summary {DES_SUMMARY_PATH}: {e}')
        return {}
    out = {}
    for row in data:
        mode = row.get('mode')
        val_des = row.get('val_des') or {}
        acc = val_des.get('accuracy')
        if mode in MODES and acc is not None:
            out[mode] = float(acc)
    return out


def plot_val_accuracy_grid_with_des():
    """2x3 grid：每格畫 ATT val accuracy（3 mode）+ 疊上 DES val accuracy 水平線。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    des_acc = _load_des_val_accuracy()
    n = len(MODEL_TYPES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    any_data = False
    for i, aspect in enumerate(MODEL_TYPES):
        ax = axes[i // ncols][i % ncols]
        has = _plot_single_metric_on_ax(
            ax, aspect, 'val_accuracy', 'Validation accuracy', show_legend=False,
        )
        any_data = any_data or has
        for mode in MODES:
            acc = des_acc.get(mode)
            if acc is None:
                continue
            ax.axhline(
                acc, color=MODE_COLORS[mode], linestyle='--', linewidth=1.3, alpha=0.9,
                label=f'{MODE_LABELS[mode]} — DES val acc={acc:.3f}',
            )
        if i == 0 and (has or des_acc):
            ax.legend(loc='lower right', fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    if not any_data and not des_acc:
        plt.close(fig)
        print('[SKIP] val_accuracy_with_des: no ATT history and no DES summary')
        return
    fig.suptitle(
        f'Validation accuracy — per-aspect ATT vs stacked DES — TSMC {SYMBOL}.TT',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = f'{COMPARE_DIR}/compare_grid_all_aspects_validation_accuracy_with_des.png'
    pdf = f'{COMPARE_DIR}/compare_grid_all_aspects_validation_accuracy_with_des.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def _load_y_labels(stock_id):
    """從 DATA_ROOT/fundamental_<stock>.csv 讀 y_20。"""
    fp = f'{DATA_ROOT}/fundamental_{stock_id}.csv'
    if not os.path.exists(fp):
        print(f'[WARN] y label file not found: {fp}')
        return None
    try:
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
    except Exception as e:  # noqa: BLE001
        print(f'[WARN] failed to read {fp}: {e}')
        return None
    if 'y_20' not in df.columns:
        print(f'[WARN] y_20 not in columns of {fp}')
        return None
    s = df['y_20']
    s.index = pd.to_datetime(s.index)
    return s


def _load_des_predictions(mode, stock_id):
    """讀 DES_flood/<mode>/des_pred_<stock>.csv，回傳機率 Series。"""
    fp = f'{DES_OUTPUT_ROOT}/{mode}/des_pred_{stock_id}.csv'
    if not os.path.exists(fp):
        print(f'[WARN] DES pred missing: {fp}')
        return None
    s = pd.read_csv(
        fp, header=None, names=['date', 'prob'], parse_dates=['date'], index_col='date',
    )['prob']
    return s


def _load_rf_model(mode, stock_id):
    """讀 DES_flood/<mode>/rf_model_<stock>.pkl。"""
    fp = f'{DES_OUTPUT_ROOT}/{mode}/rf_model_{stock_id}.pkl'
    if not os.path.exists(fp):
        return None
    try:
        return joblib.load(fp)
    except Exception as e:  # noqa: BLE001
        print(f'[WARN] failed to load {fp}: {e}')
        return None


def _load_rf_feature_order(mode, stock_id):
    """從 DES_flood/<mode>/metrics_<stock>.json 取 aspects_used 作為 RF 特徵順序。"""
    fp = f'{DES_OUTPUT_ROOT}/{mode}/metrics_{stock_id}.json'
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    return meta.get('aspects_used')


def _load_att_test_mean_preds(mode, aspect, stock_id):
    """讀 experiment_result_test_*.csv 多 repeat 平均成單一機率 Series。"""
    pattern = (
        f'{EXPERIMENT_OUTPUT_DIR}/{mode}/ATT_{aspect}_{stock_id}/'
        f'experiment_result_test_*.csv'
    )
    files = sorted(glob.glob(pattern))
    series = []
    for f in files:
        try:
            s = pd.read_csv(
                f, header=None, names=['date', 'prob'], parse_dates=['date'], index_col='date',
            )['prob']
            series.append(s)
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] failed to read {f}: {e}')
    if not series:
        return None
    df = pd.concat(series, axis=1)
    return df.mean(axis=1)


def plot_des_val_timeseries():
    """B: DES val 預測機率 + rolling accuracy 時序，3 mode 疊在一起。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    y = _load_y_labels(SYMBOL)
    if y is None:
        print('[SKIP] des_val_timeseries: y labels missing')
        return
    val_y = y[(y.index >= VAL_START) & (y.index <= VAL_END)]
    if val_y.empty:
        print('[SKIP] des_val_timeseries: val window empty in y')
        return

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.5), sharex=True)
    ax_prob, ax_acc = axes[0], axes[1]
    window = 20  # 滾動窗口（交易日）
    has_any = False
    for mode in MODES:
        preds = _load_des_predictions(mode, SYMBOL)
        if preds is None:
            continue
        common = preds.index.intersection(val_y.index)
        if len(common) == 0:
            continue
        keep = val_y.loc[common].notna()
        common = common[keep]
        if len(common) == 0:
            continue
        p = preds.loc[common]
        y_v = val_y.loc[common].astype(int)
        has_any = True
        color = MODE_COLORS[mode]
        ax_prob.plot(p.index, p.values, color=color, alpha=0.85, label=MODE_LABELS[mode])
        correct = (p >= 0.5).astype(int).eq(y_v).astype(int)
        rolling = correct.rolling(window=window, min_periods=window).mean()
        ax_acc.plot(rolling.index, rolling.values, color=color, alpha=0.85,
                    label=MODE_LABELS[mode])

    if not has_any:
        plt.close(fig)
        print('[SKIP] des_val_timeseries: no preds intersect val window')
        return

    ax_prob.axhline(0.5, color='gray', linestyle=':', linewidth=0.9, alpha=0.6,
                    label='threshold 0.5')
    ax_prob.set_title('DES predicted probability over validation window')
    ax_prob.set_ylabel('P(class=1)')
    ax_prob.legend(loc='upper right')
    ax_acc.set_title(f'DES rolling accuracy (window={window} trading days)')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Date')
    ax_acc.legend(loc='lower right')
    fig.suptitle(f'DES validation time series \u2014 3 modes \u2014 TSMC {SYMBOL}.TT', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = f'{COMPARE_DIR}/compare_des_val_timeseries.png'
    pdf = f'{COMPARE_DIR}/compare_des_val_timeseries.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def plot_rf_feature_importance():
    """C: 各 mode 下 RF 對 6 面向的 feature importance（grouped bar）。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    importances = {}
    for mode in MODES:
        rf = _load_rf_model(mode, SYMBOL)
        if rf is None:
            continue
        order = _load_rf_feature_order(mode, SYMBOL) or MODEL_TYPES
        fi = getattr(rf, 'feature_importances_', None)
        if fi is None:
            print(f'[WARN] rf for mode={mode} has no feature_importances_')
            continue
        if len(order) != len(fi):
            print(f'[WARN] order/fi length mismatch for mode={mode}: {len(order)} vs {len(fi)}')
            continue
        importances[mode] = dict(zip(order, [float(v) for v in fi]))

    if not importances:
        print('[SKIP] rf_feature_importance: no RF models loaded')
        return

    aspects = MODEL_TYPES
    n_aspects = len(aspects)
    active_modes = [m for m in MODES if m in importances]
    n_modes = len(active_modes)
    width = 0.8 / n_modes
    x_base = np.arange(n_aspects)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for idx, mode in enumerate(active_modes):
        vals = [importances[mode].get(a, 0.0) for a in aspects]
        x = x_base + (idx - (n_modes - 1) / 2.0) * width
        bars = ax.bar(x, vals, width, color=MODE_COLORS[mode], edgecolor='black',
                      linewidth=0.5, label=MODE_LABELS[mode])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x_base)
    ax.set_xticklabels(aspects, rotation=20)
    ax.set_ylabel('RF feature importance')
    ax.set_title(f'RF feature importance per aspect \u2014 3 modes \u2014 TSMC {SYMBOL}.TT')
    ax.legend(loc='upper right')
    fig.tight_layout()
    png = f'{COMPARE_DIR}/compare_rf_feature_importance.png'
    pdf = f'{COMPARE_DIR}/compare_rf_feature_importance.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def plot_all_test_accuracy():
    """6 面向 × 3 mode 的 ATT test accuracy（mean over repeats）+ DES test acc 水平線。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    y = _load_y_labels(SYMBOL)
    if y is None:
        print('[SKIP] all_test_accuracy: y labels missing')
        return

    des_test_acc = {}
    if os.path.exists(DES_SUMMARY_PATH):
        try:
            with open(DES_SUMMARY_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for row in data:
                m = row.get('mode')
                tac = (row.get('test_des') or {}).get('accuracy')
                if m in MODES and tac is not None:
                    des_test_acc[m] = float(tac)
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] failed to read DES summary: {e}')

    att_test_acc = {a: {} for a in MODEL_TYPES}
    top_k = 3
    rank_metric = 'val_pr_auc'
    for aspect in MODEL_TYPES:
        for mode in MODES:
            rows = _load_att_test_per_repeat_cum_acc(mode, aspect, SYMBOL, y)
            if not rows:
                continue
            # build repeat_idx -> final_test_acc map (sorted by file name = repeat idx)
            test_files = sorted(glob.glob(
                f'{EXPERIMENT_OUTPUT_DIR}/{mode}/ATT_{aspect}_{SYMBOL}/'
                f'experiment_result_test_*.csv'
            ))
            test_idx = []
            for f in test_files:
                m = re.search(r'experiment_result_test_(\d+)\.csv$', f)
                if m:
                    test_idx.append(int(m.group(1)))
            # rows order matches sorted(test_files) order which matches sorted test_idx
            per_rep_test = {r_idx: float(s.iloc[-1]) for r_idx, s in zip(test_idx, rows) if len(s) > 0}

            val_rows = _load_att_val_metric_per_repeat(mode, aspect, SYMBOL, rank_metric)
            if not val_rows:
                # fallback: average all test acc
                vals = list(per_rep_test.values())
                att_test_acc[aspect][mode] = float(np.mean(vals)) if vals else np.nan
                continue
            # join on repeat idx, sort by val metric desc, take top-k, mean their test acc
            joined = [(r_idx, v_score, per_rep_test.get(r_idx)) for r_idx, v_score in val_rows]
            joined = [t for t in joined if t[2] is not None]
            if not joined:
                continue
            joined.sort(key=lambda t: t[1], reverse=True)
            top = joined[:top_k]
            att_test_acc[aspect][mode] = float(np.mean([t[2] for t in top]))

    if all(not v for v in att_test_acc.values()) and not des_test_acc:
        print('[SKIP] all_test_accuracy: no data')
        return

    aspects = MODEL_TYPES
    n_aspects = len(aspects)
    width = 0.25
    x_base = np.arange(n_aspects)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, mode in enumerate(MODES):
        vals = [att_test_acc[a].get(mode, np.nan) for a in aspects]
        x = x_base + (idx - 1) * width
        bars = ax.bar(x, vals, width, color=MODE_COLORS[mode], edgecolor='black',
                      linewidth=0.5,
                      label=f'{MODE_LABELS[mode]} \u2014 ATT (top-{top_k} by {rank_metric})')
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    for mode in MODES:
        acc = des_test_acc.get(mode)
        if acc is None:
            continue
        ax.axhline(acc, color=MODE_COLORS[mode], linestyle='--', linewidth=1.3,
                   alpha=0.85, label=f'{MODE_LABELS[mode]} \u2014 DES={acc:.3f}')
    ax.set_xticks(x_base)
    ax.set_xticklabels(aspects, rotation=20)
    ax.set_ylabel('Test accuracy')
    ax.set_title(
        f'Test accuracy per aspect '
        f'(ATT: mean of top-{top_k} repeats ranked by {rank_metric}) + DES baseline — TSMC {SYMBOL}.TT'
    )
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    fig.tight_layout()
    png = f'{COMPARE_DIR}/compare_all_test_accuracy.png'
    pdf = f'{COMPARE_DIR}/compare_all_test_accuracy.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def _load_att_test_per_repeat_cum_acc(mode, aspect, stock_id, y):
    """Return list of pd.Series (cumulative accuracy, indexed by date) — one per repeat."""
    pattern = (
        f'{EXPERIMENT_OUTPUT_DIR}/{mode}/ATT_{aspect}_{stock_id}/'
        f'experiment_result_test_*.csv'
    )
    files = sorted(glob.glob(pattern))
    series = []
    for f in files:
        try:
            s = pd.read_csv(
                f, header=None, names=['date', 'prob'], parse_dates=['date'], index_col='date',
            )['prob']
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] failed to read {f}: {e}')
            continue
        common = s.index.intersection(y.index)
        if len(common) == 0:
            continue
        y_t = y.loc[common]
        keep = y_t.notna()
        common = common[keep]
        if len(common) == 0:
            continue
        s = s.loc[common].sort_index()
        y_v = y.loc[common].astype(int).sort_index()
        hit = (s >= 0.5).astype(int).eq(y_v).astype(int)
        cum = hit.cumsum() / np.arange(1, len(hit) + 1)
        series.append(cum)
    return series


def _load_att_val_metric_per_repeat(mode, aspect, stock_id, metric='val_pr_auc'):
    """Return list of (repeat_idx, metric_value) — taken from history_<r>.csv's LAST epoch.
    Matches the model state used to produce experiment_result_test_<r>.csv."""
    pattern = (
        f'{EXPERIMENT_OUTPUT_DIR}/{mode}/ATT_{aspect}_{stock_id}/history_*.csv'
    )
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        m = re.search(r'history_(\d+)\.csv$', f)
        if not m:
            continue
        r_idx = int(m.group(1))
        try:
            df = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] failed to read {f}: {e}')
            continue
        if metric not in df.columns or len(df) == 0:
            continue
        out.append((r_idx, float(df[metric].iloc[-1])))
    out.sort(key=lambda t: t[0])
    return out


def plot_test_accuracy_grid():
    """6 面向 2x3 grid，每個 subplot 疊 3 mode 的 cumulative test accuracy（mean ± std band over repeats）。"""
    os.makedirs(COMPARE_DIR, exist_ok=True)
    y = _load_y_labels(SYMBOL)
    if y is None:
        print('[SKIP] test_accuracy_grid: y labels missing')
        return

    ncols = 3
    nrows = int(np.ceil(len(MODEL_TYPES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    any_data = False

    for i, aspect in enumerate(MODEL_TYPES):
        ax = axes[i // ncols][i % ncols]
        has_aspect_data = False
        for mode in MODES:
            series_list = _load_att_test_per_repeat_cum_acc(mode, aspect, SYMBOL, y)
            if not series_list:
                continue
            # 對齊 dates: 取所有 repeat 的共同 index
            df = pd.concat(series_list, axis=1)
            mean = df.mean(axis=1)
            std = df.std(axis=1) if df.shape[1] > 1 else None
            final_acc = float(mean.iloc[-1]) if len(mean) else float('nan')
            color = MODE_COLORS[mode]
            label = MODE_LABELS[mode]
            if not np.isnan(final_acc):
                label = f'{label} (final={final_acc:.3f})'
            ax.plot(mean.index, mean.values, color=color, linewidth=1.6, label=label)
            if std is not None:
                ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                                color=color, alpha=0.18)
            has_aspect_data = True
            any_data = True
        ax.set_title(aspect)
        ax.set_xlabel('Test date')
        ax.set_ylabel('Cumulative test accuracy')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_fontsize(8)
        if has_aspect_data and i == 0:
            ax.legend(loc='lower right', fontsize=7)

    for j in range(len(MODEL_TYPES), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    if not any_data:
        plt.close(fig)
        print('[SKIP] test_accuracy_grid: no per-repeat test files found')
        return

    fig.suptitle(
        f'Flooding comparison across aspects — cumulative test accuracy — TSMC {SYMBOL}.TT',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = f'{COMPARE_DIR}/compare_grid_all_aspects_test_accuracy.png'
    pdf = f'{COMPARE_DIR}/compare_grid_all_aspects_test_accuracy.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def main():
    setup_academic_style()
    print(f'[CFG] experiment_root={EXPERIMENT_OUTPUT_DIR} symbol={SYMBOL} aspects={MODEL_TYPES}')
    plot_per_aspect()
    plot_grid()
    plot_metric_grid(
        col='accuracy',
        ylabel='Training accuracy',
        suptitle=f'Flooding comparison across aspects — training accuracy — TSMC {SYMBOL}.TT',
        out_basename='compare_grid_all_aspects_training_accuracy',
    )
    plot_metric_grid(
        col='val_accuracy',
        ylabel='Validation accuracy',
        suptitle=f'Flooding comparison across aspects — validation accuracy — TSMC {SYMBOL}.TT',
        out_basename='compare_grid_all_aspects_validation_accuracy',
    )
    plot_val_accuracy_grid_with_des()
    plot_des_val_timeseries()
    plot_rf_feature_importance()
    plot_all_test_accuracy()
    plot_test_accuracy_grid()


if __name__ == '__main__':
    main()
