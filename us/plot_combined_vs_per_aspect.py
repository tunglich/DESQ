"""Combined5 vs 5 per-aspect 比較圖。

讀取 ATT+Dflooding_floodexp.py 與 ATT+Dflooding_combined.py 寫出的
history_*.csv，比較 combined5 與 5 個 per-aspect 模型在三種 flooding mode 下
的 val_pr_auc。

輸入（預設）：
    D:/experiment_flood/<mode>/ATT_<aspect>_<symbol>/history_<r>.csv
    aspect ∈ {combined5, fundamental, trade, moment, tech_trend, macro}
    mode   ∈ {none, static, dynamic}

輸出：
    D:/evaluation_plot/_compare/combined_vs_per_aspect.{png,pdf}

可用環境變數：
    EXPERIMENT_OUTPUT_DIR (預設 D:/experiment_flood)
    EVAL_PLOT_DIR         (預設 D:/evaluation_plot)
    STOCK_IDS             (預設 2330；取第一個)
    METRIC                (預設 val_pr_auc；可選 val_accuracy / val_f1 / val_auc)
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def platform_path(path_str):
    if os.name != 'nt' and len(path_str) >= 2 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path_str


EXPERIMENT_OUTPUT_DIR = platform_path(os.getenv('EXPERIMENT_OUTPUT_DIR', 'D:/experiment_flood'))
EVAL_PLOT_DIR = platform_path(os.getenv('EVAL_PLOT_DIR', 'D:/evaluation_plot'))
SYMBOL = [x.strip() for x in os.getenv('STOCK_IDS', '2330').split(',') if x.strip()][0]

ASPECTS = ['combined5', 'fundamental', 'trade', 'moment', 'tech_trend', 'macro']
ASPECT_LABELS = {
    'combined5': 'Combined-5',
    'fundamental': 'Fundamental',
    'trade': 'Trade',
    'moment': 'Moment',
    'tech_trend': 'Tech-trend',
    'macro': 'Macro',
}

# 每個 aspect 的 val_pos_base_rate（從 DIAG 行的 dynamic mode 平均算出）。
# 這些值用於計算 PR-AUC lift = val_pr_auc - base_rate。
ASPECT_BASE_RATE = {
    'combined5': 0.807,
    'fundamental': 0.641,
    'trade': 0.754,
    'moment': 0.642,
    'tech_trend': 0.642,
    'macro': 0.807,
}

MODES = ['none', 'static', 'dynamic']
MODE_COLORS = {'none': '#4C72B0', 'static': '#DD8452', 'dynamic': '#55A868'}
MODE_LABELS = {'none': 'No flooding', 'static': 'Static flooding', 'dynamic': 'Dynamic flooding'}

# 多 metric 設定：raw col、是否需減 base rate、label、y range、是否 higher-better
METRICS = {
    'val_pr_auc': {
        'col': 'val_pr_auc',
        'lift': False,
        'label': 'val PR-AUC',
        'title_suffix': 'raw PR-AUC (base-rate sensitive)',
    },
    'val_auc': {
        'col': 'val_auc',
        'lift': False,
        'label': 'val ROC-AUC',
        'title_suffix': 'ROC-AUC (base-rate invariant)',
    },
    'val_pr_auc_lift': {
        'col': 'val_pr_auc',
        'lift': True,
        'label': 'val PR-AUC lift  (= PR-AUC − base rate)',
        'title_suffix': 'PR-AUC lift above base rate',
    },
}
DEFAULT_METRICS = ['val_pr_auc', 'val_auc', 'val_pr_auc_lift']

COMPARE_DIR = f'{EVAL_PLOT_DIR}/_compare'


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.6,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_runs(mode, aspect):
    pattern = f'{EXPERIMENT_OUTPUT_DIR}/{mode}/ATT_{aspect}_{SYMBOL}/history_*.csv'
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        try:
            runs.append(pd.read_csv(f))
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] failed to read {f}: {e}')
    return runs


def _best_metric_per_run(runs, col, lift_aspect=None):
    """每個 run 取該 metric 在 epoch 維度的最大值；若 lift_aspect 給定，先減 base rate。"""
    base = ASPECT_BASE_RATE.get(lift_aspect, 0.0) if lift_aspect else 0.0
    vals = []
    for r in runs:
        if col in r.columns and len(r[col]) > 0:
            arr = r[col].to_numpy(dtype=float) - base
            vals.append(float(np.nanmax(arr)))
    return np.array(vals) if vals else np.array([])


def _mean_std_curve(runs, col, lift_aspect=None):
    base = ASPECT_BASE_RATE.get(lift_aspect, 0.0) if lift_aspect else 0.0
    series = [r[col].to_numpy(dtype=float) - base for r in runs if col in r.columns]
    if not series:
        return None, None, None
    min_len = min(len(s) for s in series)
    if min_len == 0:
        return None, None, None
    mat = np.vstack([s[:min_len] for s in series])
    epochs = np.arange(1, min_len + 1)
    return epochs, mat.mean(axis=0), mat.std(axis=0)


def collect(metric_key):
    """收集所有 (aspect, mode) 的資料；metric_key in METRICS。"""
    spec = METRICS[metric_key]
    col = spec['col']
    use_lift = spec['lift']
    data = {}  # (aspect, mode) -> {'best': np.array, 'curve': (epochs, mean, std), 'n': int}
    for aspect in ASPECTS:
        lift_aspect = aspect if use_lift else None
        for mode in MODES:
            runs = _load_runs(mode, aspect)
            if not runs:
                continue
            best = _best_metric_per_run(runs, col, lift_aspect=lift_aspect)
            curve = _mean_std_curve(runs, col, lift_aspect=lift_aspect)
            data[(aspect, mode)] = {'best': best, 'curve': curve, 'n': len(runs)}
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _bar_panel(ax, data, metric_key):
    """Grouped bar chart：x = aspect, group = mode, y = mean best metric。"""
    spec = METRICS[metric_key]
    n_aspect = len(ASPECTS)
    n_mode = len(MODES)
    x = np.arange(n_aspect)
    width = 0.8 / n_mode

    for j, mode in enumerate(MODES):
        means, stds = [], []
        for aspect in ASPECTS:
            arr = data.get((aspect, mode), {}).get('best', np.array([]))
            means.append(arr.mean() if len(arr) else np.nan)
            stds.append(arr.std() if len(arr) else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        offset = (j - (n_mode - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3,
               color=MODE_COLORS[mode], alpha=0.85,
               edgecolor='black', linewidth=0.5,
               error_kw={'elinewidth': 0.8, 'ecolor': '#333333'},
               label=MODE_LABELS[mode])

    ax.set_xticks(x)
    ax.set_xticklabels([ASPECT_LABELS[a] for a in ASPECTS], rotation=0)
    for tl in ax.get_xticklabels():
        if tl.get_text() == ASPECT_LABELS['combined5']:
            tl.set_fontweight('bold')
    ax.set_ylabel(f'Best {spec["label"]}  (mean ± std)')
    ax.set_title(f'Best {spec["label"]} across aspects × flooding modes')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    # ROC-AUC chance line / lift zero line
    if metric_key == 'val_auc':
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.6,
                   label='Chance (0.5)')
    if metric_key == 'val_pr_auc_lift':
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=0.8, alpha=0.6,
                   label='No lift (0)')
    ax.legend(loc='upper right', ncol=4, frameon=False)


def _curve_panel(ax, data, mode, metric_key):
    """單一 mode 下 6 aspects 的 val metric 曲線。"""
    spec = METRICS[metric_key]
    cmap = plt.get_cmap('tab10')
    aspect_colors = {
        'combined5': '#000000',
        'fundamental': cmap(0),
        'trade': cmap(1),
        'moment': cmap(2),
        'tech_trend': cmap(3),
        'macro': cmap(4),
    }
    for aspect in ASPECTS:
        d = data.get((aspect, mode))
        if not d or d['curve'][0] is None:
            continue
        epochs, mean, std = d['curve']
        is_combined = (aspect == 'combined5')
        color = aspect_colors[aspect]
        lw = 2.2 if is_combined else 1.4
        alpha_line = 1.0 if is_combined else 0.85
        ax.plot(epochs, mean, color=color, linewidth=lw, alpha=alpha_line,
                label=ASPECT_LABELS[aspect],
                zorder=5 if is_combined else 2)
        if std is not None and len(d['best']) > 1:
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=0.10 if is_combined else 0.06, linewidth=0)
    if metric_key == 'val_auc':
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    if metric_key == 'val_pr_auc_lift':
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(f'{MODE_LABELS[mode]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(spec['label'])


def plot_for_metric(metric_key, data):
    """單一 metric 的主圖：頂列 bar chart，下列 3 panel 曲線。"""
    if not data:
        print(f'[FAIL] no data for {metric_key}; skip')
        return
    spec = METRICS[metric_key]
    os.makedirs(COMPARE_DIR, exist_ok=True)

    fig = plt.figure(figsize=(15.6, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.22)

    ax_bar = fig.add_subplot(gs[0, :])
    _bar_panel(ax_bar, data, metric_key)

    axes_curve = [fig.add_subplot(gs[1, j]) for j in range(3)]
    for ax, mode in zip(axes_curve, MODES):
        _curve_panel(ax, data, mode, metric_key)

    aspect_handles = [
        Line2D([0], [0], color='#000000', linewidth=2.2, label=ASPECT_LABELS['combined5']),
    ]
    cmap = plt.get_cmap('tab10')
    for i, aspect in enumerate(['fundamental', 'trade', 'moment', 'tech_trend', 'macro']):
        aspect_handles.append(
            Line2D([0], [0], color=cmap(i), linewidth=1.4, label=ASPECT_LABELS[aspect])
        )
    axes_curve[-1].legend(handles=aspect_handles, loc='lower right',
                          ncol=1, frameon=False, fontsize=9)

    fig.suptitle(
        f'Combined-5 vs per-aspect ATT — TSMC {SYMBOL}.TT — {spec["title_suffix"]}',
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png = f'{COMPARE_DIR}/combined_vs_per_aspect_{metric_key}.png'
    pdf = f'{COMPARE_DIR}/combined_vs_per_aspect_{metric_key}.pdf'
    fig.savefig(png, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(f'[OK] {png}')


def print_summary_table(metric_key, data):
    """印出每個 (aspect, mode) 的 mean ± std 表，方便人類核對。"""
    spec = METRICS[metric_key]
    print()
    print(f'=== Summary: best {spec["label"]} (mean ± std over runs) ===')
    header = f'{"aspect":<13s} | ' + ' | '.join(f'{MODE_LABELS[m]:^22s}' for m in MODES)
    print(header)
    print('-' * len(header))
    for aspect in ASPECTS:
        cells = []
        for mode in MODES:
            d = data.get((aspect, mode))
            if d and len(d['best']) > 0:
                cells.append(f'{d["best"].mean():+.4f} ± {d["best"].std():.4f} (n={d["n"]:2d})')
            else:
                cells.append('n/a')
        print(f'{aspect:<13s} | ' + ' | '.join(f'{c:^22s}' for c in cells))


def main():
    setup_style()
    metrics = os.getenv('METRICS', ','.join(DEFAULT_METRICS)).split(',')
    metrics = [m.strip() for m in metrics if m.strip() in METRICS]
    if not metrics:
        metrics = DEFAULT_METRICS
    for metric_key in metrics:
        print(f'\n========== metric = {metric_key} ==========')
        data = collect(metric_key)
        print_summary_table(metric_key, data)
        plot_for_metric(metric_key, data)


if __name__ == '__main__':
    main()
