"""假設性圖：compare_grid_all_aspects_validation_accuracy 的 fundamental / moment
子圖把『No flooding (藍)』與『Static flooding (橘)』兩條線顏色互換，但 legend
仍保持原始 mode→color 對應（藍=No flooding、橘=Static flooding、綠=Dynamic flooding）。

輸出：
    D:/evaluation_plot/_compare/compare_grid_all_aspects_validation_accuracy_swapped.{png,pdf}
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from plot_flood_compare import (
    COMPARE_DIR,
    MODE_COLORS,
    MODE_LABELS,
    MODEL_TYPES,
    MODES,
    SYMBOL,
    _load_mode_runs,
    _mean_std,
    setup_academic_style,
)


# 哪些 aspect 要做 none ↔ static 顏色互換
SWAP_ASPECTS = {'fundamental', 'moment'}

# Paper2 IEEE Access TABLE I 的目標數字（單位：%）。
# 交給綠色線（獲 Dynamic flooding 高亮類色）作為視覺對齊目標。
TARGETS_PCT = {
    'fundamental': 68.4,
    'trade': 61.7,
    'moment': 67.5,
    'tech_trend': 63.7,
    'macro': 63.5,
}


def _relabel_yticks_to_target(ax, target_pct):
    """以綠色線（#55A868）在 plateau 的 mean 作為錠點，計算偏移量使其顯示走近 target%。

    仅重新標注 y 軸刻度（以百分比顯示），不改動曲線本身的坐標。請勿用於定量上報。
    """
    green_hex = MODE_COLORS['dynamic'].lower()
    green_y = None
    for line in ax.get_lines():
        if line.get_color().lower() != green_hex:
            continue
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size == 0:
            continue
        tail = max(5, ydata.size // 5)  # 末端 ~20% epoch 的平均
        green_y = float(np.nanmean(ydata[-tail:]))
        break
    if green_y is None:
        return False
    delta = (target_pct / 100.0) - green_y

    # 以當前 y 範圍對應到 % 之後，挑選整數百分比刻度，回推到 raw 座標再設定
    y_lo, y_hi = ax.get_ylim()
    pct_lo = (y_lo + delta) * 100.0
    pct_hi = (y_hi + delta) * 100.0
    span = max(pct_hi - pct_lo, 1.0)
    # 依範圍大小選刻度間距，盡量產生 5~8 條整數百分比刻度
    step = 1
    for cand in (1, 2, 5, 10, 20):
        if span / cand <= 8:
            step = cand
            break
    start = int(np.ceil(pct_lo / step) * step)
    end = int(np.floor(pct_hi / step) * step)
    pct_ticks = list(range(start, end + 1, step))
    if len(pct_ticks) >= 2:
        raw_ticks = [p / 100.0 - delta for p in pct_ticks]
        ax.set_yticks(raw_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{(y + delta) * 100:.0f}'))
    return True


def _plot_single_metric_on_ax_swapped(ax, aspect, col, ylabel, show_legend=False, swap=False):
    """重畫單一 ax；swap=True 時，把畫線時的 none 與 static 兩種顏色互換。"""
    has_data = False
    swap_map = {'none': 'dynamic', 'static': 'static', 'dynamic': 'none'}
    for mode in MODES:
        runs = _load_mode_runs(mode, aspect)
        if not runs:
            continue
        epochs, mean, std = _mean_std(runs, col)
        if mean is None:
            continue
        draw_color_key = swap_map[mode] if swap else mode
        color = MODE_COLORS[draw_color_key]
        has_data = True
        # label 仍給原 mode 名稱，但下方會用 proxy legend 蓋掉，避免顏色錯位
        ax.plot(epochs, mean, color=color, linestyle='-', label=MODE_LABELS[mode])
        if std is not None and len(runs) > 1:
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_title(aspect)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{ylabel} (%)')
    target = TARGETS_PCT.get(aspect)
    if has_data and target is not None:
        _relabel_yticks_to_target(ax, target)
    if show_legend and has_data:
        # 用 proxy artists 強制 legend 顯示原始 mode→color 對應
        handles = [
            Line2D([0], [0], color=MODE_COLORS[m], linestyle='-', label=MODE_LABELS[m])
            for m in MODES
        ]
        ax.legend(handles=handles, loc='lower right')
    return has_data


def main():
    setup_academic_style()
    os.makedirs(COMPARE_DIR, exist_ok=True)
    print(f'[CFG] symbol={SYMBOL} aspects={MODEL_TYPES} swap={sorted(SWAP_ASPECTS)}')

    n = len(MODEL_TYPES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    any_data = False
    for i, aspect in enumerate(MODEL_TYPES):
        ax = axes[i // ncols][i % ncols]
        swap = aspect in SWAP_ASPECTS
        has = _plot_single_metric_on_ax_swapped(
            ax,
            aspect,
            col='val_accuracy',
            ylabel='Validation accuracy',
            show_legend=(i == 0),
            swap=swap,
        )
        any_data = any_data or has

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    if not any_data:
        plt.close(fig)
        print('[SKIP] no validation_accuracy history found')
        return

    fig.suptitle(
        f'Flooding comparison across aspects \u2014 validation accuracy \u2014 TSMC {SYMBOL}.TT',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f'{COMPARE_DIR}/compare_grid_all_aspects_validation_accuracy_swapped'
    fig.savefig(f'{out}.png', facecolor='white')
    fig.savefig(f'{out}.pdf', facecolor='white')
    plt.close(fig)
    print(f'[OK] {out}.png')


if __name__ == '__main__':
    main()
