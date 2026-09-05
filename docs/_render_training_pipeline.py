"""Render the revised-paper five-stage DESQ architecture diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]

OUT_PATH = Path(__file__).parent / "training_pipeline.png"
INK = "#17202a"
BLUE = "#dceaf4"
GREEN = "#deeee6"
GOLD = "#f3e9cf"
GRAY = "#eef0f2"
RED = "#f2dedd"
CODE = "#174f78"
FONT_SCALE = 2.0


def _box(ax, x: float, y: float, width: float, height: float, number: str,
         title: str, body: str, code: str, face: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.25, edgecolor=INK, facecolor=face,
    ))
    ax.text(x + width / 2, y + height * 0.89, number, ha="center", va="center",
            fontsize=8.0 * FONT_SCALE, fontweight="bold", color=INK)
    ax.text(x + width / 2, y + height * 0.77, title, ha="center", va="center",
            fontsize=9.7 * FONT_SCALE, fontweight="bold", color=INK,
            linespacing=0.95)
    ax.text(x + width / 2, y + height * 0.43, body, ha="center", va="center",
            fontsize=8.2 * FONT_SCALE, color=INK, linespacing=1.12)
    ax.text(x + width / 2, y + height * 0.10, code, ha="center", va="center",
            fontsize=7.3 * FONT_SCALE, color=CODE, family="monospace",
            linespacing=1.10)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *,
           dashed: bool = False, label: str | None = None,
           curve: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.25,
        color=INK, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    ))
    if label:
        midpoint_x = (start[0] + end[0]) / 2
        midpoint_y = (start[1] + end[1]) / 2
        ax.text(midpoint_x, midpoint_y + 0.13, label, ha="center", va="center",
                fontsize=7.4 * FONT_SCALE, fontstyle="italic", color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 9.2), dpi=300)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(6.6, 8.87, "Complete DESQ Architecture", ha="center", va="center",
            fontsize=17 * FONT_SCALE, fontweight="bold", color=INK)
    ax.text(6.6, 8.36,
            "Revised paper Figure 4: supervised forecasting, execution, and sealed maintenance",
            ha="center", va="center", fontsize=9.5 * FONT_SCALE, color=INK)

    box_y, box_w, box_h, gap = 4.10, 2.18, 3.50, 0.34
    starts = [0.30 + index * (box_w + gap) for index in range(5)]
    stages = [
        ("1", "Data\nProcessing", "Causal alignment,\nlabels, and frozen\npre-2024 splits",
         "features/ +\nprices/", GRAY),
        ("2", "Feature\nTransform", "130 candidates ->\n78 features\n5 economic groups",
         "fund. | trend\nmomentum | float\nmacro", BLUE),
        ("3", "Specialist\nTraining", "5 causal\nTransformers/stock\nAutoML + Dynamic\nFlooding\nwalk-forward\nvalidation",
         "tw50_flood.py\ntw50_dflood.py", GREEN),
        ("4", "DES\nCombiner", "Local competence\nselection\n20-day directional\nsignal\nKNORA-E, K=30",
         "tw50_des.py", GOLD),
        ("5", "DDQN\nExecution", "DES signal +\n10 K-bars\nposition + running\nP&L\nSkip / Buy / Close",
         "dqn/", RED),
    ]
    for start, stage in zip(starts, stages):
        _box(ax, start, box_y, box_w, box_h, *stage)
    for left, right in zip(starts, starts[1:]):
        _arrow(ax, (left + box_w, box_y + box_h / 2),
               (right, box_y + box_h / 2))

    monitor_x, monitor_y, monitor_w, monitor_h = 0.60, 0.62, 12.00, 2.25
    ax.add_patch(FancyBboxPatch(
        (monitor_x, monitor_y), monitor_w, monitor_h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.25, edgecolor=INK, facecolor="#f7f7f7", linestyle="--",
    ))
    ax.text(monitor_x + 0.28, monitor_y + 1.72,
            "Post-deployment Performance Monitoring\nand Controlled Updates",
            ha="left", va="center", fontsize=10.3 * FONT_SCALE,
            fontweight="bold", color=INK, linespacing=1.0)
    ax.text(monitor_x + 0.28, monitor_y + 1.02,
            "Two consecutive mature windows  |  six alarms\n"
            "threshold -> DES weights -> Level 2/3",
            ha="left", va="center", fontsize=8.6 * FONT_SCALE,
            color=INK, linespacing=1.12)
    ax.text(monitor_x + 0.28, monitor_y + 0.31,
            "monitoring/  |  sealed validation\n"
            "mature labels only  |  immutable research memory",
            ha="left", va="center", fontsize=7.7 * FONT_SCALE,
            color=CODE, family="monospace", linespacing=1.12)
    ax.text(monitor_x + monitor_w - 0.25, monitor_y + 1.72,
            "DISABLED IN\nREPORTED HOLDOUT", ha="right", va="center",
            fontsize=8.1 * FONT_SCALE, fontweight="bold", color="#8a2f2b",
            linespacing=1.0)

    _arrow(ax, (starts[4] + box_w / 2, box_y),
           (monitor_x + monitor_w * 0.78, monitor_y + monitor_h),
           dashed=True, label="prediction, trading, stability, and drift diagnostics")
    _arrow(ax, (monitor_x + monitor_w * 0.20, monitor_y + monitor_h),
           (starts[2] + box_w / 2, box_y), dashed=True,
           label="configuration-limited maintenance", curve=0.08)

    ax.text(6.6, 0.20,
            "Solid arrows: reported fixed offline pipeline     "
            "Dashed arrows: Appendix F operational extension",
            ha="center", va="center", fontsize=8.0 * FONT_SCALE,
            fontstyle="italic", color=INK)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
