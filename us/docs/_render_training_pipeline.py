"""Render us/docs/training_pipeline.png — DESQ US-extension training pipeline.

Same IEEE-style 7-stage layout as the TW-50 pipeline, but with US-specific
scripts and 4 feature aspects (fundamental / moment / tech_trend / macro).
Stage 7 additionally applies a CUSUM directional filter before the portfolio
backtest.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = Path(__file__).parent / "training_pipeline.png"

BG = "#ffffff"
LINE = "#000000"
FILL = "#ffffff"
FILL_ALT = "#f0f0f0"
FILL_DEC = "#e6e6e6"
TEXT = "#000000"
CODE_COLOR = "#1a4b8c"


def _box(ax, cx, cy, w, h, title, body=None, code=None, *,
         face=FILL, lw=1.2, title_size=9.5, body_size=8.5, code_size=8.2):
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=LINE,
        facecolor=face,
    )
    ax.add_patch(patch)
    if body and code:
        ax.text(cx, cy + h * 0.30, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold")
        ax.text(cx, cy + h * 0.02, body, ha="center", va="center",
                color=TEXT, fontsize=body_size)
        ax.text(cx, cy - h * 0.34, code, ha="center", va="center",
                color=CODE_COLOR, fontsize=code_size, family="monospace",
                fontstyle="italic")
    elif body:
        ax.text(cx, cy + h * 0.22, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold")
        ax.text(cx, cy - h * 0.18, body, ha="center", va="center",
                color=TEXT, fontsize=body_size)
    elif code:
        ax.text(cx, cy + h * 0.18, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold")
        ax.text(cx, cy - h * 0.28, code, ha="center", va="center",
                color=CODE_COLOR, fontsize=code_size, family="monospace",
                fontstyle="italic")
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                color=TEXT, fontsize=title_size, fontweight="bold")


def _diamond(ax, cx, cy, w, h, text, code=None, *, face=FILL_DEC, lw=1.2, fs=8.5, code_size=8.0):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    poly = plt.Polygon(pts, closed=True, facecolor=face, edgecolor=LINE, linewidth=lw)
    ax.add_patch(poly)
    if code:
        ax.text(cx, cy + h * 0.10, text, ha="center", va="center",
                color=TEXT, fontsize=fs)
        ax.text(cx, cy - h * 0.30, code, ha="center", va="center",
                color=CODE_COLOR, fontsize=code_size, family="monospace",
                fontstyle="italic")
    else:
        ax.text(cx, cy, text, ha="center", va="center", color=TEXT, fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, *, lw=1.1, label=None,
           label_offset=(0.0, 0.10), style="-|>"):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        color=LINE,
        linewidth=lw,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            color=TEXT,
            fontsize=8,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 10.0), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 13.4)
    ax.axis("off")

    _box(
        ax, 5.0, 12.35, 6.6, 1.35,
        "Stage 1: Data Preparation (US)",
        "Alpha Vantage / yfinance daily bars aligned on the\n"
        "NYSE / NASDAQ trading calendar; multi-aspect factors",
        code="FeatureUS_US.py   +   Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
    )

    _box(
        ax, 5.0, 10.70, 6.6, 1.20,
        "Stage 2: For each feature aspect (4 in total)",
        "fundamental, moment, tech_trend, macro",
        code="ATT+Flood_US.py   |   ATT+Dflooding_US.py",
        face=FILL_ALT,
    )

    _box(
        ax, 5.0, 9.15, 6.6, 1.05,
        "Stage 3: Aspect-specific feature transform and scaling",
        code=None,
    )
    ax.text(5.0, 9.15 - 1.05 * 0.28,
            "ATT+Flood_US.py  (aspect-specific StandardScaler + windowing)",
            ha="center", va="center", color=CODE_COLOR, fontsize=8.2,
            family="monospace", fontstyle="italic")

    _diamond(
        ax, 5.0, 7.65, 4.4, 1.55,
        "Validation split\n(blocking / expanding /\nrolling walk-forward)",
        code="ATT+Flood_US.py  (WF_N_SPLITS=5, WF_VAL_RATIO=0.20)",
        code_size=7.6,
    )

    _box(
        ax, 5.0, 5.75, 7.2, 1.60,
        "Stage 4: Two-stage Bayesian Hyperparameter Search",
        ("Stage A: n trials  x  300 epochs  (broad),   n \u2265 18\n"
         "Stage B: m trials  x  300 epochs  (refine),  m \u2265 32\n"
         "search space includes lookback window and network HPs"),
        code="ATT+Flood_US.py   (Bayesian tuning + static Flooding grid b \u2208 {0.00..0.40})",
    )

    _box(
        ax, 5.0, 3.85, 7.2, 1.40,
        "Stage 5: Fixed-hyperparameter Retraining",
        ("Attention encoder with dynamic flooding regularizer;\n"
         "k repeats (k \u2265 18); keep the top-3 models by validation score"),
        code="ATT+Dflooding_US.py   (Dynamic Flooding + top-3 by val score)",
    )

    _box(
        ax, 9.45, 5.75, 1.05, 0.65,
        "next aspect",
        face=FILL_ALT, title_size=8.2, lw=0.9,
    )

    _box(
        ax, 5.0, 2.20, 7.2, 1.30,
        "Stage 6: Per-aspect Prediction Generation",
        "Load top-3 models per aspect; produce out-of-sample\n"
        "probability sequences on the evaluation window",
        code="prediction_update_US.py \u2192 model_pred_DES_US/DES_pred_{ticker}.csv",
    )

    _box(
        ax, 5.0, 0.65, 7.2, 1.30,
        "Stage 7: DES + CUSUM Filter + Portfolio Backtest",
        ("KNORA-E over 4-aspect predictions, CUSUM directional filter,\n"
         "then price / market-weighted portfolio backtest"),
        code="DES_update_ATT_US.py \u2192 CumsumPro_US.py \u2192 Backtest_Portfolio_US.py",
    )

    _arrow(ax, 5.0, 11.675, 5.0, 11.30)
    _arrow(ax, 5.0, 10.10, 5.0, 9.675)
    _arrow(ax, 5.0, 8.625, 5.0, 8.425)
    _arrow(ax, 5.0, 6.875, 5.0, 6.55)
    _arrow(ax, 5.0, 4.95, 5.0, 4.55,
           label="best HPs", label_offset=(0.75, 0))

    _arrow(ax, 8.6, 3.85, 9.45, 3.85, style="-")
    _arrow(ax, 9.45, 3.85, 9.45, 5.425, style="-")
    _arrow(ax, 9.45, 6.075, 9.45, 10.70, style="-")
    _arrow(ax, 9.45, 10.70, 8.30, 10.70)

    _arrow(ax, 5.0, 3.15, 5.0, 2.85,
           label="after 4 aspects", label_offset=(1.0, 0))

    _arrow(ax, 5.0, 1.55, 5.0, 1.30)

    plt.tight_layout()
    fig.savefig(OUT_PATH, facecolor=BG, bbox_inches="tight")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
