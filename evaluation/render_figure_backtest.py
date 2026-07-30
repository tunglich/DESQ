"""Regenerate the 3-panel out-of-sample backtest figure from the shipped CSVs.

Reads:
    evaluation/backtest_2330.csv           (TSMC)
    evaluation/backtest_2454.csv           (MediaTek)
    evaluation/backtest_portfolio_tw50.csv (TW-50 model portfolio vs TWA02)

Writes:
    evaluation/figure_backtest_overview.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
OUT_PNG = EVAL_DIR / "figure_backtest_overview.png"


def _load_stock(fp: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date")
    return df.index, df["Model_Return_Pct"], df["Stock_Return_Pct"]


def _load_portfolio(fp: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date")
    return df.index, df["Model_CumRet_Pct"], df["Benchmark_CumRet_Pct"]


def _plot(ax, dates, model, bench, title, model_label, bench_label):
    ax.plot(dates, model, color="#1f77b4", linewidth=1.3, label=model_label)
    ax.plot(dates, bench, color="black", linewidth=1.3, label=bench_label)
    m_last = float(model.dropna().iloc[-1])
    b_last = float(bench.dropna().iloc[-1])
    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.annotate(f"{m_last:+.2f}%", xy=(dates[-1], m_last),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color="#1f77b4", va="center")
    ax.annotate(f"{b_last:+.2f}%", xy=(dates[-1], b_last),
                xytext=(6, -12), textcoords="offset points",
                fontsize=9, color="black", va="center")


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)

    d1, m1, b1 = _load_stock(EVAL_DIR / "backtest_2330.csv")
    _plot(axes[0], d1, m1, b1, "TSMC (2330.TT) Backtest", "DESQ", "TSMC buy-and-hold")

    d2, m2, b2 = _load_stock(EVAL_DIR / "backtest_2454.csv")
    _plot(axes[1], d2, m2, b2, "MediaTek (2454.TT) Backtest", "DESQ", "MediaTek buy-and-hold")

    d3, m3, b3 = _load_portfolio(EVAL_DIR / "backtest_portfolio_tw50.csv")
    _plot(axes[2], d3, m3, b3, "Model Portfolio Backtest", "DESQ", "TWA02 buy-and-hold")

    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(35)
            lbl.set_ha("right")

    fig.suptitle("Out-of-sample back-tests over 2024-01-02 to 2026-03-31",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
