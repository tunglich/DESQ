"""Plot DES portfolio backtest (price- and market-weighted) on the same chart
as the FinRL Dow-30 ensemble backtest results.

Inputs (read-only):
  - backtest_portfolio_US/equity_dow30_price_<S>_<E>.csv
  - backtest_portfolio_US/equity_dow30_market_<S>_<E>.csv
  - FinRL/backtest_dow30_2024_20260331/backtest_results.csv
  - FinRL/backtest_dow30_2024_20260331/metrics.csv

Output:
  - backtest_portfolio_US/cum_return_dow30_vs_finrl_<S>_<E>.png
  - backtest_portfolio_US/summary_dow30_vs_finrl_<S>_<E>.csv

Environment overrides:
  BT_START, BT_END   — date range used in the DES output filenames
  MPLBACKEND         — set to 'Agg' for headless runs
  SHOW_FIG, SAVE_FIG — '0' to suppress (default '1')
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Academic style mirrors Backtest_Portfolio_US.py
plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


SHOW_FIG = os.environ.get("SHOW_FIG", "1") == "1"
SAVE_FIG = os.environ.get("SAVE_FIG", "1") == "1"

START = os.environ.get("BT_START", "2024-01-02")
END = os.environ.get("BT_END", "2026-03-31")

ROOT = os.path.dirname(os.path.abspath(__file__))
DES_DIR = os.path.join(ROOT, "backtest_portfolio_US")
FINRL_DIR = os.path.join(ROOT, "FinRL", "backtest_dow30_2024_20260331")


def _load_des(weight: str) -> pd.Series:
    """Return cumulative-return Series (decimal, start≈0) indexed by Date."""
    fn = os.path.join(DES_DIR, f"equity_dow30_{weight}_{START}_{END}.csv")
    df = pd.read_csv(fn, parse_dates=["Date"]).set_index("Date")
    s = df["portfolio_cum_return"].astype(float)
    s.name = f"DES ({weight}-weighted)"
    return s


def _load_des_benchmark() -> pd.Series:
    fn = os.path.join(DES_DIR, f"equity_dow30_price_{START}_{END}.csv")
    df = pd.read_csv(fn, parse_dates=["Date"]).set_index("Date")
    s = df["benchmark_cum_return"].astype(float)
    s.name = "^DJI (DES bench)"
    return s


def _load_finrl() -> pd.DataFrame:
    """Return FinRL agents as cumulative returns (decimal). NaN tails are ffill'd."""
    fn = os.path.join(FINRL_DIR, "backtest_results.csv")
    df = pd.read_csv(fn, parse_dates=["date"]).set_index("date")
    df.index.name = "Date"
    df = df.astype(float).ffill()
    cum = df / df.iloc[0] - 1.0
    cum.columns = [c.upper() for c in cum.columns]
    return cum


def _load_finrl_metrics() -> pd.DataFrame:
    fn = os.path.join(FINRL_DIR, "metrics.csv")
    return pd.read_csv(fn)


def _ann_sharpe_maxdd(eq: pd.Series) -> tuple[float, float]:
    """Annualised Sharpe (rf=0, √252) and MaxDD% from a cumulative-return series."""
    eq = eq.dropna()
    if len(eq) < 3:
        return float("nan"), float("nan")
    nav = 1.0 + eq.values
    rets = np.diff(nav) / nav[:-1]
    mu = float(np.nanmean(rets))
    sd = float(np.nanstd(rets, ddof=1))
    sharpe = (mu / sd) * np.sqrt(252.0) if sd > 0 else float("nan")
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    maxdd_pct = float(dd.min() * 100.0)
    return sharpe, maxdd_pct


def main() -> None:
    des_price = _load_des("price")
    des_market = _load_des("market")
    bench = _load_des_benchmark()
    finrl = _load_finrl()

    # ---- summary table ---------------------------------------------------
    rows = []
    for name, s in [
        ("DES price-weighted", des_price),
        ("DES market-weighted", des_market),
        ("^DJI (yfinance)", bench),
    ]:
        sh, dd = _ann_sharpe_maxdd(s)
        rows.append(
            {
                "strategy": name,
                "final_return_%": float(s.iloc[-1] * 100.0),
                "sharpe": sh,
                "max_dd_%": dd,
                "n_days": int(s.shape[0]),
                "first_date": str(s.index[0].date()),
                "last_date": str(s.index[-1].date()),
            }
        )
    for col in finrl.columns:
        s = finrl[col]
        sh, dd = _ann_sharpe_maxdd(s)
        rows.append(
            {
                "strategy": f"FinRL {col}",
                "final_return_%": float(s.iloc[-1] * 100.0),
                "sharpe": sh,
                "max_dd_%": dd,
                "n_days": int(s.shape[0]),
                "first_date": str(s.index[0].date()),
                "last_date": str(s.index[-1].date()),
            }
        )
    summary = pd.DataFrame(rows)
    print("\n=== Combined summary (DES + FinRL) ===")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    summary_path = os.path.join(
        DES_DIR, f"summary_dow30_vs_finrl_{START}_{END}.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"\nsaved summary → {summary_path}")

    # ---- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # FinRL agents (thin, distinct colors); skip DJI — redundant with yfinance bench
    finrl_styles = {
        "A2C": ("#1f77b4", "-"),
        "PPO": ("#2ca02c", "-"),
        "DDPG": ("#9467bd", "-"),
        "TD3": ("#8c564b", "-"),
        "SAC": ("#e377c2", "-"),
        "MVO": ("#17becf", "--"),
    }
    for col in finrl.columns:
        if col == "DJI":
            continue
        color, ls = finrl_styles.get(col, ("#444444", "-"))
        s = finrl[col]
        label = f"FinRL {col} ({s.iloc[-1] * 100:+.2f}%)"
        ax.plot(s.index, s.values * 100.0, color=color, linestyle=ls, lw=1.3, label=label)

    # DES portfolios (bold) + DES benchmark (black dashed)
    ax.plot(
        bench.index,
        bench.values * 100.0,
        color="black",
        linestyle="--",
        lw=1.6,
        label=f"^DJI yfinance ({bench.iloc[-1] * 100:+.2f}%)",
    )
    ax.plot(
        des_market.index,
        des_market.values * 100.0,
        color="#ff7f0e",
        lw=2.2,
        label=f"DES market-weighted ({des_market.iloc[-1] * 100:+.2f}%)",
    )
    ax.plot(
        des_price.index,
        des_price.values * 100.0,
        color="#d62728",
        lw=2.4,
        label=f"DES price-weighted ({des_price.iloc[-1] * 100:+.2f}%)",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(
        "Dow 30 backtest — DES portfolio vs FinRL ensemble\n"
        f"{START} ~ {END}   |   start capital normalised to $1"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    fig.tight_layout()

    if SAVE_FIG:
        out = os.path.join(
            DES_DIR, f"cum_return_dow30_vs_finrl_{START}_{END}.png"
        )
        fig.savefig(out)
        print(f"saved plot    → {out}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
