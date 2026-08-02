# =============================================================================
# Backtest_Portfolio_US.py
# Purpose: US equity portfolio experiment.
#   1. User picks universe (dow30/sp100/sox30 keyword, a,b comma list, or @file.txt)
#   2. Weighting scheme (price-weighted / market-weighted)
#   3. Total capital (default 100M USD) is allocated to constituents by weight at start
#   4. Each ticker enters/exits from its trained DES prediction
#      (model_pred_DES_US/DES_pred_<t>_2019-12-31.csv) plus a CUSUM directional
#      filter (cumSum_prob_12/cusum_<t>.csv)
#   5. Per-ticker daily asset values are summed into portfolio equity, plotted
#      against the benchmark as cumulative return
#
# Interactive prompt style aligns with DES_update_ATT_US.py; enter 0 to quit.
# Supports env-driven non-interactive batch mode: BT_UNIVERSE / BT_WEIGHT /
# BT_START / BT_END / BT_CAPITAL / BT_BENCHMARK / BT_CUSUM / BT_TOTAL_RETURN /
# BT_THRESHOLD. If BT_UNIVERSE is set, runs once and exits.
# Plot/show behaviour matches DES: MPLBACKEND / SHOW_FIG / SAVE_FIG.
#
# Outputs (backtest_portfolio_US/, flat layout, params encoded in filename):
#   - equity_<universe>_<weight>_<start>_<end>.csv
#   - summary_<universe>_<weight>_<start>_<end>.csv
#   - cum_return_<universe>_<weight>_<start>_<end>.png
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from feature._us_data import (  # noqa: E402
    load_price_frames,
    load_market_index,
    DOW_30_TICKER,
    SOX_30_TICKER,
    _dividends,
)
from _sp100_diff import SP100  # noqa: E402
from DES_update_ATT_US import DOW30_NAME  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DES_PRED_DIR = _THIS_DIR / "model_pred_DES_US"
CUSUM_SIGN_DIR = _THIS_DIR / "cumSum_prob_12"
RAW_DIR = _THIS_DIR / "feature" / "_raw"
OUT_DIR = _THIS_DIR / "backtest_portfolio_US"

DEFAULT_TRAIN_END = "2019-12-31"  # matches DES_pred_*_2019-12-31.csv
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-03-31"
DEFAULT_CAPITAL = 100_000_000.0  # 100M USD
DEFAULT_THRESHOLD = 0.50
ANN_FACTOR = 252  # annualization factor (trading days)
BUY_FEE = 0.001    # 0.1 % (round-trip fee — buy side)
SELL_FEE = 0.0034  # 0.34 % (round-trip fee — sell side)

def _load_ndx100_from_file() -> list[str]:
    """Read Nasdaq-100 ticker list from _ndx100_all.txt (one per line, '#' comments)."""
    p = _THIS_DIR / "_ndx100_all.txt"
    if not p.exists():
        return []
    return [
        ln.strip().upper()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


PRESET_UNIVERSES: dict[str, tuple[list[str], str]] = {
    "dow30": (list(DOW_30_TICKER), "^DJI"),
    "sp100": (list(SP100), "^OEX"),
    "sox30": (list(SOX_30_TICKER), "^SOX"),
    "ndx100": (_load_ndx100_from_file(), "^NDX"),
}


# --------------------------------------------------------------------------- #
# Universe resolver
# --------------------------------------------------------------------------- #
def resolve_universe(spec: str) -> tuple[list[str], str | None]:
    """Resolve a universe spec into (tickers, default_benchmark).

    spec: keyword in PRESET_UNIVERSES, or comma-separated tickers, or @path/to/file.
    """
    s = (spec or "").strip()
    if not s:
        s = "dow30"
    if s.lower() in PRESET_UNIVERSES:
        tickers, bench = PRESET_UNIVERSES[s.lower()]
        return list(tickers), bench
    if s.startswith("@"):
        path = Path(s[1:]).expanduser()
        if not path.is_absolute():
            path = _THIS_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"ticker file not found: {path}")
        toks = [
            ln.strip().upper()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        return toks, None
    if "," in s:
        return [t.strip().upper() for t in s.split(",") if t.strip()], None
    return [s.upper()], None


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def compute_price_weights(tickers: list[str], start_open: pd.Series) -> pd.Series:
    """w_i = Open_i(start) / Σ Open_j(start). Tickers with no price are dropped."""
    avail = start_open.reindex(tickers).dropna()
    if avail.empty:
        raise RuntimeError("compute_price_weights: no price data on start date for any ticker")
    return avail / avail.sum()


def _shares_get_shares_full(ticker: str, start_date) -> tuple[float | None, str]:
    try:
        import yfinance as yf
        s_lo = (pd.Timestamp(start_date) - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        s_hi = (pd.Timestamp(start_date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        ser = yf.Ticker(ticker).get_shares_full(start=s_lo, end=s_hi)
        if ser is None or len(ser) == 0:
            return None, "get_shares_full:empty"
        ser = ser.dropna()
        ser.index = pd.to_datetime(ser.index).tz_localize(None)
        ser = ser[ser.index <= pd.Timestamp(start_date)]
        if ser.empty:
            return None, "get_shares_full:all_after_start"
        return float(ser.iloc[-1]), "get_shares_full"
    except Exception as e:
        return None, f"get_shares_full:err:{e!r}"


def _shares_fast_info(ticker: str) -> tuple[float | None, str]:
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        v = None
        try:
            v = fi["shares"]
        except Exception:
            v = getattr(fi, "shares", None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None, "fast_info:empty"
        return float(v), "fast_info"
    except Exception as e:
        return None, f"fast_info:err:{e!r}"


def _shares_av_cache(ticker: str) -> tuple[float | None, str]:
    path = RAW_DIR / f"av_BALANCE_SHEET_{ticker}.json"
    if not path.exists():
        return None, "av_cache:no_file"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        reports = (data.get("quarterlyReports") or []) + (data.get("annualReports") or [])
        for r in reports:
            v = r.get("commonStockSharesOutstanding")
            if v and str(v).lower() not in ("none", ""):
                return float(v), "av_cache"
        return None, "av_cache:no_field"
    except Exception as e:
        return None, f"av_cache:err:{e!r}"


def get_shares_outstanding(ticker: str, start_date) -> tuple[float | None, str]:
    """Three-tier fallback: get_shares_full → fast_info → AV balance sheet."""
    for fn, args in (
        (_shares_get_shares_full, (ticker, start_date)),
        (_shares_fast_info, (ticker,)),
        (_shares_av_cache, (ticker,)),
    ):
        v, src = fn(*args)
        if v is not None and v > 0:
            return v, src
    return None, "all_failed"


def compute_market_weights(
    tickers: list[str], start_open: pd.Series, start_date
) -> tuple[pd.Series, dict[str, str]]:
    """w_i = (shares_i × Open_i(start)) / Σ caps. Tickers missing shares or price are skipped."""
    caps: dict[str, float] = {}
    sources: dict[str, str] = {}
    for t in tickers:
        if t not in start_open.index or pd.isna(start_open[t]):
            print(f"  [WARN] {t}: no price on start date, skip")
            sources[t] = "no_price"
            continue
        sh, src = get_shares_outstanding(t, start_date)
        sources[t] = src
        if sh is None:
            print(f"  [WARN] {t}: cannot resolve shares outstanding ({src}), skip")
            continue
        caps[t] = sh * float(start_open[t])
    if not caps:
        raise RuntimeError("compute_market_weights: no market cap resolved for any ticker")
    s = pd.Series(caps, dtype=float)
    return s / s.sum(), sources


# --------------------------------------------------------------------------- #
# Single-stock trade experiment (extracted from DES_update_ATT_US.plot_backtest)
# --------------------------------------------------------------------------- #
def _make_signals(
    AGG_DES1: pd.Series,
    long: int,
    short: int,
    short_to_long: int,
    long_to_short: int,
) -> tuple[pd.Series, pd.Series]:
    """Replicate plot_backtest signal pattern logic exactly.

    Buy patterns: pat=[0]*s2l+[1]*long, pat1=[0,1]+[1]*long, pat2=[1,0]+[1]*long.
    Sell pattern: [1]*l2s + [0]*short.
    """
    arr = AGG_DES1.values
    n = len(arr)

    def _check(i: int, pats: list[list[int]]) -> bool:
        for p in pats:
            L = len(p)
            if L == 0:
                continue
            if i >= L - 1 and np.array_equal(arr[i - L + 1: i + 1], np.array(p, dtype=arr.dtype)):
                return True
        return False

    pats_buy = [
        [0] * short_to_long + [1] * long,
        [0, 1] + [1] * long,
        [1, 0] + [1] * long,
    ]
    pats_sell = [
        [1] * long_to_short + [0] * short,
    ]

    sig_buy: list[int] = []
    if n > 0 and arr[0] == 0:
        for i in range(n):
            sig_buy.append(1 if _check(i, pats_buy) else 0)
    else:
        if n > 0:
            sig_buy.append(1)
        for i in range(1, n):
            sig_buy.append(1 if _check(i, pats_buy) else 0)

    sig_sell = [(-1 if _check(i, pats_sell) else 0) for i in range(n)]
    return (
        pd.Series(sig_buy, index=AGG_DES1.index, dtype=int),
        pd.Series(sig_sell, index=AGG_DES1.index, dtype=int),
    )


def run_single_stock_experiment(
    prob: pd.Series,
    stock_price: pd.DataFrame,
    cusum: pd.Series | None,
    threshold: float,
    initial_capital: float,
    use_cusum_filter: bool = True,
    long: int = 1,
    short: int = 1,
    short_to_long: int = 0,
    long_to_short: int = 0,
    dividends: pd.Series | None = None,
) -> dict:
    """Pure trade engine matching DES_update_ATT_US.plot_backtest semantics.

    Inputs are sliced to the same time window before being passed in.
    Buy executes at next-day Open after a buy signal; sell at next-day Open after
    a sell signal. If `dividends` is provided, ex-date cash dividends are added
    to cash for the number of shares held (no auto-reinvestment into shares).
    """
    # Align indices
    idx = pd.DatetimeIndex(stock_price.index)
    prob = prob.reindex(idx).ffill().fillna(0.0)
    if use_cusum_filter:
        if cusum is None:
            raise ValueError("use_cusum_filter=True but cusum is None")
        cusum = cusum.reindex(idx).ffill().fillna(0.0)

    n = len(idx)
    if n < 2:
        raise RuntimeError("run_single_stock_experiment: <2 rows after alignment")

    AGG_DES1 = (prob > threshold).astype(int)
    sig_buy, sig_sell = _make_signals(AGG_DES1, long, short, short_to_long, long_to_short)

    cash = np.zeros(n, dtype=float)
    shares = np.zeros(n, dtype=float)
    asset = np.zeros(n, dtype=float)
    cash[0] = initial_capital
    asset[0] = initial_capital

    opens = stock_price["Open"].values
    closes = stock_price["Close"].values
    sb = sig_buy.values
    ss = sig_sell.values
    cs_vals = cusum.values if use_cusum_filter else None

    n_buy = 0
    n_sell = 0
    div_vals = None
    if dividends is not None and len(dividends) > 0:
        # only keep ex-dates inside window
        div_vals = dividends.reindex(idx).fillna(0.0).values

    for i in range(1, n):
        oi = opens[i]
        ci = closes[i]
        prev_shares = shares[i - 1]
        prev_cash = cash[i - 1]
        if use_cusum_filter:
            cv = float(np.asarray(cs_vals[i]).flatten()[0])
        else:
            cv = 1.0  # pass-through

        if (
            sb[i - 1] == 1
            and prev_shares == 0
            and (not use_cusum_filter or cv > 0)
            and oi > 0
        ):
            sh = float(prev_cash // oi)
            cost = sh * oi * BUY_FEE
            shares[i] = sh
            cash[i] = prev_cash - cost - sh * oi
            asset[i] = cash[i] + sh * ci
            n_buy += 1
        elif (
            ss[i - 1] == -1
            and prev_shares != 0
            and (not use_cusum_filter or cv < 0)
        ):
            cost = prev_shares * oi * SELL_FEE
            cash[i] = prev_shares * oi - cost + prev_cash
            shares[i] = 0.0
            asset[i] = cash[i]
            n_sell += 1
        else:
            cash[i] = prev_cash
            shares[i] = prev_shares
            asset[i] = cash[i] + prev_shares * ci

        # Dividend reinvestment into cash (total-return mode)
        if div_vals is not None and shares[i] > 0:
            d = float(div_vals[i])
            if d > 0:
                cash[i] += shares[i] * d
                asset[i] = cash[i] + shares[i] * ci

    equity = pd.Series(asset, index=idx, name="equity")
    return {
        "equity": equity,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "final_return": float(equity.iloc[-1] / initial_capital - 1.0),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _sharpe_maxdd(eq: pd.Series) -> tuple[float, float]:
    r = eq.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(r.mean() / r.std() * np.sqrt(ANN_FACTOR))
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    return sharpe, float(dd) * 100


# --------------------------------------------------------------------------- #
# Portfolio orchestration
# --------------------------------------------------------------------------- #
def run_one(
    universe_spec: str,
    weight: str,
    start: str,
    end: str,
    capital: float,
    benchmark: str | None,
    threshold: float,
    use_cusum_filter: bool,
    total_return: bool,
    train_end: str = DEFAULT_TRAIN_END,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers, default_bench = resolve_universe(universe_spec)
    bench = benchmark or default_bench
    label = universe_spec.strip().lower() if universe_spec else "dow30"
    print(f"[INFO] universe={label}  ({len(tickers)} requested)  benchmark={bench}")

    # ---- Price data ----
    print("[INFO] loading OHLCV ...")
    frames = load_price_frames(tickers)
    open_w = frames["Open"].loc[start:end]
    close_w = frames["Close"].loc[start:end]
    if open_w.empty:
        raise RuntimeError(f"no price data in window [{start}, {end}]")
    start_date = open_w.index[0]
    end_date = open_w.index[-1]
    print(f"[INFO] window {start_date.date()} -> {end_date.date()}  ({len(open_w)} bars)")
    start_open = open_w.iloc[0]

    # ---- Filter to tickers with DES prediction file ----
    valid: list[str] = []
    skipped: list[tuple[str, str]] = []
    for t in tickers:
        if not (DES_PRED_DIR / f"DES_pred_{t}_{train_end}.csv").exists():
            skipped.append((t, "no DES_pred CSV"))
            continue
        if use_cusum_filter and not (CUSUM_SIGN_DIR / f"cusum_{t}.csv").exists():
            skipped.append((t, "no cusum_prob_12 CSV"))
            continue
        if t not in start_open.index or pd.isna(start_open[t]):
            skipped.append((t, "no price on start"))
            continue
        valid.append(t)
    if skipped:
        print(f"[WARN] {len(skipped)} tickers skipped (renormalising weights over the rest):")
        for t, r in skipped:
            print(f"        {t}: {r}")
    if not valid:
        raise RuntimeError("no valid tickers after filtering")

    # ---- Weights ----
    print(f"[INFO] computing {weight}-weighted allocation over {len(valid)} tickers ...")
    sw = start_open.reindex(valid)
    if weight == "price":
        weights = compute_price_weights(valid, sw)
        share_sources: dict[str, str] = {}
    elif weight == "market":
        weights, share_sources = compute_market_weights(valid, sw, start_date)
    else:
        raise ValueError(f"unknown weight: {weight!r} (use 'price' or 'market')")
    valid = [t for t in valid if t in weights.index]  # drop ones with no cap
    initial_caps = (weights * capital).reindex(valid)
    print(f"[INFO] Σweight = {weights.sum():.6f}  Σinitial_capital = ${initial_caps.sum():,.2f}")
    print("[INFO] Top-5 by weight:")
    for t, w in weights.sort_values(ascending=False).head(5).items():
        src = share_sources.get(t, "-")
        print(f"        {t:6s}  weight={w*100:6.2f}%  init=${initial_caps[t]:>15,.2f}  src={src}")

    # ---- Per-stock experiment ----
    per_eq: dict[str, pd.Series] = {}
    per_stats: dict[str, dict] = {}
    print(f"[INFO] running per-stock trade experiments ...")
    for t in valid:
        try:
            prob = pd.read_csv(
                DES_PRED_DIR / f"DES_pred_{t}_{train_end}.csv",
                index_col=0, parse_dates=True,
            ).iloc[:, 0]
            prob = prob[~prob.index.duplicated(keep="last")].sort_index()

            cusum = None
            if use_cusum_filter:
                cusum = pd.read_csv(
                    CUSUM_SIGN_DIR / f"cusum_{t}.csv",
                    index_col=0, parse_dates=True, header=None,
                ).iloc[:, 0]
                cusum = cusum[~cusum.index.duplicated(keep="last")].sort_index()

            sp = pd.DataFrame({
                "Open":  frames["Open"][t],
                "High":  frames["High"][t],
                "Low":   frames["Low"][t],
                "Close": frames["Close"][t],
            }).loc[start:end].dropna()
            if len(sp) < 2:
                print(f"  [ERR ] {t}: <2 price bars, skip")
                continue

            divs = None
            if total_return:
                try:
                    divs = _dividends(t)
                    divs = divs[~divs.index.duplicated(keep="last")].sort_index()
                except Exception as e:
                    print(f"  [WARN] {t}: dividend fetch failed ({e}); skipping div in TR")
                    divs = None

            res = run_single_stock_experiment(
                prob=prob,
                stock_price=sp,
                cusum=cusum,
                threshold=threshold,
                initial_capital=float(initial_caps[t]),
                use_cusum_filter=use_cusum_filter,
                dividends=divs,
            )
            per_eq[t] = res["equity"]
            sharpe, maxdd = _sharpe_maxdd(res["equity"])
            per_stats[t] = {
                "weight": float(weights[t]),
                "initial_capital": float(initial_caps[t]),
                "final_asset": float(res["equity"].iloc[-1]),
                "return_pct": res["final_return"] * 100,
                "n_buy": res["n_buy"],
                "n_sell": res["n_sell"],
                "sharpe": sharpe,
                "max_dd_pct": maxdd,
            }
            print(
                f"  [{t:6s}] init=${initial_caps[t]:>13,.0f}  final=${res['equity'].iloc[-1]:>13,.0f}  "
                f"ret={res['final_return']*100:+7.2f}%  buys={res['n_buy']:3d}  sells={res['n_sell']:3d}  "
                f"Sharpe={sharpe:+.2f}  DD={maxdd:+.2f}%"
            )
        except Exception as e:
            print(f"  [ERR ] {t}: {e}")

    if not per_eq:
        raise RuntimeError("no per-stock equity series produced")

    # ---- Aggregate portfolio ----
    # Align on union of indices, forward-fill missing per-stock equity, then sum.
    eq_df = pd.DataFrame(per_eq).sort_index()
    eq_df = eq_df.ffill()
    # Fill any leading NaN with the per-stock initial capital so total equity == capital at t0
    for t in valid:
        if t in eq_df.columns and pd.isna(eq_df[t].iloc[0]):
            eq_df[t] = eq_df[t].fillna(float(initial_caps[t]))
    port_eq = eq_df.sum(axis=1)
    port_cum = port_eq / capital - 1.0
    p_sharpe, p_dd = _sharpe_maxdd(port_eq)
    print(
        f"[INFO] portfolio  final = ${port_eq.iloc[-1]:,.0f}  "
        f"ret={port_cum.iloc[-1]*100:+.2f}%  Sharpe={p_sharpe:+.2f}  DD={p_dd:+.2f}%"
    )

    # ---- Benchmark ----
    bench_cum = None
    bench_sharpe = bench_dd = np.nan
    if bench:
        try:
            b = load_market_index(index=bench, start=start, end=end)
            b = b.loc[start:end].dropna()
            if len(b) >= 2:
                bench_cum = b / b.iloc[0] - 1.0
                bench_sharpe, bench_dd = _sharpe_maxdd(b)
                print(
                    f"[INFO] benchmark {bench}  ret={bench_cum.iloc[-1]*100:+.2f}%  "
                    f"Sharpe={bench_sharpe:+.2f}  DD={bench_dd:+.2f}%"
                )
        except Exception as e:
            print(f"[WARN] benchmark {bench} fetch failed: {e}")

    # ---- Outputs ----
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    base = f"{label}_{weight}_{start_s}_{end_s}"

    eq_out = pd.DataFrame({
        "portfolio_equity": port_eq,
        "portfolio_cum_return": port_cum,
    })
    if bench_cum is not None:
        eq_out["benchmark_cum_return"] = bench_cum.reindex(port_eq.index, method="ffill")
    eq_out = pd.concat([eq_out, eq_df.add_suffix("_asset")], axis=1)
    eq_csv = OUT_DIR / f"equity_{base}.csv"
    eq_out.to_csv(eq_csv)

    rows = []
    for t in valid:
        if t in per_stats:
            row = {"ticker": t, **per_stats[t]}
            rows.append(row)
    df_sum = pd.DataFrame(rows)
    total_row = {
        "ticker": "TOTAL",
        "weight": float(df_sum["weight"].sum()) if not df_sum.empty else np.nan,
        "initial_capital": float(df_sum["initial_capital"].sum()) if not df_sum.empty else np.nan,
        "final_asset": float(port_eq.iloc[-1]),
        "return_pct": float(port_cum.iloc[-1] * 100),
        "n_buy": int(df_sum["n_buy"].sum()) if not df_sum.empty else 0,
        "n_sell": int(df_sum["n_sell"].sum()) if not df_sum.empty else 0,
        "sharpe": p_sharpe,
        "max_dd_pct": p_dd,
    }
    rows_out = [total_row] + rows
    if bench_cum is not None:
        rows_out.append({
            "ticker": f"BENCH:{bench}",
            "weight": np.nan,
            "initial_capital": np.nan,
            "final_asset": np.nan,
            "return_pct": float(bench_cum.iloc[-1] * 100),
            "n_buy": 0, "n_sell": 0,
            "sharpe": bench_sharpe, "max_dd_pct": bench_dd,
        })
    pd.DataFrame(rows_out).to_csv(OUT_DIR / f"summary_{base}.csv", index=False)

    # ---- Plot ----
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values * 100,
            label=f"Portfolio ({weight}-weighted)", linewidth=2.2, color="#C44E52")
    if bench_cum is not None:
        ax.plot(bench_cum.index, bench_cum.values * 100,
                label=f"Benchmark {bench}", linewidth=1.7, color="black", alpha=0.85)
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    title_parts = [
        f"{label.upper()} portfolio backtest ({weight}-weighted)",
        f"{start_s} ~ {end_s}   "
        f"Portfolio {port_cum.iloc[-1]*100:+.2f}%"
        + (f"   vs   {bench} {bench_cum.iloc[-1]*100:+.2f}%" if bench_cum is not None else ""),
    ]
    ax.set_title("\n".join(title_parts), fontweight="bold")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    png = OUT_DIR / f"cum_return_{base}.png"
    if os.environ.get("SAVE_FIG", "1") != "0":
        fig.savefig(png, facecolor="white")
    if os.environ.get("SHOW_FIG", "1") != "0":
        plt.show()
    plt.close(fig)

    print(f"[OK] wrote {eq_csv.relative_to(_THIS_DIR)}")
    print(f"[OK] wrote {(OUT_DIR / f'summary_{base}.csv').relative_to(_THIS_DIR)}")
    if os.environ.get("SAVE_FIG", "1") != "0":
        print(f"[OK] wrote {png.relative_to(_THIS_DIR)}")


# --------------------------------------------------------------------------- #
# Interactive / batch driver
# --------------------------------------------------------------------------- #
def _env_or(prompt: str, default: str, env_key: str) -> str:
    """Return env var if set, else prompt the user (default returned on empty Enter)."""
    if env_key in os.environ:
        v = os.environ[env_key]
        print(f"{prompt}{v}   [from env {env_key}]")
        return v
    try:
        s = input(prompt)
    except EOFError:
        return default
    return s if s.strip() else default


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print(" Backtest_Portfolio_US.py  — US portfolio backtest (DES-driven) ")
    print("=" * 64)

    non_interactive = bool(os.environ.get("BT_NON_INTERACTIVE")) or (
        "BT_UNIVERSE" in os.environ
    )

    while True:
        try:
            uspec = _env_or(
                "Universe (dow30 / sp100 / sox30 / a,b / @file, 0 to quit) [dow30]: ",
                "dow30", "BT_UNIVERSE",
            ).strip()
            if uspec == "0":
                print("Exit.")
                break

            w_in = _env_or("Weight (1=price, 2=market) [1]: ", "1", "BT_WEIGHT").strip()
            weight = "market" if w_in in ("2", "market", "m") else "price"

            start = _env_or(f"Start date [{DEFAULT_START}]: ", DEFAULT_START, "BT_START").strip()
            end = _env_or(f"End date [{DEFAULT_END}]: ", DEFAULT_END, "BT_END").strip()
            cap_in = _env_or(
                f"Initial capital USD [{int(DEFAULT_CAPITAL)}]: ",
                str(int(DEFAULT_CAPITAL)), "BT_CAPITAL",
            ).strip()
            try:
                capital = float(cap_in)
            except ValueError:
                capital = DEFAULT_CAPITAL

            bench_in = _env_or(
                "Benchmark (yfinance symbol; Enter=universe default): ",
                "", "BT_BENCHMARK",
            ).strip()
            benchmark = bench_in or None

            cusum_in = _env_or("CUSUM filter? (1=yes / 2=no) [1]: ", "1", "BT_CUSUM").strip()
            use_cusum = cusum_in not in ("2", "n", "N", "no", "false")

            tr_in = _env_or(
                "Total return (reinvest cash dividends)? (1=yes / 2=no) [2]: ",
                "2", "BT_TOTAL_RETURN",
            ).strip()
            total_return = tr_in in ("1", "y", "Y", "yes", "true")

            thr_in = _env_or(
                f"DES threshold [{DEFAULT_THRESHOLD}]: ",
                str(DEFAULT_THRESHOLD), "BT_THRESHOLD",
            ).strip()
            try:
                threshold = float(thr_in)
            except ValueError:
                threshold = DEFAULT_THRESHOLD

            run_one(
                universe_spec=uspec,
                weight=weight,
                start=start,
                end=end,
                capital=capital,
                benchmark=benchmark,
                threshold=threshold,
                use_cusum_filter=use_cusum,
                total_return=total_return,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

        if non_interactive:
            break


if __name__ == "__main__":
    main()
