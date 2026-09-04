"""Yang 2018 DSR — generalized to any universe (Dow30 / SP100 / NDX100).

Same methodology as `sp100_dsr_backtest.py`, but universe is parameterised.
Tickers without complete feature files are skipped automatically.

Usage:
    python dsr_backtest.py dow30      # → baselines/dsr_yang/backtest_dow30_2024_20260330/
    python dsr_backtest.py sp100      # → baselines/dsr_yang/backtest_sp100_2024_20260330/
    python dsr_backtest.py ndx100     # → baselines/dsr_yang/backtest_ndx100_2024_20260330/
"""
from __future__ import annotations

import csv as _csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# =============================================================================
# Universe registry
# =============================================================================
WS = Path(r"d:\US_stock")
FEATURE_DIR = WS / "feature"
RAW_DIR = FEATURE_DIR / "_raw"

sys.path.insert(0, str(WS))
from feature._us_data import DOW_30_TICKER          # noqa: E402
from _sp100_diff import SP100                        # noqa: E402


def _load_ndx100() -> List[str]:
    fp = Path(r"d:\DRL\data\ndx100_2026-06-09.csv")
    if not fp.exists():
        raise FileNotFoundError(fp)
    tickers: List[str] = []
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            tickers.append(row[1].strip())
    return tickers


UNIVERSES: Dict[str, Dict] = {
    "dow30": {
        "tickers":       DOW_30_TICKER,
        "benchmark":     "^DJI",
        "des_equity":    WS / "backtest_portfolio_US" / "equity_dow30_market_2024-01-02_2026-03-31.csv",
        "out_dir_name":  "backtest_dow30_2024_20260330",
        "label":         "Dow 30",
    },
    "sp100": {
        "tickers":       SP100,
        "benchmark":     "^OEX",
        "des_equity":    WS / "backtest_portfolio_US" / "equity_sp100_market_2024-01-02_2026-03-31.csv",
        "out_dir_name":  "backtest_sp100_2024_20260330",
        "label":         "S&P 100",
    },
    "ndx100": {
        "tickers":       None,  # lazy — populated in main()
        "benchmark":     "^NDX",
        "des_equity":    WS / "backtest_portfolio_US" / "equity_ndx100_market_2024-01-02_2026-03-31.csv",
        "out_dir_name":  "backtest_ndx100_2024_20260330",
        "label":         "Nasdaq 100",
    },
}


# =============================================================================
# Config (identical to sp100_dsr_backtest.py)
# =============================================================================
TRADE_START = pd.Timestamp("2024-01-02")
TRADE_END   = pd.Timestamp("2026-03-30")
INITIAL_CAPITAL = 1_000_000

TRAIN_Q = 16
TEST_Q  = 4
TOP_PCT = 0.20
COV_LOOKBACK_DAYS = 63

# Transaction cost applied at each quarterly rebalance (0.05 % buy + 0.05 %
# sell = 0.10 % roundtrip, matches the DRL Ensemble and MACE-baseline configs).
BUY_FEE  = 0.0005
SELL_FEE = 0.0005

FUND_COLS = [
    "PE_trailing", "PEG", "PBR", "DY",
    "R_acc_yoy",
    "E_qoq", "E_yoy", "E_acc_yoy",
    "Op_qoq", "Op_yoy", "Op_acc_yoy",
    "Gross", "Gross_qoq",
    "EPS_qoq",
]


# =============================================================================
# Data loading (identical helpers)
# =============================================================================
def _load_fundamental(tic: str) -> pd.DataFrame | None:
    fp = FEATURE_DIR / f"fundamental_{tic}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    keep = [c for c in FUND_COLS if c in df.columns]
    return df[keep]


def _load_prices(tic: str) -> pd.Series | None:
    fp = FEATURE_DIR / f"tech_trend_{tic}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, usecols=["Date", "close"], parse_dates=["Date"])
    return df.set_index("Date")["close"].sort_index().rename(tic)


def _load_av_ratios(tic: str) -> pd.DataFrame | None:
    bs_fp = RAW_DIR / f"av_BALANCE_SHEET_{tic}.json"
    is_fp = RAW_DIR / f"av_INCOME_STATEMENT_{tic}.json"
    if not (bs_fp.exists() and is_fp.exists()):
        return None
    try:
        bs = json.loads(bs_fp.read_text())
        inc = json.loads(is_fp.read_text())
    except Exception:
        return None
    bs_q = pd.DataFrame(bs.get("quarterlyReports", []))
    is_q = pd.DataFrame(inc.get("quarterlyReports", []))
    if bs_q.empty or is_q.empty:
        return None

    def _num(s):
        return pd.to_numeric(s, errors="coerce")

    bs_q["fiscalDateEnding"] = pd.to_datetime(bs_q["fiscalDateEnding"])
    is_q["fiscalDateEnding"] = pd.to_datetime(is_q["fiscalDateEnding"])
    for c in ["totalAssets", "totalShareholderEquity", "totalLiabilities"]:
        if c in bs_q.columns:
            bs_q[c] = _num(bs_q[c])
    for c in ["netIncome", "totalRevenue"]:
        if c in is_q.columns:
            is_q[c] = _num(is_q[c])

    merged = pd.merge(
        bs_q[["fiscalDateEnding", "totalAssets", "totalShareholderEquity", "totalLiabilities"]],
        is_q[["fiscalDateEnding", "netIncome", "totalRevenue"]],
        on="fiscalDateEnding", how="inner",
    ).sort_values("fiscalDateEnding").set_index("fiscalDateEnding")

    merged["ni_ttm"] = merged["netIncome"].rolling(4).sum()
    merged["rev_ttm"] = merged["totalRevenue"].rolling(4).sum()

    out = pd.DataFrame(index=merged.index)
    out["ROA"] = merged["ni_ttm"] / merged["totalAssets"]
    out["ROE"] = merged["ni_ttm"] / merged["totalShareholderEquity"]
    out["NPM"] = merged["ni_ttm"] / merged["rev_ttm"]
    out["DE"]  = merged["totalLiabilities"] / merged["totalShareholderEquity"]
    return out.replace([np.inf, -np.inf], np.nan)


def _quarter_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="QS"))


def build_quarterly_panel(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] building quarterly panel for {len(tickers)} tickers")
    price_cols: Dict[str, pd.Series] = {}
    panel_rows: List[Dict] = []
    q_starts = _quarter_starts(pd.Timestamp("2005-01-01"), TRADE_END + pd.Timedelta(days=90))
    kept = 0
    skipped: List[str] = []
    for tic in tickers:
        fund = _load_fundamental(tic)
        px   = _load_prices(tic)
        if fund is None or px is None:
            skipped.append(tic)
            continue
        ratios = _load_av_ratios(tic)
        if ratios is not None and not ratios.empty:
            r_daily = ratios.reindex(fund.index, method="ffill")
            fund = pd.concat([fund, r_daily], axis=1)
        price_cols[tic] = px
        f = fund.reindex(px.index, method="ffill")
        for q in q_starts:
            idx = px.index.searchsorted(q)
            if idx >= len(px.index):
                continue
            d0 = px.index[idx]
            if d0 not in f.index or f.loc[d0].isna().all():
                continue
            q_next = q + pd.DateOffset(months=3)
            idx2 = px.index.searchsorted(q_next)
            if idx2 >= len(px.index):
                continue
            d1 = px.index[idx2]
            p0, p1 = px.loc[d0], px.loc[d1]
            if p0 <= 0 or p1 <= 0 or pd.isna(p0) or pd.isna(p1):
                continue
            row = {"tic": tic, "quarter_start": q, "trade_date": d0}
            for c in f.columns:
                row[c] = float(f.loc[d0, c]) if not pd.isna(f.loc[d0, c]) else np.nan
            row["y_return"] = float(np.log(p1 / p0))
            panel_rows.append(row)
        kept += 1

    prices_wide = pd.DataFrame(price_cols).sort_index()
    panel = pd.DataFrame(panel_rows)
    if skipped:
        print(f"  [skip] {len(skipped)} tickers with missing feature files: "
              f"{skipped[:10]}{'...' if len(skipped)>10 else ''}")
    print(f"[OK] built panel: {len(panel):,} rows, {kept} tickers")
    return panel, prices_wide


# =============================================================================
# Rolling ML (identical)
# =============================================================================
def _train_models(X_tr, y_tr) -> Dict[str, object]:
    return {
        "Linear": LinearRegression().fit(X_tr, y_tr),
        "Ridge":  Ridge(alpha=1.0, random_state=42).fit(X_tr, y_tr),
        "Lasso":  Lasso(alpha=0.01, max_iter=5000, random_state=42).fit(X_tr, y_tr),
        "RF":     RandomForestRegressor(
                     n_estimators=200, max_depth=6, min_samples_leaf=5,
                     random_state=42, n_jobs=-1,
                 ).fit(X_tr, y_tr),
        "GBM":    GradientBoostingRegressor(
                     n_estimators=200, max_depth=3, learning_rate=0.05,
                     random_state=42,
                 ).fit(X_tr, y_tr),
    }


def rolling_predict(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in panel.columns
                    if c not in ("tic", "quarter_start", "trade_date", "y_return")]
    print(f"[INFO] feature set ({len(feature_cols)}): {feature_cols}")

    panel = panel.copy()
    for c in feature_cols:
        s = panel[c]
        med = s.median()
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        panel[c] = s.fillna(med).clip(lo, hi)

    all_quarters = sorted(panel["quarter_start"].unique())
    trade_quarters = [
        q for q in all_quarters
        if (pd.Timestamp(q) + pd.DateOffset(months=3)) > TRADE_START
        and pd.Timestamp(q) <= TRADE_END
    ]
    print(f"[INFO] {len(trade_quarters)} trade quarters: "
          f"{[str(pd.Timestamp(q).date()) for q in trade_quarters]}")

    preds, selections = [], []
    for qt in trade_quarters:
        qt_ts = pd.Timestamp(qt)
        i = all_quarters.index(qt)
        if i < TRAIN_Q + TEST_Q:
            print(f"  [skip] {qt_ts.date()}: not enough history")
            continue

        train_q  = all_quarters[i - TRAIN_Q - TEST_Q : i - TEST_Q]
        val_q    = all_quarters[i - TEST_Q : i]
        test_q   = [qt]

        tr = panel[panel["quarter_start"].isin(train_q)].dropna(subset=["y_return"])
        va = panel[panel["quarter_start"].isin(val_q)].dropna(subset=["y_return"])
        te = panel[panel["quarter_start"].isin(test_q)]

        if len(tr) < 100 or len(va) < 20 or len(te) == 0:
            print(f"  [skip] {qt_ts.date()}: tr={len(tr)}  va={len(va)}  te={len(te)}")
            continue

        X_tr, y_tr = tr[feature_cols].values, tr["y_return"].values
        X_va, y_va = va[feature_cols].values, va["y_return"].values
        X_te       = te[feature_cols].values

        models = _train_models(X_tr, y_tr)
        val_mse = {name: mean_squared_error(y_va, m.predict(X_va))
                   for name, m in models.items()}
        best = min(val_mse, key=val_mse.get)
        y_pred = models[best].predict(X_te)

        for row, yp in zip(te.itertuples(index=False), y_pred):
            preds.append({"quarter_start": pd.Timestamp(row.quarter_start),
                          "trade_date": row.trade_date, "tic": row.tic,
                          "y_pred": float(yp), "chosen_model": best})

        te2 = te.copy(); te2["y_pred"] = y_pred
        te2 = te2.sort_values("y_pred", ascending=False)
        top_k = max(1, int(round(len(te2) * TOP_PCT)))
        for row in te2.head(top_k).itertuples(index=False):
            selections.append({"quarter_start": pd.Timestamp(row.quarter_start),
                               "chosen_model": best, "val_mse": val_mse[best],
                               "n_universe": len(te2), "top_k": top_k,
                               "tic": row.tic, "y_pred": float(row.y_pred)})

        print(f"  [{qt_ts.date()}] best={best}  val_mse={val_mse[best]:.4f}  "
              f"n_universe={len(te2)}  top_k={top_k}")

    return pd.DataFrame(preds), pd.DataFrame(selections)


# =============================================================================
# Portfolio + backtest (identical)
# =============================================================================
def _cov_returns(prices: pd.DataFrame, end_date: pd.Timestamp,
                 lookback: int = COV_LOOKBACK_DAYS) -> Tuple[pd.Series, pd.DataFrame]:
    px_slice = prices.loc[:end_date].iloc[-(lookback + 1):].replace(0, np.nan)
    keep = px_slice.notna().sum() >= max(20, lookback // 2)
    px_slice = px_slice.loc[:, keep]
    rets = np.log(px_slice / px_slice.shift(1)).replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all").fillna(0.0)
    return rets.mean() * 252, rets.cov() * 252


def _mvo_weights(mu: pd.Series, cov: pd.DataFrame, mode: str = "mvo",
                 risk_aversion: float = 5.0) -> pd.Series:
    n = len(mu)
    if mode == "ew" or n == 1:
        return pd.Series(np.ones(n) / n, index=mu.index)
    cov_m = cov.values
    if mode == "mvo":
        grad = lambda w: cov_m @ w * risk_aversion - mu.values
    elif mode == "minvar":
        grad = lambda w: cov_m @ w
    else:
        raise ValueError(mode)
    w = np.ones(n) / n
    lr = 1e-2
    for _ in range(500):
        w = w - lr * grad(w)
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
        rho = rho[-1] if len(rho) else 0
        theta = cssv[rho] / (rho + 1)
        w = np.maximum(w - theta, 0)
    return pd.Series(w, index=mu.index)


def backtest(prices: pd.DataFrame, selections: pd.DataFrame,
             mode: str) -> pd.Series:
    quarters = sorted(selections["quarter_start"].unique())
    all_dates = prices.loc[TRADE_START:TRADE_END].index
    if all_dates.empty:
        raise RuntimeError("prices have no coverage inside trade window")

    equity = pd.Series(index=all_dates, dtype=float)
    equity.iloc[0] = INITIAL_CAPITAL
    shares: Dict[str, float] = {}

    def _rebalance(date, tickers, prev_capital):
        px_row = prices.loc[date, tickers].replace(0, np.nan).dropna()
        px_row = px_row[px_row > 0]
        tickers = px_row.index.tolist()
        if not tickers:
            return {}
        px_sub = prices[tickers].replace(0, np.nan)
        keep = px_sub.loc[:date].tail(COV_LOOKBACK_DAYS).notna().sum() \
                     >= COV_LOOKBACK_DAYS // 2
        tickers = [t for t in tickers if keep.get(t, False)]
        if not tickers:
            return {}
        mu, cov = _cov_returns(prices[tickers], date)
        tickers = [t for t in tickers if t in mu.index]
        if not tickers:
            return {}
        mu, cov = mu.loc[tickers], cov.loc[tickers, tickers]
        w = _mvo_weights(mu, cov, mode=mode).clip(lower=0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(tickers), index=tickers)
        alloc = prev_capital * w
        return {t: float(alloc.loc[t] / px_row.loc[t]) for t in tickers}

    first_q = min(quarters)
    tickers0 = selections[selections["quarter_start"] == first_q]["tic"].tolist()
    # Initial buy: pay buy fee only
    shares = _rebalance(all_dates[0], tickers0,
                        INITIAL_CAPITAL * (1.0 - BUY_FEE))

    for i in range(1, len(all_dates)):
        date = all_dates[i]
        for q in quarters[1:]:
            q_ts = pd.Timestamp(q)
            if all_dates[i - 1] < q_ts <= date:
                px_now = prices.loc[date, list(shares.keys())]
                cur_val = float((pd.Series(shares) * px_now).dropna().sum())
                # Quarterly rebalance: sell old + buy new → full turnover
                # cost = (BUY_FEE + SELL_FEE) × cur_val
                cur_val_after_cost = cur_val * (1.0 - BUY_FEE - SELL_FEE)
                tickers_q = selections[selections["quarter_start"] == q]["tic"].tolist()
                shares = _rebalance(date, tickers_q, cur_val_after_cost)
                break
        px_now = prices.loc[date, list(shares.keys())]
        val = float((pd.Series(shares) * px_now).dropna().sum())
        if val == 0:
            val = equity.iloc[i - 1]
        equity.iloc[i] = val
    return equity.ffill()


# =============================================================================
# Reporting (parametric)
# =============================================================================
def _metrics(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    rets = s.pct_change().dropna()
    final = float(s.iloc[-1])
    ret_pct = (final / s.iloc[0] - 1) * 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    mdd = float(((s - s.cummax()) / s.cummax()).min()) * 100
    return {"final_$": final, "return_%": ret_pct, "sharpe": sharpe, "max_dd_%": mdd}


def _load_benchmark(ticker: str, start, end) -> pd.Series:
    try:
        import yfinance as yf
    except Exception as e:
        print(f"[WARN] yfinance not available: {e}")
        return pd.Series(dtype=float)
    df = yf.download(ticker, start=start, end=end + pd.Timedelta(days=1),
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df["Close"].rename(ticker)


def _load_des_equity(fp: Path) -> pd.Series | None:
    if not fp.exists():
        print(f"[WARN] DES equity not found: {fp}")
        return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    return df["portfolio_equity"].rename("DES")


def main(universe_key: str) -> int:
    t0 = datetime.now()
    if universe_key not in UNIVERSES:
        raise SystemExit(f"unknown universe {universe_key!r}; choose one of {list(UNIVERSES)}")
    U = UNIVERSES[universe_key]
    if U["tickers"] is None and universe_key == "ndx100":
        U["tickers"] = _load_ndx100()
    tickers      = U["tickers"]
    bench_ticker = U["benchmark"]
    des_equity   = U["des_equity"]
    label        = U["label"]
    out_dir      = WS / "baselines" / "dsr_yang" / U["out_dir_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Yang 2018 DSR — universe={label!r} ({len(tickers)} tickers) "
          f"benchmark={bench_ticker!r} ===")

    panel_fp  = out_dir / "quarterly_panel.pkl"
    prices_fp = out_dir / "prices_wide.pkl"
    if panel_fp.exists() and prices_fp.exists():
        print(f"[INFO] loading cached panel/prices from {out_dir}")
        panel  = pd.read_pickle(panel_fp)
        prices = pd.read_pickle(prices_fp)
    else:
        panel, prices = build_quarterly_panel(tickers)
        panel.to_pickle(panel_fp)
        prices.to_pickle(prices_fp)

    preds, selections = rolling_predict(panel)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    selections.to_csv(out_dir / "selections.csv", index=False)
    if selections.empty:
        print("[ERR] no selections produced")
        return 1

    results = {}
    for mode in ("ew", "mvo", "minvar"):
        print(f"[INFO] backtesting mode={mode}")
        eq = backtest(prices, selections, mode=mode)
        eq.to_csv(out_dir / f"equity_{mode}.csv", header=[f"equity_{mode}"])
        results[mode.upper()] = eq

    bench = _load_benchmark(bench_ticker, TRADE_START, TRADE_END)
    if not bench.empty:
        results[bench_ticker] = INITIAL_CAPITAL * bench / bench.iloc[0]
    des = _load_des_equity(des_equity)
    if des is not None:
        des = des.loc[TRADE_START:TRADE_END]
        if not des.empty:
            results["DES"] = INITIAL_CAPITAL * des / des.iloc[0]

    metrics_rows = []
    for name, s in results.items():
        m = _metrics(s.dropna())
        metrics_rows.append({"method": name, **m})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print("\n=== Metrics ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    plt.figure(figsize=(13, 7))
    style = {
        "DES":         ("#1F77B4", 2.4, 1.0),
        "EW":          ("#8CC63F", 1.6, 0.9),
        "MVO":         ("#7F7F7F", 1.6, 0.9),
        "MINVAR":      ("#C44E52", 1.6, 0.9),
        bench_ticker:  ("black",    1.4, 0.9),
    }
    for name, s in results.items():
        colour, lw, alpha = style.get(name, ("#8888ff", 1.2, 0.7))
        plt.plot(s.index, s.values, label=name, linewidth=lw,
                 color=colour, alpha=alpha)
    plt.axhline(INITIAL_CAPITAL, color="grey", linewidth=0.8,
                linestyle="--", label=f"Initial ${INITIAL_CAPITAL:,}")
    plt.title(f"{label} — Yang 2018 Dynamic Stock Recommendation vs Ours (DES)   "
              f"{TRADE_START.date()} ~ {TRADE_END.date()}")
    plt.xlabel("Date"); plt.ylabel("Portfolio value ($, rebased $1M)")
    plt.legend(loc="upper left", fontsize=10); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(out_dir / "baseline_vs_des.png", dpi=150); plt.close()
    print(f"[OK] wrote {out_dir / 'baseline_vs_des.png'}")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Yang 2018 DSR — {label} reproduction\n")
        f.write(f"Trade period: {TRADE_START.date()} ~ {TRADE_END.date()}\n")
        f.write(f"Universe: {len(tickers)} candidates → kept {panel['tic'].nunique()} in panel\n")
        f.write(f"Rolling: {TRAIN_Q}Q train + {TEST_Q}Q validation → predict next 1Q\n")
        f.write(f"Top {TOP_PCT*100:.0f}% selection\n")
        f.write(f"Initial capital: ${INITIAL_CAPITAL:,}\n\n")
        f.write("Metrics:\n")
        f.write(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        f.write("\n")
    print(f"[OK] wrote {out_dir / 'summary.txt'}")
    print(f"[DONE] elapsed {(datetime.now() - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dsr_backtest.py <dow30|sp100|ndx100>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1].lower()))
