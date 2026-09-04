"""Phase B — Reproduce Yang, Liu, Wu (2018) — "A Practical Machine Learning
Approach for Dynamic Stock Recommendation" — on our S&P 100 universe.

Methodology follows the reference approach; the feature set is adapted to the
compute from Alpha Vantage):

    1. Quarterly panel (tic, quarter_start) of fundamental features + target
       label = next-quarter log-return.
    2. Rolling window: for each trade quarter, train 5 ML regressors
       (Linear, Ridge, Lasso, RandomForest, GradientBoosting) on the last
       16 quarters, evaluate on the trailing 4 quarters, keep the model with
       the lowest MSE and use it to predict the coming quarter.
    3. Rank all predictions; select the top 20% (top ~20 of ~99) each quarter.
    4. Allocate with three schemes:
       - EW    : equal-weight
       - MVO   : classical mean-variance (long-only, sum=1)
       - MinVar: minimum-variance         (long-only, sum=1)
       using a trailing 63-day covariance estimate.
    5. Backtest 2024-01-02 ~ 2026-03-30 with quarterly rebalance, initial
       capital = $1,000,000, benchmark = ^OEX; overlay our DES SP100 curve.

Outputs (under baselines/dsr_yang/backtest_2024_20260330/):
    quarterly_panel.parquet    ......  panel dataset used for training
    predictions.csv            ......  per-quarter predicted returns
    selections.csv             ......  per-quarter top-20 tickers per model pick
    equity_ew.csv / _mvo.csv / _minvar.csv .... daily portfolio equity
    metrics.csv                ......  return / sharpe / mdd per method
    baseline_vs_des.png        ......  DES vs 3 Yang variants vs ^OEX
    summary.txt                ......  human-readable summary
"""
from __future__ import annotations

import json
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
# Configuration
# =============================================================================
WS = Path(r"d:\US_stock")
FEATURE_DIR = WS / "feature"
RAW_DIR = FEATURE_DIR / "_raw"
OUT_DIR = WS / "baselines" / "dsr_yang" / "backtest_2024_20260330"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DES_EQUITY = WS / "backtest_portfolio_US" / "equity_sp100_market_2024-01-02_2026-03-31.csv"
BENCH_TICKER = "^OEX"

TRADE_START = pd.Timestamp("2024-01-02")
TRADE_END   = pd.Timestamp("2026-03-30")
INITIAL_CAPITAL = 1_000_000

# Rolling-window parameters in quarters.
TRAIN_Q = 16   # 4 years training
TEST_Q  = 4    # 1 year validation used for model selection

# Portfolio params
TOP_PCT   = 0.20        # buy top 20 %
COV_LOOKBACK_DAYS = 63  # trailing quarter for covariance

# SP100 tickers (from _sp100_diff.SP100; keep in sync)
SP100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AMAT","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BKNG","BLK","BMY","BNY","BRK.B","C",
    "CAT","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GEV","GILD",
    "GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG",
    "JNJ","JPM","KO","LIN","LLY","LMT","LOW","LRCX","MA","MCD",
    "MDLZ","MDT","META","MMM","MO","MRK","MS","MSFT","MU","NEE",
    "NFLX","NKE","NOW","NVDA","ORCL","PEP","PFE","PG","PLTR","PM",
    "QCOM","RTX","SBUX","SCHW","SO","SPG","T","TMO","TMUS","TSLA",
    "TXN","UBER","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM",
]


# =============================================================================
# Data loading
# =============================================================================
FUND_COLS = [
    "PE_trailing", "PEG", "PBR", "DY",
    "R_acc_yoy",
    "E_qoq", "E_yoy", "E_acc_yoy",
    "Op_qoq", "Op_yoy", "Op_acc_yoy",
    "Gross", "Gross_qoq",
    "EPS_qoq",
]


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
    """Compute quarterly ROA / ROE / NPM / DE from cached AV JSON blobs.

    Returns a DataFrame indexed by *fiscalDateEnding* (quarter-end date).
    """
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

    # TTM net income / revenue for stability
    merged["ni_ttm"] = merged["netIncome"].rolling(4).sum()
    merged["rev_ttm"] = merged["totalRevenue"].rolling(4).sum()

    out = pd.DataFrame(index=merged.index)
    out["ROA"] = merged["ni_ttm"] / merged["totalAssets"]
    out["ROE"] = merged["ni_ttm"] / merged["totalShareholderEquity"]
    out["NPM"] = merged["ni_ttm"] / merged["rev_ttm"]
    out["DE"]  = merged["totalLiabilities"] / merged["totalShareholderEquity"]
    return out.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# Quarterly panel builder
# =============================================================================
def _quarter_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="QS"))


def build_quarterly_panel(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (panel, prices_wide).

    panel columns: tic, quarter_start, <features>, y_return (log-return of
                   next-quarter close vs current-quarter close, on first
                   trading day of each quarter).
    prices_wide : DataFrame indexed by Date, columns = tickers, values = close.
    """
    print(f"[INFO] building quarterly panel for {len(tickers)} tickers")
    price_cols: Dict[str, pd.Series] = {}
    panel_rows: List[Dict] = []

    # Universal quarter starts across a broad window (2005..2026)
    q_starts = _quarter_starts(pd.Timestamp("2005-01-01"), TRADE_END + pd.Timedelta(days=90))

    kept = 0
    for tic in tickers:
        fund = _load_fundamental(tic)
        px   = _load_prices(tic)
        if fund is None or px is None:
            print(f"  [skip] {tic}: missing feature files")
            continue

        # Merge AV-derived ratios (ROA/ROE/NPM/DE)
        ratios = _load_av_ratios(tic)
        if ratios is not None and not ratios.empty:
            # Forward-fill onto daily index of `fund`
            r_daily = ratios.reindex(fund.index, method="ffill")
            fund = pd.concat([fund, r_daily], axis=1)

        price_cols[tic] = px

        # Merge features to the same daily index as prices, forward-fill
        # (fundamentals are step functions between reports).
        f = fund.reindex(px.index, method="ffill")

        for q in q_starts:
            # First trading day on/after q
            idx = px.index.searchsorted(q)
            if idx >= len(px.index):
                continue
            d0 = px.index[idx]
            # Skip if before the ticker's earliest fundamental record
            if d0 not in f.index or f.loc[d0].isna().all():
                continue

            # Next-quarter start
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
    print(f"[OK] built panel: {len(panel):,} rows, {kept} tickers, "
          f"{panel.columns.tolist()[:12]} ...")
    return panel, prices_wide


# =============================================================================
# Rolling ML — Yang 2018 style model selection
# =============================================================================
def _train_models(X_tr, y_tr) -> Dict[str, object]:
    return {
        "Linear": LinearRegression().fit(X_tr, y_tr),
        "Ridge":  Ridge(alpha=1.0, random_state=42).fit(X_tr, y_tr),
        "Lasso":  Lasso(alpha=0.01, max_iter=5000, random_state=42).fit(X_tr, y_tr),
        "RF":     RandomForestRegressor(
                     n_estimators=200, max_depth=6, min_samples_leaf=5,
                     random_state=42, n_jobs=-1
                 ).fit(X_tr, y_tr),
        "GBM":    GradientBoostingRegressor(
                     n_estimators=200, max_depth=3, learning_rate=0.05,
                     random_state=42
                 ).fit(X_tr, y_tr),
    }


def rolling_predict(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Yang 2018-style rolling ML.

    For every trade quarter q_t that is within TRADE_START..TRADE_END:
      - Train on quarters [q_{t-TEST_Q-TRAIN_Q}, q_{t-TEST_Q})
      - Validate on quarters [q_{t-TEST_Q}, q_t)
      - Pick model with lowest validation MSE
      - Predict on quarter q_t itself
    """
    feature_cols = [c for c in panel.columns
                    if c not in ("tic", "quarter_start", "trade_date", "y_return")]
    print(f"[INFO] feature set ({len(feature_cols)}): {feature_cols}")

    # Preprocess: fill NaN, robust scaling per feature (winsorise)
    panel = panel.copy()
    for c in feature_cols:
        s = panel[c]
        med = s.median()
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        panel[c] = s.fillna(med).clip(lo, hi)

    all_quarters = sorted(panel["quarter_start"].unique())
    # Include quarters whose *trading period* overlaps [TRADE_START, TRADE_END].
    # A quarter q trades from its first trading day (≈ q) until the next
    # quarter start (~ q + 3 months).  So include q iff q + 3mo > TRADE_START
    # AND q < TRADE_END.
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
        # Index qt in all_quarters
        i = all_quarters.index(qt)
        if i < TRAIN_Q + TEST_Q:
            print(f"  [skip] {qt_ts.date()}: not enough history "
                  f"(i={i}, need {TRAIN_Q+TEST_Q})")
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
            preds.append({
                "quarter_start": pd.Timestamp(row.quarter_start),
                "trade_date":    row.trade_date,
                "tic":           row.tic,
                "y_pred":        float(yp),
                "chosen_model":  best,
            })

        # Log top-K selection
        te2 = te.copy()
        te2["y_pred"] = y_pred
        te2 = te2.sort_values("y_pred", ascending=False)
        top_k = max(1, int(round(len(te2) * TOP_PCT)))
        for row in te2.head(top_k).itertuples(index=False):
            selections.append({
                "quarter_start": pd.Timestamp(row.quarter_start),
                "chosen_model":  best,
                "val_mse":       val_mse[best],
                "n_universe":    len(te2),
                "top_k":         top_k,
                "tic":           row.tic,
                "y_pred":        float(row.y_pred),
            })

        print(f"  [{qt_ts.date()}] best={best}  val_mse={val_mse[best]:.4f}  "
              f"n_universe={len(te2)}  top_k={top_k}")

    return pd.DataFrame(preds), pd.DataFrame(selections)


# =============================================================================
# Portfolio allocation
# =============================================================================
def _cov_returns(prices: pd.DataFrame, end_date: pd.Timestamp,
                 lookback: int = COV_LOOKBACK_DAYS) -> Tuple[pd.Series, pd.DataFrame]:
    # Replace zero prices (missing / pre-IPO padding) with NaN so log-returns
    # don't blow up.
    px_slice = prices.loc[:end_date].iloc[-(lookback + 1):].replace(0, np.nan)
    # Keep tickers with at least half the lookback populated
    keep = px_slice.notna().sum() >= max(20, lookback // 2)
    px_slice = px_slice.loc[:, keep]
    rets = np.log(px_slice / px_slice.shift(1)).replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all")
    # Fill remaining single-day gaps with 0 to keep covariance well-defined
    rets = rets.fillna(0.0)
    return rets.mean() * 252, rets.cov() * 252


def _mvo_weights(mu: pd.Series, cov: pd.DataFrame,
                 mode: str = "mvo", risk_aversion: float = 5.0) -> pd.Series:
    """Long-only, sum=1 weights.  mode ∈ {mvo, minvar, ew}.

    We solve small QPs via a projected gradient descent (simplex projection)
    to avoid a hard cvxpy dependency.
    """
    n = len(mu)
    if mode == "ew" or n == 1:
        return pd.Series(np.ones(n) / n, index=mu.index)

    cov_m = cov.values
    if mode == "mvo":
        # maximise  mu.T w - 0.5 * risk_aversion * w.T cov w
        grad = lambda w: cov_m @ w * risk_aversion - mu.values
    elif mode == "minvar":
        grad = lambda w: cov_m @ w
    else:
        raise ValueError(mode)

    w = np.ones(n) / n
    lr = 1e-2
    for _ in range(500):
        w = w - lr * grad(w)
        # project onto simplex {w>=0, sum=1}
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
        rho = rho[-1] if len(rho) else 0
        theta = cssv[rho] / (rho + 1)
        w = np.maximum(w - theta, 0)
    return pd.Series(w, index=mu.index)


# =============================================================================
# Backtest engine
# =============================================================================
def backtest(prices: pd.DataFrame, selections: pd.DataFrame,
             mode: str) -> pd.Series:
    """Daily equity, quarterly rebalance to top-K from `selections`."""
    quarters = sorted(selections["quarter_start"].unique())
    all_dates = prices.loc[TRADE_START:TRADE_END].index
    if all_dates.empty:
        raise RuntimeError("prices have no coverage inside trade window")

    equity = pd.Series(index=all_dates, dtype=float)
    equity.iloc[0] = INITIAL_CAPITAL
    shares: Dict[str, float] = {}

    def _rebalance(date, tickers, prev_capital):
        # Drop tickers with zero / NaN / negative price on the rebalance day
        px_row = prices.loc[date, tickers].replace(0, np.nan).dropna()
        px_row = px_row[px_row > 0]
        tickers = px_row.index.tolist()
        if not tickers:
            return {}
        # Restrict covariance universe to tickers with reliable history
        px_sub = prices[tickers].replace(0, np.nan)
        keep = px_sub.loc[:date].tail(COV_LOOKBACK_DAYS).notna().sum() \
                     >= COV_LOOKBACK_DAYS // 2
        tickers = [t for t in tickers if keep.get(t, False)]
        if not tickers:
            return {}
        mu, cov = _cov_returns(prices[tickers], date)
        # Ensure alignment
        tickers = [t for t in tickers if t in mu.index]
        if not tickers:
            return {}
        mu, cov = mu.loc[tickers], cov.loc[tickers, tickers]
        w = _mvo_weights(mu, cov, mode=mode)
        w = w.clip(lower=0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(tickers),
                                                       index=tickers)
        alloc = prev_capital * w
        return {t: float(alloc.loc[t] / px_row.loc[t]) for t in tickers}

    # First rebalance on TRADE_START
    first_q = min(quarters)
    tickers0 = selections[selections["quarter_start"] == first_q]["tic"].tolist()
    shares = _rebalance(all_dates[0], tickers0, INITIAL_CAPITAL)

    for i in range(1, len(all_dates)):
        date = all_dates[i]
        # Rebalance on next available date >= quarter_start (skip first)
        for q in quarters[1:]:
            q_ts = pd.Timestamp(q)
            if all_dates[i - 1] < q_ts <= date:
                # Value current portfolio
                px_now = prices.loc[date, list(shares.keys())]
                cur_val = float((pd.Series(shares) * px_now).dropna().sum())
                tickers_q = selections[selections["quarter_start"] == q]["tic"].tolist()
                shares = _rebalance(date, tickers_q, cur_val)
                break

        px_now = prices.loc[date, list(shares.keys())]
        val = float((pd.Series(shares) * px_now).dropna().sum())
        # Forward-fill zero holdings if any ticker went missing that day
        if val == 0:
            val = equity.iloc[i - 1]
        equity.iloc[i] = val

    return equity.ffill()


# =============================================================================
# Reporting
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


def _load_des_equity() -> pd.Series | None:
    if not DES_EQUITY.exists():
        print(f"[WARN] DES equity not found: {DES_EQUITY}")
        return None
    df = pd.read_csv(DES_EQUITY, index_col=0, parse_dates=True)
    return df["portfolio_equity"].rename("DES")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    t0 = datetime.now()

    panel_fp  = OUT_DIR / "quarterly_panel.pkl"
    prices_fp = OUT_DIR / "prices_wide.pkl"
    if panel_fp.exists() and prices_fp.exists():
        print(f"[INFO] loading cached panel/prices from {OUT_DIR}")
        panel  = pd.read_pickle(panel_fp)
        prices = pd.read_pickle(prices_fp)
    else:
        panel, prices = build_quarterly_panel(SP100)
        panel.to_pickle(panel_fp)
        prices.to_pickle(prices_fp)

    preds, selections = rolling_predict(panel)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    selections.to_csv(OUT_DIR / "selections.csv", index=False)

    # Only proceed if we have selections for the trade window
    if selections.empty:
        print("[ERR] no selections produced")
        return 1

    results = {}
    for mode in ("ew", "mvo", "minvar"):
        print(f"[INFO] backtesting mode={mode}")
        eq = backtest(prices, selections, mode=mode)
        eq.to_csv(OUT_DIR / f"equity_{mode}.csv", header=[f"equity_{mode}"])
        results[mode.upper()] = eq

    # Load benchmark and DES
    bench = _load_benchmark(BENCH_TICKER, TRADE_START, TRADE_END)
    if not bench.empty:
        bench_rebased = INITIAL_CAPITAL * bench / bench.iloc[0]
        results[BENCH_TICKER] = bench_rebased
    des = _load_des_equity()
    if des is not None:
        des = des.loc[TRADE_START:TRADE_END]
        if not des.empty:
            results["DES"] = INITIAL_CAPITAL * des / des.iloc[0]

    # Metrics table
    metrics_rows = []
    for name, s in results.items():
        m = _metrics(s.dropna())
        m = {"method": name, **m}
        metrics_rows.append(m)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\n=== Metrics ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # Plot
    plt.figure(figsize=(13, 7))
    style = {
        "DES":         ("#1F77B4", 2.4, 1.0),
        "EW":          ("#8CC63F", 1.6, 0.9),
        "MVO":         ("#7F7F7F", 1.6, 0.9),
        "MINVAR":      ("#C44E52", 1.6, 0.9),
        BENCH_TICKER:  ("black",    1.4, 0.9),
    }
    for name, s in results.items():
        colour, lw, alpha = style.get(name, ("#8888ff", 1.2, 0.7))
        plt.plot(s.index, s.values, label=name, linewidth=lw,
                 color=colour, alpha=alpha)
    plt.axhline(INITIAL_CAPITAL, color="grey", linewidth=0.8,
                linestyle="--", label=f"Initial ${INITIAL_CAPITAL:,}")
    plt.title(f"S&P 100 — Yang 2018 Dynamic Stock Recommendation  "
              f"vs Ours (DES)   "
              f"{TRADE_START.date()} ~ {TRADE_END.date()}")
    plt.xlabel("Date"); plt.ylabel("Portfolio value ($, rebased $1M)")
    plt.legend(loc="upper left", fontsize=10); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_vs_des.png", dpi=150); plt.close()
    print(f"[OK] wrote {OUT_DIR / 'baseline_vs_des.png'}")

    # Summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Yang 2018 DSR — S&P 100 reproduction\n")
        f.write(f"Trade period: {TRADE_START.date()} ~ {TRADE_END.date()}\n")
        f.write(f"Universe: {SP100!r} ({len(SP100)} candidates)\n")
        f.write(f"Kept in panel: {panel['tic'].nunique()} tickers\n")
        f.write(f"Feature set: {[c for c in panel.columns if c not in ('tic','quarter_start','trade_date','y_return')]}\n")
        f.write(f"Rolling: {TRAIN_Q}Q train + {TEST_Q}Q validation → predict next 1Q\n")
        f.write(f"Top {TOP_PCT*100:.0f}% selection\n")
        f.write(f"Initial capital: ${INITIAL_CAPITAL:,}\n\n")
        f.write("Metrics:\n")
        f.write(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        f.write("\n")
    print(f"[OK] wrote {OUT_DIR / 'summary.txt'}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[DONE] elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
