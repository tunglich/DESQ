"""US (Dow 30) feature generator.

Adapted from the Taiwan FeatureUS.py pipeline. Produces four feature families
per ticker (no sentiment / no trade) and writes them to feature/:
    tech_trend_{TICKER}.csv, moment_{TICKER}.csv,
    fundamental_{TICKER}.csv, macro_{TICKER}.csv

Differences vs the Taiwan version:
- Prices / market index come from yfinance (^DJI replaces TWA00).
- Fundamentals come from FMP (yfinance fallback); US has no monthly revenue,
  so R_mom and R_yoy are dropped (per task requirement). CMDTY is dropped.
- tech_trend additionally carries raw OHLCV columns because the training-script
  expanders (_expand_tech_trend_features) derive ret/gap/hl_range/vol_ratio20
  from them.
- Each feature file ends with label columns y_10, y_20, y_40, y_60.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import talib
import ta
from sklearn.linear_model import LinearRegression

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature"))
import _us_data as usd

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "feature")  # feature/
MACRO_CSV = os.path.join(THIS_DIR, "MacroFactor.csv")

_last_valid_date = None


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def label(data, period=20):
    """Binary forward label: 1 if forward `period`-day mean return > 0.6%."""
    target = data.shift(-1).rolling(period).apply(np.nanmean, raw=True).dropna().values
    if len(target) < data.shape[0]:
        target = np.append(target, np.zeros(data.shape[0] - len(target)) + np.nan)
    return (target > data * 1.006).astype(int)


# --------------------------------------------------------------------------- #
# Indicator helpers (ported unchanged from FeatureUS.py)
# --------------------------------------------------------------------------- #
def bias(data, period_f=5, period_s=20):
    return (data / data.rolling(period_f, min_periods=1).mean()).fillna(0) - \
           (data / data.rolling(period_s, min_periods=1).mean()).fillna(0)


def hullma(data, period=60):
    period_2 = period // 2
    period_sqrt = np.floor(np.sqrt(period))
    wma1 = talib.EMA(data, timeperiod=period_2).fillna(0)
    wma2 = talib.EMA(data, timeperiod=period).fillna(0)
    return ((data / talib.EMA(wma1 * 2 - wma2, timeperiod=period_sqrt).fillna(0) - 1) * 100) \
        .fillna(0).replace([np.inf, -np.inf], 0)


def mmi(data, period):
    median = data.rolling(period).median()
    return ((data > median) & (data.shift() > median)).rolling(period).mean().fillna(0)


def sma(data, period=20):
    return ((data / talib.SMA(data, timeperiod=period) - 1) * 100).fillna(0)


def macd(data, n_fast=20, n_slow=50, n_sign=20):
    return ta.trend.macd_diff(data, window_fast=n_fast, window_slow=n_slow,
                              window_sign=n_sign, fillna=False).fillna(0)


def bb(data, period=20, times=2):
    up_bb = ta.volatility.bollinger_hband(data, window=period, window_dev=times, fillna=False)
    down_bb = ta.volatility.bollinger_lband(data, window=period, window_dev=times, fillna=False)
    return ((data - down_bb) / (up_bb - down_bb)).fillna(0)


def aroon_osc(d1, d2, period=14):
    return talib.AROONOSC(d1, d2, timeperiod=period).fillna(0)


def stoch(close, high, low, fastk=20, slowk=10, slowd=10):
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=fastk,
                               slowk_period=slowk, slowd_period=slowd)
    return slowk, slowd


def wr(close, high, low, period=20):
    return talib.WILLR(high, low, close, timeperiod=period).iloc[period - 1:].fillna(0)


def rsi(data, period=20):
    return talib.RSI(data, timeperiod=period).iloc[period:]


def cci(close, high, low, period=14):
    return talib.CCI(high, low, close, timeperiod=period).fillna(0)


def acc(data, period=20):
    return (data.shift(period) / (data.shift(2 * period) + data) * 2).fillna(0)


def adx(close, high, low, period=14):
    return talib.ADX(high, low, close, timeperiod=period).fillna(0)


def vpt(close, volume):
    return ta.volume.volume_price_trend(close, volume)


def alpha_beta(close, market, rolling=90):
    """Rolling regression of stock returns on market (^DJI) returns."""
    market_return = market.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    close_return = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    obs = len(market_return)
    betas = np.full(obs, np.nan)
    alphas = np.full(obs, np.nan)
    m = market_return.to_numpy()
    c = close_return.to_numpy()
    for i in range(obs - rolling):
        reg = LinearRegression()
        reg.fit(m[i:i + rolling + 1].reshape(-1, 1), c[i:i + rolling + 1])
        betas[i + rolling] = round(reg.coef_[0], 6)
        alphas[i + rolling] = round(reg.intercept_ * 100, 6)
    results = pd.DataFrame({"alpha": alphas, "beta": betas}, index=close.index)
    results = results.sort_index().bfill().fillna(0)
    return results


# --------------------------------------------------------------------------- #
# Feature families
# --------------------------------------------------------------------------- #
def make_tech_trend(close, high, low, volume, market, ticker):
    c = close[ticker]
    h = high[ticker]
    lo = low[ticker]
    v = volume[ticker]
    ab = alpha_beta(c, market, 90)
    out = pd.DataFrame({
        "sma_5": sma(c, 5), "sma_10": sma(c, 10), "sma_20": sma(c, 20),
        "sma_60": sma(c, 60), "sma_120": sma(c, 120),
        "hullma_20": hullma(c, 20), "hullma_60": hullma(c, 60),
        "hullma_120": hullma(c, 120),
        "mmi_5": mmi(c, 5), "mmi_10": mmi(c, 10), "mmi_20": mmi(c, 20),
        "aroon_osc": aroon_osc(h, lo, 14),
        "osc": macd(c),
        "bb": bb(c),
        "bias": bias(c),
        "alpha": ab["alpha"],
    })
    # raw OHLCV required by the training-script tech_trend expander
    out["open"] = _open[ticker]
    out["high"] = h
    out["low"] = lo
    out["close"] = c
    out["volume"] = v
    out["y_10"] = label(c, 10)
    out["y_20"] = label(c, 20)
    out["y_40"] = label(c, 40)
    out["y_60"] = label(c, 60)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_moment(close, high, low, volume, market, ticker):
    c = close[ticker]
    h = high[ticker]
    lo = low[ticker]
    v = volume[ticker]
    k, d = stoch(c, h, lo)
    ab = alpha_beta(c, market, 90)
    out = pd.DataFrame({
        "rsi": rsi(c),
        "k": k, "d": d,
        "wr": wr(c, h, lo),
        "cci": cci(c, h, lo),
        "adx": adx(c, h, lo),
        "acc_5": acc(c, 5), "acc_10": acc(c, 10), "acc_20": acc(c, 20),
        "acc_60": acc(c, 60), "acc_120": acc(c, 120),
        "vpt": vpt(c, v),
        "beta": ab["beta"],
        "y_10": label(c, 10),
        "y_20": label(c, 20),
        "y_40": label(c, 40),
        "y_60": label(c, 60),
    })
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def _pct_signed(series):
    """qoq/yoy % change robust to sign: (cur-prev)/|prev|*100."""
    prev = series.shift(1)
    return (series - prev) / prev.abs() * 100


def _pct_n(series, n):
    prev = series.shift(n)
    return (series - prev) / prev.abs() * 100


def make_fundamental(close, market, ticker, fund):
    """Build fundamental features from FMP quarterly data.

    US has no monthly revenue → R_mom, R_yoy dropped. CMDTY dropped.
    Columns: PE_trailing, PEG, PBR, DY, R_acc_yoy, E_qoq, E_yoy, E_acc_yoy,
             Op_qoq, Op_yoy, Op_acc_yoy, Gross, Gross_qoq, EPS_qoq + labels.
    """
    c = close[ticker].copy()
    idx = c.index

    income = fund["income"]
    metrics = fund["metrics"]
    dividends = fund["dividends"]

    # Statement-based metrics must be computed on consecutive *true* quarters,
    # so split the income-statement rows (revenue present) from the EPS-only
    # rows (deeper EARNINGS history) before taking diffs / rolling sums.
    q = pd.DataFrame()
    if not income.empty:
        eps_col = "eps" if "eps" in income.columns else (
            "epsDiluted" if "epsDiluted" in income.columns else None)

        stmt_cols = [c_ for c_ in ("revenue", "grossProfit", "operatingIncome",
                                   "incomeBeforeTax") if c_ in income.columns]
        stmt = income[stmt_cols].dropna(how="all").sort_index() if stmt_cols else pd.DataFrame()
        stmt = stmt[~stmt.index.duplicated(keep="last")]

        sq = pd.DataFrame(index=stmt.index)
        rev = stmt.get("revenue")
        gp = stmt.get("grossProfit")
        op = stmt.get("operatingIncome")
        pretax = stmt.get("incomeBeforeTax")

        # Gross margin (%) and its qoq change
        if gp is not None and rev is not None:
            gross = gp / rev.replace(0, np.nan) * 100
            sq["Gross"] = gross
            sq["Gross_qoq"] = gross.diff()

        # Revenue accumulated (TTM) yoy
        if rev is not None:
            sq["R_acc_yoy"] = _pct_n(rev.rolling(4).sum(), 4)

        # Pretax income (E_*)
        if pretax is not None:
            sq["E_qoq"] = _pct_signed(pretax)
            sq["E_yoy"] = _pct_n(pretax, 4)
            sq["E_acc_yoy"] = _pct_n(pretax.rolling(4).sum(), 4)

        # Operating income (Op_*)
        if op is not None:
            sq["Op_qoq"] = _pct_signed(op)
            sq["Op_yoy"] = _pct_n(op, 4)
            sq["Op_acc_yoy"] = _pct_n(op.rolling(4).sum(), 4)

        # EPS metrics on the (deeper) clean EPS quarterly series
        eq = pd.DataFrame()
        if eps_col is not None:
            eps_q = income[eps_col].dropna().sort_index()
            eps_q = eps_q[~eps_q.index.duplicated(keep="last")]
            eq = pd.DataFrame(index=eps_q.index)
            eq["EPS_qoq"] = _pct_signed(eps_q)
            eps_ttm = eps_q.rolling(4).sum()
            eq["_eps_ttm"] = eps_ttm
            eq["_eps_growth"] = _pct_n(eps_ttm, 4)

        # Merge the two quarterly frames on the union of their report dates
        q = sq.join(eq, how="outer") if not eq.empty else sq

    # Reindex quarterly → daily (ffill)
    q_daily = q.reindex(idx.union(q.index)).sort_index().ffill().reindex(idx)

    out = pd.DataFrame(index=idx)

    # PE_trailing: river level of daily PE over trailing ~3y (750d) window
    if "_eps_ttm" in q_daily.columns:
        eps_ttm_d = q_daily["_eps_ttm"].replace(0, np.nan)
        pe = (c / eps_ttm_d).replace([np.inf, -np.inf], np.nan)
        pe = pe.where(pe > 0)
        rmin = pe.rolling(750, min_periods=60).min()
        rmax = pe.rolling(750, min_periods=60).max()
        out["PE_trailing"] = ((pe - rmin) / (rmax - rmin)).clip(0, 1)

        eps_forecast = q_daily["_eps_ttm"] * (1 + q_daily["_eps_growth"] / 100)
        pe_forecast = c / eps_forecast.replace(0, np.nan)
        out["PEG"] = (pe_forecast / q_daily["_eps_growth"].replace(0, np.nan))

    # PBR: river level of daily price-to-book
    if not metrics.empty and "bookValuePerShare" in metrics.columns:
        bvps = metrics["bookValuePerShare"].reindex(idx.union(metrics.index)) \
            .sort_index().ffill().reindex(idx).replace(0, np.nan)
        pb = (c / bvps).replace([np.inf, -np.inf], np.nan)
        pb = pb.where(pb > 0)
        rmin = pb.rolling(750, min_periods=60).min()
        rmax = pb.rolling(750, min_periods=60).max()
        out["PBR"] = ((pb - rmin) / (rmax - rmin)).clip(0, 1)

    # DY: trailing 12-month dividends / price (%)
    if not dividends.empty:
        div = dividends[~dividends.index.duplicated(keep="last")].sort_index()
        div_daily = div.reindex(idx.union(div.index)).sort_index().fillna(0).reindex(idx)
        ttm_div = div_daily.rolling(252, min_periods=1).sum()
        out["DY"] = (ttm_div / c * 100).replace([np.inf, -np.inf], 0)
    elif not metrics.empty and "dividendYield" in metrics.columns:
        dy = metrics["dividendYield"].reindex(idx.union(metrics.index)) \
            .sort_index().ffill().reindex(idx)
        out["DY"] = (dy * 100)
    else:
        # No dividend history (e.g. AMZN) — keep column for cross-ticker consistency
        out["DY"] = 0.0

    # Growth / margin columns
    for col in ("R_acc_yoy", "E_qoq", "E_yoy", "E_acc_yoy",
                "Op_qoq", "Op_yoy", "Op_acc_yoy", "Gross", "Gross_qoq", "EPS_qoq"):
        if col in q_daily.columns:
            out[col] = q_daily[col]

    # Order to match training-script expectations (bounded first, then growth)
    preferred = ["PE_trailing", "PEG", "PBR", "DY", "R_acc_yoy",
                 "E_qoq", "E_yoy", "E_acc_yoy", "Op_qoq", "Op_yoy", "Op_acc_yoy",
                 "Gross", "Gross_qoq", "EPS_qoq"]
    cols = [c_ for c_ in preferred if c_ in out.columns]
    out = out[cols]

    out["y_10"] = label(c, 10)
    out["y_20"] = label(c, 20)
    out["y_40"] = label(c, 40)
    out["y_60"] = label(c, 60)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def make_macro(close, ticker, macro_df):
    """Reindex the prepared MacroFactor.csv onto the ticker trading dates."""
    c = close[ticker]
    processed = macro_df.reindex(c.index).ffill().fillna(0)
    processed = processed.copy()
    processed["y_10"] = label(c, 10)
    processed["y_20"] = label(c, 20)
    processed["y_40"] = label(c, 40)
    processed["y_60"] = label(c, 60)
    return processed.replace([np.inf, -np.inf], 0)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_open = None  # populated in main(); used by make_tech_trend


def main(tickers=None, market_index=None):
    global _open
    if tickers is None:
        tickers = usd.DOW_30_TICKER
    if market_index is None:
        market_index = os.getenv("MARKET_INDEX", usd.MARKET_INDEX)

    print(f"[1/3] loading prices + {market_index} ...")
    frames = usd.load_price_frames(tickers)
    market = usd.load_market_index(index=market_index)
    df_open = frames["Open"]
    df_close = frames["Close"]
    df_high = frames["High"]
    df_low = frames["Low"]
    df_volume = frames["Volume"]
    _open = df_open
    market = market.reindex(df_close.index).ffill()

    print("[2/3] loading macro ...")
    macro_df = pd.read_csv(MACRO_CSV, index_col=0, parse_dates=True)

    print("[3/3] generating features ...")
    for tk in tickers:
        if tk not in df_close.columns:
            print(f"  [SKIP] {tk}: no price data")
            continue
        try:
            tt = make_tech_trend(df_close, df_high, df_low, df_volume, market, tk)
            tt.to_csv(os.path.join(OUT_DIR, f"tech_trend_{tk}.csv"))

            mo = make_moment(df_close, df_high, df_low, df_volume, market, tk)
            mo.to_csv(os.path.join(OUT_DIR, f"moment_{tk}.csv"))

            ma = make_macro(df_close, tk, macro_df)
            ma.to_csv(os.path.join(OUT_DIR, f"macro_{tk}.csv"))

            fund = usd.load_fundamentals(tk)
            fu = make_fundamental(df_close, market, tk, fund)
            fu.to_csv(os.path.join(OUT_DIR, f"fundamental_{tk}.csv"))

            print(f"  [OK] {tk}: tt{tt.shape} mo{mo.shape} ma{ma.shape} fu{fu.shape}")
        except Exception as exc:
            print(f"  [ERR] {tk}: {exc}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    # Preset shortcuts: --dow / --sox select the matching ticker list and
    # benchmark index. Bare tickers (e.g. AAPL MSFT) still work as before.
    market_override = None
    tickers_override = None
    remaining = []
    for a in args:
        low = a.lower()
        if low == "--sox":
            tickers_override = list(usd.SOX_30_TICKER)
            market_override = usd.SOX_INDEX
        elif low == "--dow":
            tickers_override = list(usd.DOW_30_TICKER)
            market_override = usd.MARKET_INDEX
        else:
            remaining.append(a)
    if tickers_override is None and remaining:
        tickers_override = remaining
    main(tickers=tickers_override, market_index=market_override)
