from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from feature._us_data import DOW_30_TICKER, SOX_30_TICKER
from _sp100_diff import SP100

ROOT = Path(__file__).resolve().parent
DES_PRED_ROOT = ROOT / "model_pred_DES_US"
CUSUM6_ROOT = ROOT / "cumSum_prob_6"
CUSUM12_ROOT = ROOT / "cumSum_prob_12"
OUT_ROOT = ROOT / "AI_us"
SNAPSHOT_ROOT = OUT_ROOT / "snapshots"
SUMMARY_ROOT = OUT_ROOT / "summary"
SCORE_ROOT = OUT_ROOT / "score"

# If both share classes exist, keep the preferred ticker only.
SHARE_CLASS_ALIAS = {
    "GOOG": "GOOGL",
}

# Tickers whose DES model degenerated to `classes_=[0]` (single-class output),
# so `predict_proba` always returns 0 and they drag down the weighted AI score.
# Root cause is upstream in DES training: the walk-forward window contained no
# positive labels (`y_20 > 0.6%` never triggered). Fix by retraining via
# `us-stock-pipeline` with a longer window or lower label threshold; verify
# with `_diag_des_zero.py`. Until then, exclude from AI score aggregation.
#
# 2026-07-05 batch retrain (_des_retrain_only.py) surfaced the full list of
# 16 tickers with degenerate training windows. Most are recent IPOs (ARM,
# PLTR, DDOG, RKLB, NBIS, DASH, GEHC, ALAB, CRWD, APP, CEG, SNDK, ALGM, WOLF)
# where the label horizon rarely crossed +0.6% in 20 days; KHC and PDD are
# unusual (older listings) — recheck their y_20 threshold or window.
#
# 2026-07-17: GEV added — GE Vernova IPO'd 2024-04-02 so it has only ~500
# usable trading days; `FeatureUS_US.py` back-fills pre-IPO rows with 0.0
# (not NaN), which corrupts scaler fits and makes ATT/DES training fail
# silently. Excluded from aggregation until the feature pipeline is fixed
# to emit NaN for pre-IPO rows (or GEV accumulates enough post-IPO history).
DEGENERATE_DES_TICKERS: set[str] = {
    "ALAB", "ALGM", "APP", "ARM", "CEG", "CRWD", "DASH", "DDOG",
    "GEHC", "GEV", "KHC", "NBIS", "PDD", "PLTR", "RKLB", "SNDK", "WOLF",
}

for d in (OUT_ROOT, SNAPSHOT_ROOT, SUMMARY_ROOT, SCORE_ROOT):
    d.mkdir(parents=True, exist_ok=True)


def _to_file_safe(ticker: str) -> str:
    return str(ticker).upper().replace(".", "-")


def _load_ndx100_from_file() -> list[str]:
    p = ROOT / "_ndx100_all.txt"
    if not p.exists():
        return []
    return [
        _to_file_safe(ln.strip())
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


PRESET_UNIVERSES: dict[str, dict[str, object]] = {
    "dow30": {
        "tickers": [_to_file_safe(t) for t in DOW_30_TICKER],
        "benchmark": "^DJI",
    },
    "sox30": {
        "tickers": [_to_file_safe(t) for t in SOX_30_TICKER],
        "benchmark": "^SOX",
    },
    "sp100": {
        "tickers": [_to_file_safe(t) for t in SP100],
        "benchmark": "^OEX",
    },
    "ndx100": {
        "tickers": _load_ndx100_from_file(),
        "benchmark": "^NDX",
    },
}


def resolve_universe(universe: str | None) -> tuple[str | None, list[str], str | None]:
    key = (universe or "").strip().lower()
    if not key:
        return None, [], None
    if key not in PRESET_UNIVERSES:
        raise ValueError(f"unknown universe: {universe}")
    spec = PRESET_UNIVERSES[key]
    tickers = list(spec.get("tickers", []))
    benchmark = str(spec.get("benchmark")) if spec.get("benchmark") else None
    return key, tickers, benchmark


def discover_tickers() -> list[str]:
    tickers = []
    for p in DES_PRED_ROOT.glob("DES_pred_*_*.csv"):
        m = re.match(r"^DES_pred_(.+)_\d{4}-\d{2}-\d{2}\.csv$", p.name)
        if m:
            tickers.append(m.group(1))
    return _dedupe_share_classes(sorted(set(tickers)))[0]


def _dedupe_share_classes(tickers: list[str]) -> tuple[list[str], list[str]]:
    tset = set([str(t).upper() for t in tickers])
    dropped = []
    for src, keep in SHARE_CLASS_ALIAS.items():
        if src in tset and keep in tset:
            tset.discard(src)
            dropped.append(src)
    return sorted(tset), sorted(dropped)


def _read_score_csv(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.index = pd.to_datetime(s.index)
    s.name = "Score"
    return s.sort_index()


def read_des(ticker: str) -> pd.Series:
    cand = sorted(DES_PRED_ROOT.glob(f"DES_pred_{ticker}_*.csv"))
    if not cand:
        raise FileNotFoundError(f"DES pred missing for {ticker}")
    return _read_score_csv(cand[-1])


def read_cusum_prob6(ticker: str) -> pd.Series:
    p = CUSUM6_ROOT / f"cusum_{ticker}.csv"
    if not p.exists():
        raise FileNotFoundError(f"cumSum_prob_6 missing for {ticker}")
    s = pd.read_csv(p, index_col=0, parse_dates=True, header=None).squeeze("columns")
    s.index = pd.to_datetime(s.index)
    s.name = "Score"
    return s.sort_index()


def read_cusum_prob12(ticker: str) -> pd.Series:
    p = CUSUM12_ROOT / f"cusum_{ticker}.csv"
    if not p.exists():
        raise FileNotFoundError(f"cumSum_prob_12 missing for {ticker}")
    s = pd.read_csv(p, index_col=0, parse_dates=True, header=None).squeeze("columns")
    s.index = pd.to_datetime(s.index)
    s.name = "Score"
    return s.sort_index()


def _to_unit_interval(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    finite = x[np.isfinite(x)]
    if finite.empty:
        return x.fillna(0.5)

    if float(finite.min()) < 0.0 or float(finite.max()) > 1.0:
        x = (x + 1.0) / 2.0

    x = x.clip(lower=0.0, upper=1.0)
    return x.fillna(0.5)


def blended_score(ticker: str) -> pd.Series:
    des = read_des(ticker)
    c6 = _to_unit_interval(read_cusum_prob6(ticker).reindex(des.index).ffill())
    c12 = _to_unit_interval(read_cusum_prob12(ticker).reindex(des.index).ffill())
    score = 0.50 * des + 0.20 * c6 + 0.30 * c12
    score = score.astype(float).fillna(0.5)
    score.name = ticker
    return score


def blended_components(ticker: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    des = read_des(ticker)
    c6 = _to_unit_interval(read_cusum_prob6(ticker).reindex(des.index).ffill())
    c12 = _to_unit_interval(read_cusum_prob12(ticker).reindex(des.index).ffill())
    score = 0.50 * des + 0.20 * c6 + 0.30 * c12

    des = des.astype(float).fillna(0.5)
    c6 = c6.astype(float).fillna(0.5)
    c12 = c12.astype(float).fillna(0.5)
    score = score.astype(float).fillna(0.5)
    des.name = ticker
    c6.name = ticker
    c12.name = ticker
    score.name = ticker
    return des, c6, c12, score


def _get_shares_outstanding(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        shares = None
        try:
            shares = fi["shares"]
        except Exception:
            shares = getattr(fi, "shares", None)
        if shares is not None and not (isinstance(shares, float) and np.isnan(shares)):
            return float(shares)
    except Exception:
        pass

    try:
        t = yf.Ticker(ticker)
        ser = t.get_shares_full()
        if ser is not None and len(ser.dropna()) > 0:
            return float(ser.dropna().iloc[-1])
    except Exception:
        pass
    return None


def compute_market_weights(scores_wide: pd.DataFrame) -> pd.Series:
    if scores_wide.empty or scores_wide.shape[1] == 0:
        return pd.Series(dtype=float)
    latest_date = scores_wide.index.max()
    if pd.isna(latest_date):
        return pd.Series(dtype=float)
    latest_scores = scores_wide.loc[latest_date]
    tickers = [t for t in scores_wide.columns if pd.notna(latest_scores.get(t))]

    closes = {}
    for tk in tickers:
        try:
            c = yf.Ticker(tk).history(period="7d", auto_adjust=True)["Close"].dropna()
            if len(c) > 0:
                closes[tk] = float(c.iloc[-1])
        except Exception:
            continue

    caps = {}
    for tk in tickers:
        if tk not in closes:
            continue
        sh = _get_shares_outstanding(tk)
        if sh is None or sh <= 0:
            continue
        caps[tk] = sh * closes[tk]

    if not caps:
        # fallback to equal weight on available columns
        w = pd.Series(1.0 / len(tickers), index=tickers) if tickers else pd.Series(dtype=float)
        return w

    s = pd.Series(caps, dtype=float)
    return s / s.sum()


def load_benchmark(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    hist = yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"), auto_adjust=True)
    if "Close" not in hist.columns or hist["Close"].dropna().empty:
        raise RuntimeError(f"benchmark {symbol} has no data")
    s = hist["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def plot_ai_score(
    weighted: pd.Series,
    benchmark: pd.Series,
    out_html: Path,
    benchmark_symbol: str,
    delta_zoom: float = 10.0,
):
    b = benchmark.reindex(weighted.index).ffill().bfill()
    score = weighted
    diff = weighted.diff().fillna(0)
    threshold = 0.5

    bar_colors = np.where(diff.values > 0, "#E74C3C", np.where(diff.values < 0, "#27AE60", "#BDC3C7"))

    # Make small day-to-day changes visible while using a recent-window y-scale.
    delta_zoom = max(float(delta_zoom), 1.0)
    diff_plot = diff * delta_zoom
    diff_ma5 = diff_plot.rolling(5, min_periods=1).mean()

    # Use full plotted period max range with generous headroom to avoid clipping.
    all_abs = np.abs(diff_plot.values)
    all_abs = all_abs[np.isfinite(all_abs)]
    if all_abs.size > 0:
        y_lim = max(float(np.max(all_abs)) * 1.40, 0.001)
    else:
        y_lim = 0.01

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=("", "Score Difference (Day-over-Day)"),
        specs=[[{"secondary_y": True}], [{}]],
    )

    fig.add_trace(
        go.Scatter(x=weighted.index, y=b.values, mode="lines", name=benchmark_symbol, line=dict(color="#1F4E79", width=2)),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=weighted.index, y=score.values, mode="lines", name="AIQuant Score", line=dict(color="#E07B00", width=2)),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=weighted.index,
            y=np.full(len(score), threshold),
            mode="lines",
            name=f"Threshold {threshold:.2f}",
            line=dict(color="#C0392B", width=1.2, dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=weighted.index,
            y=score.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=weighted.index,
            y=np.full(len(score), threshold),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(39,174,96,0.13)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=weighted.index,
            y=diff_plot.values,
            customdata=diff.values,
            marker_color=bar_colors,
            name="Delta (scaled)",
            opacity=0.85,
            hovertemplate="Date=%{x}<br>Scaled Δ=%{y:.3f}<br>Raw Δ=%{customdata:.5f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=diff_ma5.index, y=diff_ma5.values, mode="lines", name="5MA", line=dict(color="#1B1B1B", width=1.1)),
        row=2,
        col=1,
    )

    last_x = score.index[-1]
    last_y = float(score.iloc[-1])
    last_date_str = pd.Timestamp(last_x).strftime("%Y-%m-%d")
    fig.add_trace(
        go.Scatter(
            x=[last_x],
            y=[last_y],
            mode="markers+text",
            # Date intentionally omitted — the vertical dashed-line annotation
            # already carries "Latest: YYYY-MM-DD" at the top of the plot, and
            # showing it here overlaps the score value at the latest marker.
            text=[f"{last_y:.2f}"],
            textposition="middle right",
            textfont=dict(size=12, color="#1B1B1B"),
            marker=dict(size=9, color="#E07B00", line=dict(color="#FFFFFF", width=1)),
            name=f"Latest ({last_date_str})",
            hovertemplate=f"Latest {last_date_str}<br>Score=%{{y:.4f}}<extra></extra>",
            cliponaxis=False,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Vertical dashed line marking the latest date across both subplots
    fig.add_vline(
        x=last_x,
        line_width=1,
        line_dash="dot",
        line_color="#888888",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=last_x,
        line_width=1,
        line_dash="dot",
        line_color="#888888",
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=last_x,
        y=1.0,
        yref="paper",
        xref="x",
        text=f"Latest: {last_date_str}",
        showarrow=False,
        yanchor="bottom",
        xanchor="right",
        font=dict(size=11, color="#555555"),
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=2,
    )

    fig.add_hline(y=0, line_width=1, line_color="#444444", row=2, col=1)

    fig.update_layout(
        height=920,
        width=1400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        bargap=0,
        title=dict(text="US AIQuant Score", x=0.5),
        margin=dict(r=150),
    )
    fig.update_yaxes(title_text=benchmark_symbol, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="AIQuant Score", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=f"Δ Score (scaled, zoom x{delta_zoom:.1f})", row=2, col=1, range=[-y_lim, y_lim])
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.write_html(str(out_html), include_plotlyjs=True)


def generate_ai_score(
    tickers: list[str],
    benchmark_symbol: str = "^NDX",
    start: str | None = None,
    end: str | None = None,
    delta_zoom: float = 10.0,
    smooth_window: int = 5,
    universe: str | None = None,
) -> dict:
    tickers = [_to_file_safe(t) for t in tickers]
    tickers, dropped_aliases = _dedupe_share_classes(tickers)
    dropped_degenerate = sorted(t for t in tickers if t in DEGENERATE_DES_TICKERS)
    tickers = [t for t in tickers if t not in DEGENERATE_DES_TICKERS]

    universe_key = (universe or "us").strip().lower() or "us"
    score_stem = f"AI_score_{universe_key}"
    score_series_stem = f"AI_score_{universe_key}_series"
    snapshot_stem = f"AI_score_snapshot_{universe_key}"
    summary_stem = f"AI_score_summary_{universe_key}"

    if not start:
        start = "2024-01-01"

    records = {"ok": [], "skipped": []}
    scores = []
    des_series = []
    c6_series = []
    c12_series = []

    for tk in tickers:
        try:
            des, c6, c12, s = blended_components(tk)
            des_series.append(des)
            c6_series.append(c6)
            c12_series.append(c12)
            scores.append(s)
            records["ok"].append(tk)
        except Exception as exc:
            records["skipped"].append({"ticker": tk, "reason": str(exc)})

    if not scores:
        raise RuntimeError("No score series available")

    wide_all = pd.concat(scores, axis=1).sort_index()
    des_wide_all = pd.concat(des_series, axis=1).sort_index()
    c6_wide_all = pd.concat(c6_series, axis=1).sort_index()
    c12_wide_all = pd.concat(c12_series, axis=1).sort_index()

    wide = wide_all.copy()
    des_wide = des_wide_all.copy()
    c6_wide = c6_wide_all.copy()
    c12_wide = c12_wide_all.copy()

    if start:
        wide = wide[wide.index >= pd.Timestamp(start)]
        des_wide = des_wide[des_wide.index >= pd.Timestamp(start)]
        c6_wide = c6_wide[c6_wide.index >= pd.Timestamp(start)]
        c12_wide = c12_wide[c12_wide.index >= pd.Timestamp(start)]
    if end:
        wide = wide[wide.index <= pd.Timestamp(end)]
        des_wide = des_wide[des_wide.index <= pd.Timestamp(end)]
        c6_wide = c6_wide[c6_wide.index <= pd.Timestamp(end)]
        c12_wide = c12_wide[c12_wide.index <= pd.Timestamp(end)]
    if wide.empty:
        # When requested date window has no trading session, fallback to full available range.
        wide = wide_all.copy()
        des_wide = des_wide_all.copy()
        c6_wide = c6_wide_all.copy()
        c12_wide = c12_wide_all.copy()
    wide = wide.ffill().bfill()
    des_wide = des_wide.ffill().bfill()
    c6_wide = c6_wide.ffill().bfill()
    c12_wide = c12_wide.ffill().bfill()

    weights = compute_market_weights(wide)
    common_cols = [c for c in wide.columns if c in weights.index]
    if not common_cols:
        raise RuntimeError("No common columns between scores and weights")

    weighted = (wide[common_cols] * weights[common_cols]).sum(axis=1)
    weighted.name = f"AI_SCORE_{universe_key.upper()}"
    if smooth_window and int(smooth_window) > 1:
        weighted = weighted.rolling(int(smooth_window), min_periods=1).mean().bfill()

    sdate = wide.index.min()
    edate = wide.index.max()
    benchmark = load_benchmark(benchmark_symbol, sdate, edate)

    stamp = edate.strftime("%Y%m%d")
    out_html = SCORE_ROOT / f"{score_stem}_{stamp}.html"
    out_csv = SCORE_ROOT / f"{score_series_stem}_{stamp}.csv"
    out_snap = SNAPSHOT_ROOT / f"{snapshot_stem}_{stamp}.csv"
    out_json = SUMMARY_ROOT / f"{summary_stem}_{stamp}.json"

    weighted.to_csv(out_csv, header=True, encoding="utf-8")
    plot_ai_score(weighted, benchmark, out_html, benchmark_symbol, delta_zoom=delta_zoom)

    latest = wide.loc[edate].rename("score").to_frame()
    latest.index.name = "ticker"
    latest["weight"] = latest.index.map(weights).fillna(0.0)
    latest["weighted_score"] = latest["score"] * latest["weight"]
    latest.to_csv(out_snap, encoding="utf-8")

    latest_des = des_wide.reindex(wide.index).loc[edate]
    latest_c6 = c6_wide.reindex(wide.index).loc[edate]
    latest_c12 = c12_wide.reindex(wide.index).loc[edate]
    latest_score_cross = wide.loc[edate]
    low_mask = latest_score_cross < 0.5
    low_ratio = float(low_mask.mean()) if len(latest_score_cross) else 0.0

    summary = {
        "universe": universe_key,
        "date": str(edate.date()),
        "benchmark": benchmark_symbol,
        "start": str(pd.Timestamp(start).date()) if start else None,
        "delta_zoom": float(delta_zoom),
        "smooth_window": int(smooth_window),
        "tickers_requested": len(tickers),
        "dropped_share_class_aliases": dropped_aliases,
        "dropped_degenerate_des": dropped_degenerate,
        "tickers_used": int(len(records["ok"])),
        "tickers_skipped": int(len(records["skipped"])),
        "latest_score": float(weighted.iloc[-1]),
        "diagnostics": {
            "latest_cross_section_mean": float(latest_score_cross.mean()),
            "latest_cross_section_median": float(latest_score_cross.median()),
            "latest_below_0_5_count": int(low_mask.sum()),
            "latest_below_0_5_ratio": low_ratio,
            "latest_component_means": {
                "des_mean": float(latest_des.mean()),
                "cusum6_mean": float(latest_c6.mean()),
                "cusum12_mean": float(latest_c12.mean()),
            },
            "blended_formula": "0.50*DES + 0.20*cusum6_norm01 + 0.30*cusum12_norm01",
        },
        "paths": {
            "score_html": str(out_html),
            "score_series_csv": str(out_csv),
            "snapshot_csv": str(out_snap),
        },
        "skipped": records["skipped"],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # latest aliases (per-universe only; legacy US-only alias removed to avoid
    # duplicates with the last universe written in --all-universes runs).
    (SCORE_ROOT / f"{score_stem}_latest.html").write_bytes(out_html.read_bytes())
    (SCORE_ROOT / f"{score_series_stem}_latest.csv").write_bytes(out_csv.read_bytes())
    (SNAPSHOT_ROOT / f"{snapshot_stem}_latest.csv").write_bytes(out_snap.read_bytes())
    (SUMMARY_ROOT / f"{summary_stem}_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def generate_ai_score_for_universe(
    universe: str,
    start: str | None = None,
    end: str | None = None,
    delta_zoom: float = 10.0,
    smooth_window: int = 5,
) -> dict:
    key, tickers, benchmark = resolve_universe(universe)
    if not key:
        raise ValueError("universe is required")
    if not tickers:
        raise RuntimeError(f"universe {key} has empty ticker list")

    return generate_ai_score(
        tickers=tickers,
        benchmark_symbol=benchmark or "^NDX",
        start=start,
        end=end,
        delta_zoom=delta_zoom,
        smooth_window=smooth_window,
        universe=key,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate US AI score from DES + CUSUM")
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated tickers; default auto from model_pred_DES_US")
    parser.add_argument("--benchmark", type=str, default="^NDX", help="benchmark symbol, default ^NDX")
    parser.add_argument("--universe", type=str, default=None, help="preset universe: dow30|sox30|ndx100|sp100")
    parser.add_argument("--all-universes", action="store_true", help="generate all preset universes")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--delta-zoom", type=float, default=10.0, help="delta-panel y-axis zoom factor (>=1); larger = more visible swings")
    parser.add_argument("--smooth-window", type=int, default=5, help="rolling smoothing window for AI score (trading days; default 5 = ~1 week)")
    args = parser.parse_args()

    if args.all_universes:
        for uni in ("dow30", "sox30", "ndx100", "sp100"):
            summary = generate_ai_score_for_universe(
                universe=uni,
                start=args.start,
                end=args.end,
                delta_zoom=args.delta_zoom,
                smooth_window=args.smooth_window,
            )
            print(f"[DONE][{uni}] score={summary['latest_score']:.4f} used={summary['tickers_used']} skipped={summary['tickers_skipped']}")
            print(f"[OUT][{uni}] {summary['paths']['score_html']}")
        return

    if args.universe:
        summary = generate_ai_score_for_universe(
            universe=args.universe,
            start=args.start,
            end=args.end,
            delta_zoom=args.delta_zoom,
            smooth_window=args.smooth_window,
        )
        print(f"[DONE][{summary['universe']}] score={summary['latest_score']:.4f} used={summary['tickers_used']} skipped={summary['tickers_skipped']}")
        print(f"[OUT][{summary['universe']}] {summary['paths']['score_html']}")
        return

    tickers = discover_tickers() if not args.tickers else sorted(set([_to_file_safe(x.strip()) for x in args.tickers.split(",") if x.strip()]))
    if not tickers:
        raise SystemExit("No tickers found")

    summary = generate_ai_score(
        tickers=tickers,
        benchmark_symbol=args.benchmark,
        start=args.start,
        end=args.end,
        delta_zoom=args.delta_zoom,
        smooth_window=args.smooth_window,
        universe="us",
    )
    print(f"[DONE][us] score={summary['latest_score']:.4f} used={summary['tickers_used']} skipped={summary['tickers_skipped']}")
    print(f"[OUT][us] {summary['paths']['score_html']}")


if __name__ == "__main__":
    main()
