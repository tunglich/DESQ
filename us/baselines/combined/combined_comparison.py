"""Build the combined 1×3 comparison figure + extended stats table.

Layout: one figure with 3 side-by-side subplots — Dow 30 | S&P 100 | NASDAQ 100.
Each subplot shows 5 cumulative-return curves (DESQ + 3 papers + benchmark),
styled to match the academic reference (serif font, muted palette, subtle grid).

Outputs:
    baselines/combined/four_methods_1x3.png       — main 1×3 combined figure
    baselines/combined/{u}_comparison.png         — per-universe standalone (kept)
    baselines/combined/{u}_comparison.csv         — per-universe cum-return data
    baselines/combined/combined_stats.csv         — extended statistics table
    baselines/combined/combined_stats.md          — same table in Markdown
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Style — academic / serif, muted palette (mirrors the reference figure).
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Palatino Linotype", "Georgia", "serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "regular",
    "axes.labelsize":    10,
    "axes.edgecolor":    "#666666",
    "axes.linewidth":    0.6,
    "axes.labelcolor":   "#333333",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.frameon":    True,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#DDDDDD",
    "grid.color":        "#DDDDDD",
    "grid.linewidth":    0.5,
    "grid.linestyle":    "-",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# Seaborn-deep muted palette — matches the reference chart family
PALETTE = {
    "DESQ":                            "#4C72B0",   # muted blue (thick primary)
    "DRL Ensemble":                   "#DD8452",   # muted orange
    "Dynamic Stock Recommendation":   "#55A868",   # muted green
    "MACE":                           "#8172B2",   # muted purple
    # index colour is added per-panel below (matches whichever ^DJI/^OEX/^NDX)
}
INDEX_COLOUR = "#333333"


WS = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_START = pd.Timestamp("2024-01-02")
TRADE_END   = pd.Timestamp("2026-03-30")
INITIAL_CAPITAL = 1_000_000

UNIVERSES = {
    "dow30": {
        "label":         "Dow 30",
        "index_ticker":  "^DJI",
        "des_csv":       WS / "backtest_portfolio_US" / "equity_dow30_market_2024-01-02_2026-03-31.csv",
        "yang2020_csv":  WS / "FinRL" / "backtest_dow30_2024_20260331" / "backtest_results.csv",
        "yang2020_col":  "ensemble",
        "yang2018_dir":  WS / "baselines" / "dsr_yang" / "backtest_dow30_2024_20260330",
        "abbade_dir":    WS / "baselines" / "mi_abbade" / "backtest_dow30_2024_20260330",
    },
    "sp100": {
        "label":         "S&P 100",
        "index_ticker":  "^OEX",
        "des_csv":       WS / "backtest_portfolio_US" / "equity_sp100_market_2024-01-02_2026-03-31.csv",
        "yang2020_csv":  WS / "FinRL" / "backtest_sp100_variantA_2024_20260331" / "backtest_results.csv",
        "yang2020_col":  "ensemble",
        "yang2018_dir":  WS / "baselines" / "dsr_yang" / "backtest_sp100_2024_20260330",
        "abbade_dir":    WS / "baselines" / "mi_abbade" / "backtest_sp100_2024_20260330",
    },
    "ndx100": {
        "label":         "NASDAQ 100",
        "index_ticker":  "^NDX",
        "des_csv":       WS / "backtest_portfolio_US" / "equity_ndx100_market_2024-01-02_2026-03-31.csv",
        "yang2020_csv":  WS / "FinRL" / "backtest_ndx100_variantA_2024_20260331" / "backtest_results.csv",
        "yang2020_col":  "ensemble",
        "yang2018_dir":  WS / "baselines" / "dsr_yang" / "backtest_ndx100_2024_20260330",
        "abbade_dir":    WS / "baselines" / "mi_abbade" / "backtest_2024_20260330",
    },
}

METHOD_ORDER = ["DESQ",
                "Dynamic Stock Recommendation",
                "DRL Ensemble",
                "MACE"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _rebase(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return INITIAL_CAPITAL * s / s.iloc[0]


def _load_des(fp: Path) -> Optional[pd.Series]:
    if not fp.exists():
        print(f"  [warn] DES CSV missing: {fp}"); return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    s = df["portfolio_equity"].loc[TRADE_START:TRADE_END]
    return s if not s.empty else None


def _load_yang2020(fp: Path, col: str) -> Optional[pd.Series]:
    if not fp.exists():
        print(f"  [warn] Yang 2020 CSV missing: {fp}"); return None
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    if col not in df.columns:
        return None
    s = df[col].loc[TRADE_START:TRADE_END]
    return s if not s.empty else None


def _load_yang2018(out_dir: Path) -> Optional[pd.Series]:
    best_s, best_ret = None, -np.inf
    for mode in ("ew", "mvo", "minvar"):
        fp = out_dir / f"equity_{mode}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        s = df[df.columns[0]].loc[TRADE_START:TRADE_END]
        if s.empty:
            continue
        ret = s.iloc[-1] / s.iloc[0] - 1
        if ret > best_ret:
            best_ret, best_s = ret, s
    return best_s


def _load_abbade(out_dir: Path) -> Optional[pd.Series]:
    best_s, best_ret = None, -np.inf
    for fp in out_dir.glob("equity_*_ac.csv"):
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        s = df[df.columns[0]].loc[TRADE_START:TRADE_END]
        if s.empty:
            continue
        ret = s.iloc[-1] / s.iloc[0] - 1
        if ret > best_ret:
            best_ret, best_s = ret, s
    return best_s


def _load_index(ticker: str) -> Optional[pd.Series]:
    df = yf.download(ticker, start=TRADE_START,
                     end=TRADE_END + pd.Timedelta(days=1),
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df["Close"].loc[TRADE_START:TRADE_END]


def _load_shipped_comparison(u: str) -> Dict[str, pd.Series]:
    path = OUT_DIR / f"{u}_comparison.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    return {
        name: INITIAL_CAPITAL * (1.0 + frame[name].dropna() / 100.0)
        for name in frame.columns
    }


def _load_universe(u: str) -> Dict[str, pd.Series]:
    shipped = _load_shipped_comparison(u)
    if shipped:
        return shipped
    U = UNIVERSES[u]
    series: Dict[str, pd.Series] = {}
    if (s := _load_des(U["des_csv"])) is not None:
        series["DESQ"] = _rebase(s)
    if (s := _load_yang2020(U["yang2020_csv"], U["yang2020_col"])) is not None:
        series["DRL Ensemble"] = _rebase(s)
    if (s := _load_yang2018(U["yang2018_dir"])) is not None:
        series["Dynamic Stock Recommendation"] = _rebase(s)
    if (s := _load_abbade(U["abbade_dir"])) is not None:
        series["MACE"] = _rebase(s)
    if (s := _load_index(U["index_ticker"])) is not None:
        series[U["index_ticker"]] = _rebase(s)
    return series


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
ANN = 252


def _stats(s: pd.Series, bench: Optional[pd.Series] = None) -> Dict[str, float]:
    s = s.dropna()
    if len(s) < 2:
        return {}
    r = s.pct_change().dropna()
    days = len(s)
    years = days / ANN
    final = float(s.iloc[-1])
    total_ret = float(s.iloc[-1] / s.iloc[0] - 1)
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else float("nan")
    ann_vol = float(r.std() * np.sqrt(ANN))
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else float("nan")
    neg = r[r < 0]
    sortino = float(r.mean() / neg.std() * np.sqrt(ANN)) if len(neg) > 0 and neg.std() > 0 else float("nan")
    mdd = float(((s - s.cummax()) / s.cummax()).min())
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")

    out = {
        "final_$":     final,
        "total_ret_%": total_ret * 100,
        "ann_ret_%":   ann_ret * 100,
        "ann_vol_%":   ann_vol * 100,
        "sharpe":      sharpe,
        "sortino":     sortino,
        "max_dd_%":    mdd * 100,
        "calmar":      calmar,
        "days":        days,
    }
    if bench is not None and len(bench.dropna()) >= 2:
        bench_ret = float(bench.dropna().iloc[-1] / bench.dropna().iloc[0] - 1)
        out["excess_ret_%"] = (total_ret - bench_ret) * 100
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_panel(ax, series: Dict[str, pd.Series], universe_label: str,
                index_ticker: str, show_ylabel: bool = True) -> pd.DataFrame:
    """Plot one universe onto `ax`. Returns cumulative-return DataFrame."""
    cum = pd.DataFrame({name: (s / INITIAL_CAPITAL - 1.0) * 100
                        for name, s in series.items()})
    cum.index.name = "date"
    colours = dict(PALETTE)
    colours[index_ticker] = INDEX_COLOUR

    # Draw non-DESQ first (so DESQ sits on top)
    for name in cum.columns:
        if name == "DESQ":
            continue
        s = cum[name].dropna()
        is_index = name == index_ticker
        ax.plot(s.index, s.values,
                label=name,
                color=colours.get(name, "#666"),
                linewidth=1.4 if is_index else 1.5,
                linestyle=(0, (5, 2)) if is_index else "-",
                alpha=0.85, zorder=2)
    # DESQ on top
    if "DESQ" in cum.columns:
        s = cum["DESQ"].dropna()
        ax.plot(s.index, s.values,
            label="Legacy DES+CUSUM",
                color=colours["DESQ"], linewidth=2.4,
                alpha=1.0, zorder=5)

    # Anti-collision end-of-line labels
    finals = [(name, float(cum[name].dropna().iloc[-1])) for name in cum.columns]
    y_hi = max(v for _, v in finals); y_lo = min(v for _, v in finals)
    y_range = max(y_hi - y_lo, 1.0)
    min_gap = max(3.5, 0.05 * y_range)
    ordered = sorted(finals, key=lambda t: t[1])
    adj = {n: y for n, y in ordered}
    names_up = [n for n, _ in ordered]
    for i in range(1, len(names_up)):
        prev_y = adj[names_up[i - 1]]
        if adj[names_up[i]] - prev_y < min_gap:
            adj[names_up[i]] = prev_y + min_gap
    for i in range(len(names_up) - 2, -1, -1):
        nxt_y = adj[names_up[i + 1]]
        if nxt_y - adj[names_up[i]] < min_gap:
            adj[names_up[i]] = nxt_y - min_gap

    last_x = cum.index[-1]; dx = pd.Timedelta(days=10)
    for name in cum.columns:
        actual_y = float(cum[name].dropna().iloc[-1])
        text_y = adj[name]
        colour = colours.get(name, "#666")
        if abs(text_y - actual_y) > 0.5:
            ax.plot([last_x, last_x + dx], [actual_y, text_y],
                    color=colour, linewidth=0.6, alpha=0.55, zorder=1)
        ax.text(last_x + dx, text_y, f"{actual_y:+.1f}%",
                color=colour, fontsize=8.5,
                fontweight="bold" if name == "DESQ" else "normal",
                va="center", ha="left")

    ax.axhline(0.0, color="#999999", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_title(f"{universe_label}", loc="center")
    ax.set_xlabel("Date")
    if show_ylabel:
        ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, which="major", alpha=0.5)
    ax.set_xlim(right=last_x + pd.Timedelta(days=55))
    for label in ax.get_xticklabels():
        label.set_rotation(30)

    return cum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    universe_series = {u: _load_universe(u) for u in UNIVERSES}

    # ---- Standalone per-universe charts (kept for reference) --------------
    for u, series in universe_series.items():
        U = UNIVERSES[u]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        cum = _plot_panel(ax, series, U["label"], U["index_ticker"],
                          show_ylabel=True)
        ax.legend(loc="upper left")
        fig.suptitle(f"Legacy DES+CUSUM diagnostic vs {U['index_ticker']}   "
                     f"{TRADE_START.date()} \u2192 {TRADE_END.date()}",
                     y=0.995, fontsize=11)
        fig.tight_layout()
        png = OUT_DIR / f"{u}_comparison.png"
        fig.savefig(png, dpi=160); plt.close(fig)
        cum.to_csv(
            OUT_DIR / f"{u}_comparison.csv", float_format="%.6f", lineterminator="\n"
        )
        print(f"[OK] {u}: {png.name} + {u}_comparison.csv")

    # ---- Main 1×3 combined figure ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for i, (u, series) in enumerate(universe_series.items()):
        U = UNIVERSES[u]
        _plot_panel(axes[i], series, U["label"], U["index_ticker"],
                    show_ylabel=(i == 0))
    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate benchmark by adding all 3 as distinct labels
    all_h, all_l = [], []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                seen.add(li); all_h.append(hi); all_l.append(li)
    fig.suptitle(f"Legacy DES+CUSUM diagnostic across three universes   "
                 f"{TRADE_START.date()} \u2192 {TRADE_END.date()}",
                 y=1.02, fontsize=12)
    fig.legend(all_h, all_l, loc="lower center", ncol=len(all_l),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    combined_png = OUT_DIR / "four_methods_1x3.png"
    fig.savefig(combined_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {combined_png}")

    # ---- Extended statistics table ---------------------------------------
    all_rows = []
    for u, series in universe_series.items():
        U = UNIVERSES[u]
        bench = series.get(U["index_ticker"])
        # bench itself has no excess-ret
        for name, s in series.items():
            row = {"universe": u,
                   "universe_label": U["label"],
                   "method": name}
            row.update(_stats(s, bench=None if name == U["index_ticker"] else bench))
            all_rows.append(row)
    stats = pd.DataFrame(all_rows)
    cols = ["universe", "universe_label", "method", "final_$",
            "total_ret_%", "excess_ret_%",
            "ann_ret_%", "ann_vol_%",
            "sharpe", "sortino", "max_dd_%", "calmar", "days"]
    stats = stats.reindex(columns=cols)
    stats.to_csv(OUT_DIR / "combined_stats.csv", index=False,
                 float_format="%.4f", lineterminator="\n")
    print(f"[OK] wrote {OUT_DIR / 'combined_stats.csv'}")

    # ---- Markdown snapshot ----------------------------------------------
    md_lines = ["| Universe | Method | Final $ | Total Ret % | Excess Ret % "
                "| Ann Ret % | Ann Vol % | Sharpe | Sortino | MaxDD % | Calmar |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for u in UNIVERSES:
        for _, r in stats[stats["universe"] == u].iterrows():
            def _f(x, prec=2):
                return f"{x:+.{prec}f}" if isinstance(x, float) and not np.isnan(x) else "—"
            md_lines.append(
                f"| {r['universe_label']} | {r['method']} | "
                f"{r['final_$']:,.0f} | {_f(r['total_ret_%'])} | "
                f"{_f(r.get('excess_ret_%', float('nan')))} | "
                f"{_f(r['ann_ret_%'])} | {_f(r['ann_vol_%'])} | "
                f"{_f(r['sharpe'])} | {_f(r['sortino'])} | "
                f"{_f(r['max_dd_%'])} | {_f(r['calmar'])} |"
            )
    (OUT_DIR / "combined_stats.md").write_text("\n".join(md_lines) + "\n",
                                                encoding="utf-8", newline="\n")
    print(f"[OK] wrote {OUT_DIR / 'combined_stats.md'}")

    print("\n=== Summary (total_ret_%) ===")
    piv = stats.pivot(index="universe", columns="method", values="total_ret_%")
    print(piv.to_string(float_format=lambda x: f"{x:>+7.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
