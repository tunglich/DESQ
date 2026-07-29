from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "AI_us"
SNAPSHOT_ROOT = OUT_ROOT / "snapshots"
TREE_ROOT = OUT_ROOT / "tree"
SUMMARY_ROOT = OUT_ROOT / "summary"
META_CACHE_ROOT = ROOT / "feature" / "_raw"

SHARE_CLASS_ALIAS = {
    "GOOG": "GOOGL",
}

for d in (TREE_ROOT, SUMMARY_ROOT):
    d.mkdir(parents=True, exist_ok=True)


def _latest_snapshot() -> Path:
    cands = sorted(SNAPSHOT_ROOT.glob("AI_score_snapshot_*_latest.csv"))
    if cands:
        return cands[-1]
    cands = sorted(SNAPSHOT_ROOT.glob("AI_score_snapshot_*.csv"))
    if cands:
        return cands[-1]
    raise FileNotFoundError("No AI score snapshot found under AI_us/snapshots")


def _meta_cache_path(ticker: str) -> Path:
    return META_CACHE_ROOT / f"us_meta_{ticker}.json"


def _get_meta(ticker: str) -> dict:
    cache = _meta_cache_path(ticker)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    out = {
        "ticker": ticker,
        "sector": "Unknown",
        "industry": "Unknown",
        "shortName": ticker,
        "marketCap": None,
    }
    try:
        info = yf.Ticker(ticker).info
        out["sector"] = info.get("sector") or "Unknown"
        out["industry"] = info.get("industry") or "Unknown"
        out["shortName"] = info.get("shortName") or ticker
        out["marketCap"] = info.get("marketCap")
    except Exception:
        pass

    try:
        cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass
    return out


def generate_ai_tree(snapshot_csv: str | None = None) -> dict:
    try:
        import plotly.express as px
    except Exception as exc:
        raise RuntimeError(
            "plotly is required for AI tree generation. Install with: pip install plotly"
        ) from exc

    snap_path = Path(snapshot_csv) if snapshot_csv else _latest_snapshot()
    df = pd.read_csv(snap_path)

    if "ticker" not in df.columns or "score" not in df.columns:
        raise ValueError("snapshot must contain columns: ticker, score")

    # Drop duplicate share classes when both source/target exist.
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for src, keep in SHARE_CLASS_ALIAS.items():
        if src in set(df["ticker"]) and keep in set(df["ticker"]):
            df = df[df["ticker"] != src].copy()

    # Normalize numeric fields so treemap color/hover never receives string NaN payloads.
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.5)
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    else:
        df["weight"] = 0.0
    if "weighted_score" in df.columns:
        df["weighted_score"] = pd.to_numeric(df["weighted_score"], errors="coerce").fillna(0.0)
    else:
        df["weighted_score"] = 0.0

    rows = []
    for _, r in df.iterrows():
        tk = str(r["ticker"])
        meta = _get_meta(tk)
        rows.append(
            {
                "ticker": tk,
                "name": meta.get("shortName") or tk,
                "sector": meta.get("sector") or "Unknown",
                "industry": meta.get("industry") or "Unknown",
                "score": float(r["score"]),
                "weight": float(r["weight"]),
                "weighted_score": float(r["weighted_score"]),
                "market_cap": float(meta["marketCap"]) if meta.get("marketCap") not in (None, "", "nan") else 0.0,
            }
        )

    tree_df = pd.DataFrame(rows)
    if tree_df["market_cap"].sum() <= 0:
        # fallback: use weight as area if market cap unavailable
        tree_df["market_cap"] = tree_df["weight"].clip(lower=0.0)

    tree_df["country"] = "US-Stock"
    tree_df["label"] = tree_df.apply(
        lambda r: f"{r['ticker']} {r['name']}<br>{r['score']:.2f}",
        axis=1,
    )

    fig = px.treemap(
        tree_df,
        path=["country", "sector", "industry", "label"],
        values="market_cap",
        color="score",
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0.5,
        hover_data={
            "ticker": True,
            "score": ":.4f",
            "weight": ":.4f",
            "weighted_score": ":.4f",
            "market_cap": ":,.0f",
        },
        title="US AI Tree (Sector -> Industry)",
        width=1600,
        height=900,
    )

    # Parent nodes in treemap don't always have a direct color scalar,
    # so avoid rendering `%{color}` in text to prevent visual NaN labels.
    fig.update_traces(texttemplate="%{label}")

    # Infer universe key from snapshot filename: AI_score_snapshot_{universe}_{stamp}.csv
    import re as _re
    m = _re.match(r"AI_score_snapshot_([A-Za-z0-9]+)_", snap_path.name)
    universe_key = m.group(1).lower() if m else "us"

    stamp = datetime.now().strftime("%Y%m%d")
    tree_stem = f"AI_tree_{universe_key}"
    summary_stem = f"AI_tree_summary_{universe_key}"

    html_path = TREE_ROOT / f"{tree_stem}_{stamp}.html"
    csv_path = TREE_ROOT / f"{tree_stem}_{stamp}.csv"
    tree_df.to_csv(csv_path, index=False, encoding="utf-8")
    fig.write_html(str(html_path), include_plotlyjs=True)

    summary = {
        "universe": universe_key,
        "snapshot": str(snap_path),
        "tickers": int(len(tree_df)),
        "sectors": int(tree_df["sector"].nunique()),
        "industries": int(tree_df["industry"].nunique()),
        "paths": {
            "html": str(html_path),
            "csv": str(csv_path),
        },
    }

    jpath = SUMMARY_ROOT / f"{summary_stem}_{stamp}.json"
    jpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (TREE_ROOT / f"{tree_stem}_latest.html").write_bytes(html_path.read_bytes())
    (TREE_ROOT / f"{tree_stem}_latest.csv").write_bytes(csv_path.read_bytes())
    (SUMMARY_ROOT / f"{summary_stem}_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Legacy US-only aliases removed: they duplicated whichever universe was
    # written last in --all-universes runs. Consumers must read the
    # per-universe *_{universe}_latest.* files instead.
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate US AI treemap from latest AI score snapshot")
    parser.add_argument("--snapshot", type=str, default=None, help="optional snapshot csv path")
    args = parser.parse_args()

    summary = generate_ai_tree(args.snapshot)
    print(f"[DONE] tree tickers={summary['tickers']} sectors={summary['sectors']} industries={summary['industries']}")
    print(f"[OUT] {summary['paths']['html']}")


if __name__ == "__main__":
    main()
