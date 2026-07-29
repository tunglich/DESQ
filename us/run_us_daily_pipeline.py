from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from CumsumPro_US import T_WARMUPS, compute_cusum_series, load_price_frames
from FeatureUS_US import main as feature_main
from AIScore_US import generate_ai_score, generate_ai_score_for_universe
from AITree_US import generate_ai_tree

ROOT = Path(__file__).resolve().parent
DES_PRED_ROOT = ROOT / "model_pred_DES_US"
OUT_ROOT = ROOT / "AI_us"
SUMMARY_ROOT = OUT_ROOT / "summary"

for d in (OUT_ROOT, SUMMARY_ROOT):
    d.mkdir(parents=True, exist_ok=True)


def discover_universe_from_des() -> list[str]:
    tickers = set()
    for p in DES_PRED_ROOT.glob("DES_pred_*_*.csv"):
        m = re.match(r"^DES_pred_(.+)_\d{4}-\d{2}-\d{2}\.csv$", p.name)
        if m:
            tickers.add(m.group(1))
    return sorted(tickers)


def _recent_trading_day() -> str:
    # Use last completed US business day. During today's US session
    # (before ~16:15 ET close), yfinance daily bars still stop at the
    # previous business day, so we default to that to keep the pipeline's
    # start/end aligned with actually-available data.
    now_et = pd.Timestamp.now(tz="US/Eastern")
    ts = now_et.normalize()
    # Roll back to previous business day when today's cash session isn't
    # confirmed closed yet (safety margin 30 min past 16:00 ET close).
    market_closed = now_et.hour > 16 or (now_et.hour == 16 and now_et.minute >= 30)
    if ts.dayofweek >= 5 or not market_closed:
        ts = ts - pd.offsets.BDay(1)
    return ts.strftime("%Y-%m-%d")


def _run_feature_update(tickers: list[str]) -> dict:
    ok, failed = [], []
    for tk in tickers:
        try:
            feature_main(tickers=[tk])
            ok.append(tk)
            print(f"[FEATURE][OK] {tk}")
        except Exception as exc:
            failed.append({"ticker": tk, "reason": str(exc)})
            print(f"[FEATURE][FAIL] {tk}: {exc}")
    return {"ok": ok, "failed": failed}


def _run_cusum_update(tickers: list[str], end: str | None = None) -> dict:
    records = {"ok": [], "failed": []}
    frames = load_price_frames(tickers)
    close_wide = frames["Close"]

    end_ts = pd.Timestamp(end) if end else None
    for tw in T_WARMUPS:
        out_dir = ROOT / f"cumSum_prob_{tw}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for tk in tickers:
            try:
                if tk not in close_wide.columns:
                    raise ValueError("ticker missing in close frame")
                s = close_wide[tk].dropna()
                if end_ts is not None:
                    s = s[s.index <= end_ts]
                if s.empty:
                    raise ValueError("empty close series")
                probs = compute_cusum_series(s, tw)
                probs.to_csv(out_dir / f"cusum_{tk}.csv", header=False)
                records["ok"].append({"ticker": tk, "warmup": tw, "rows": int(len(probs))})
            except Exception as exc:
                records["failed"].append({"ticker": tk, "warmup": tw, "reason": str(exc)})
    print(f"[CUSUM] ok={len(records['ok'])} failed={len(records['failed'])}")
    return records


def _run_prediction_update_subprocess(
    tickers: list[str], start: str | None, end: str | None, strict: bool
) -> dict:
    summary_path = SUMMARY_ROOT / "prediction_update_summary_tmp.json"
    cmd = [
        sys.executable,
        str(ROOT / "prediction_update_US.py"),
        "--tickers",
        ",".join(tickers),
        "--summary",
        str(summary_path),
    ]
    if start:
        cmd.extend(["--start", start])
    if end:
        cmd.extend(["--end", end])
    if strict:
        cmd.append("--strict")

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())

    if summary_path.exists():
        try:
            obj = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            obj = {"ok": [], "skipped": [], "failed": []}
    else:
        obj = {"ok": [], "skipped": [], "failed": []}

    obj["runtime"] = "windows"
    obj["returncode"] = int(proc.returncode)

    # Legacy .keras files trained under py3.11 may fail to deserialize Lambda
    # bytecode in Windows py3.10 (`bad marshal data`). Auto-fallback to WSL py3.11.
    failed_reasons = [str(x.get("reason", "")) for x in obj.get("failed", []) if isinstance(x, dict)]
    need_wsl_fallback = (
        proc.returncode != 0
        or any("bad marshal data" in r for r in failed_reasons)
        or ("bad marshal data" in (proc.stdout or ""))
        or ("bad marshal data" in (proc.stderr or ""))
    )

    if need_wsl_fallback:
        wsl_summary_path = SUMMARY_ROOT / "prediction_update_summary_tmp_wsl.json"
        wsl_root = "/mnt/d/US_stock"
        wsl_summary = "/mnt/d/US_stock/AI_us/summary/prediction_update_summary_tmp_wsl.json"
        wsl_py = "/home/tungl/miniconda3/envs/finlabUS/bin/python"

        wsl_cmd = [
            "wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc",
            (
                f"cd {wsl_root} ; "
                f"{wsl_py} prediction_update_US.py "
                f"--tickers {','.join(tickers)} "
                f"--summary {wsl_summary}"
                + (f" --start {start}" if start else "")
                + (f" --end {end}" if end else "")
                + (" --strict" if strict else "")
            )
        ]
        try:
            wsl_proc = subprocess.run(
                wsl_cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if wsl_proc.stdout:
                print(wsl_proc.stdout.rstrip())
            if wsl_proc.stderr:
                print(wsl_proc.stderr.rstrip())

            if wsl_summary_path.exists():
                wsl_obj = json.loads(wsl_summary_path.read_text(encoding="utf-8"))
                wsl_obj["runtime"] = "wsl"
                wsl_obj["returncode"] = int(wsl_proc.returncode)
                # Prefer WSL result if it improved (fewer failed, or windows non-zero)
                if (
                    wsl_obj.get("failed") is not None
                    and (
                        proc.returncode != 0
                        or len(wsl_obj.get("failed", [])) < len(obj.get("failed", []))
                    )
                ):
                    obj = wsl_obj
        except Exception as exc:
            obj.setdefault("failed", []).append(
                {"ticker": "<stage>", "status": "failed", "reason": f"WSL fallback error: {exc}"}
            )

    if proc.returncode != 0 and not obj.get("failed"):
        obj["failed"] = [{"ticker": "<stage>", "status": "failed", "reason": f"prediction subprocess exited {proc.returncode}"}]
    return obj


def run_pipeline(
    tickers: list[str],
    start: str | None,
    end: str | None,
    benchmark: str,
    strict: bool,
    universe: str | None = None,
) -> dict:
    summary = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "start": start,
        "end": end,
        "benchmark": benchmark,
    }

    feature_res = _run_feature_update(tickers)
    summary["feature"] = feature_res
    if strict and feature_res["failed"]:
        raise RuntimeError("Feature update failed under strict mode")

    cusum_res = _run_cusum_update(tickers, end=end)
    summary["cusum"] = cusum_res
    if strict and cusum_res["failed"]:
        raise RuntimeError("CUSUM update failed under strict mode")

    pred_res = _run_prediction_update_subprocess(tickers=tickers, start=start, end=end, strict=strict)
    summary["prediction"] = {
        "ok": len(pred_res["ok"]),
        "skipped": len(pred_res["skipped"]),
        "failed": len(pred_res["failed"]),
        "returncode": pred_res.get("returncode", 0),
        "skipped_detail": pred_res["skipped"],
        "failed_detail": pred_res["failed"],
    }

    score_summary = None
    try:
        # NOTE: intentionally do NOT forward the pipeline's daily start/end
        # (which target the feature/CUSUM/prediction refresh window) to the
        # AI-score renderer. AIScore_US filters wide by [start,end]; passing a
        # single-day window collapses to 0-1 rows, triggers the empty-fallback
        # branch, and dumps the full 2006->today history with different
        # smoothing than the previous manual --start 2024-01-01 runs.
        if universe:
            score_summary = generate_ai_score_for_universe(universe=universe, smooth_window=10)
        else:
            score_summary = generate_ai_score(tickers=tickers, benchmark_symbol=benchmark, smooth_window=10)
        summary["ai_score"] = score_summary
    except Exception as exc:
        summary["ai_score"] = {"status": "failed", "reason": str(exc)}
        if strict:
            raise

    try:
        if score_summary and "paths" in score_summary and "snapshot_csv" in score_summary["paths"]:
            tree_summary = generate_ai_tree(snapshot_csv=score_summary["paths"]["snapshot_csv"])
        else:
            tree_summary = generate_ai_tree(snapshot_csv=None)
        summary["ai_tree"] = tree_summary
    except Exception as exc:
        summary["ai_tree"] = {"status": "failed", "reason": str(exc)}
        if strict:
            raise

    summary["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return summary


def main():
    parser = argparse.ArgumentParser(description="US daily inference pipeline: feature -> cusum -> DES -> AI score/tree")
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated tickers; default discover from model_pred_DES_US")
    parser.add_argument("--universe", type=str, default=None, help="preset universe for AI score/tickers: dow30|sox30|ndx100|sp100")
    parser.add_argument("--all-universes", action="store_true", help="run AI score/tree for all preset universes after feature/cusum/prediction")
    parser.add_argument("--start", type=str, default=None, help="start date; default recent trading day")
    parser.add_argument("--end", type=str, default=None, help="end date; default recent trading day")
    parser.add_argument("--benchmark", type=str, default="^NDX", help="benchmark for AI score chart")
    parser.add_argument("--strict", action="store_true", help="fail-fast on any stage error")
    parser.add_argument("--summary", type=str, default=None, help="optional summary json path")
    args = parser.parse_args()

    if args.universe:
        from AIScore_US import resolve_universe

        _, uni_tickers, uni_benchmark = resolve_universe(args.universe)
        tickers = sorted(set(uni_tickers))
        if not tickers:
            raise SystemExit(f"Universe {args.universe} has no tickers")
        benchmark = uni_benchmark or args.benchmark
    else:
        tickers = discover_universe_from_des() if not args.tickers else sorted(set([x.strip().upper() for x in args.tickers.split(",") if x.strip()]))
        benchmark = args.benchmark

    if not tickers:
        raise SystemExit("No tickers found")

    default_day = _recent_trading_day()
    start = args.start or default_day
    end = args.end or default_day

    summary = run_pipeline(
        tickers=tickers,
        start=start,
        end=end,
        benchmark=benchmark,
        strict=args.strict,
        universe=args.universe,
    )

    if args.all_universes:
        all_uni_summary = {}
        for uni in ("dow30", "sox30", "ndx100", "sp100"):
            try:
                # See note in run_pipeline: do NOT pass the feature/prediction
                # window here; AIScore_US uses its own default start
                # (2024-01-01) so the daily series stays comparable to prior
                # manual runs.
                s = generate_ai_score_for_universe(universe=uni, smooth_window=10)
                tree = generate_ai_tree(snapshot_csv=s["paths"]["snapshot_csv"])
                all_uni_summary[uni] = {"ai_score": s, "ai_tree": tree}
            except Exception as exc:
                all_uni_summary[uni] = {"status": "failed", "reason": str(exc)}
                if args.strict:
                    raise
        summary["all_universes"] = all_uni_summary

    if args.summary:
        out_path = Path(args.summary)
    else:
        stamp = pd.Timestamp(end).strftime("%Y%m%d")
        out_path = SUMMARY_ROOT / f"daily_pipeline_summary_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (SUMMARY_ROOT / "daily_pipeline_summary_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] US daily pipeline finished")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
