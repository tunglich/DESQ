"""Command-line entry points for read-only monitoring and synthetic smoke tests."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .adapters import ASPECT_NAMES, adapt_stage2_prediction
from .config import load_contract, load_policy
from .decision import DiagnosticWindow, decide
from .planner import build_plan
from .schemas import MonitoringSnapshot, canonical_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic(stock: str, drift: bool, groups: tuple[str, ...] = ()) -> DiagnosticWindow:
    return DiagnosticWindow(stock, 60, 0.08 if drift else 0.01,
                            0.02 if drift else -0.01, 0.8, 0.7,
                            0.1, 0.2, 0.1, 0.30 if drift else 0.1, groups)


def run_smoke() -> int:
    policy, _ = load_policy()
    stable = decide([_synthetic("2330", False)], [_synthetic("2330", False)], policy)
    local_windows = [_synthetic("2330", True, ("macro",))]
    level_one = decide(local_windows, local_windows, policy)
    level_two = decide(local_windows, local_windows, policy, "failed")
    broad_windows = [_synthetic(str(2300 + index), True,
                                ("fundamental", "trend", "macro")) for index in range(10)]
    level_three = decide(broad_windows, broad_windows, policy, "failed")
    contract, _ = load_contract()
    plans = [build_plan(report, contract) for report in
             (stable, level_one, level_two, level_three)]
    observed = [stable.level, level_one.level, level_two.level, level_three.level]
    if observed != [0, 1, 2, 3]:
        raise RuntimeError(f"monitoring smoke failed: {observed}")
    if any(plan.executable or not plan.dry_run for plan in plans):
        raise RuntimeError("smoke candidate plans must remain non-executable dry runs")
    print(canonical_json({"status": "ok", "levels": observed, "dry_run": True,
                          "plan_ids": [plan.plan_id for plan in plans]}))
    return 0


def collect_stage2(args: argparse.Namespace) -> int:
    contract, contract_hash = load_contract(args.contract)
    _, policy_hash = load_policy(args.policy)
    predictions = args.predictions_dir or REPO_ROOT / "artifacts" / "dflood" / "pred"
    results = []
    for raw_aspect in ASPECT_NAMES:
        path = predictions / f"{args.stock_id}_{raw_aspect}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        results.append(adapt_stage2_prediction(path, args.stock_id, raw_aspect, args.as_of,
                                               contract["label_horizon_trading_days"],
                                               contract["mature_window_trading_days"]))
    cutoffs = {result.mature_label_cutoff for result in results}
    if len(cutoffs) != 1:
        raise RuntimeError(f"Stage 2 mature cutoffs disagree: {sorted(cutoffs)}")
    snapshot = MonitoringSnapshot(
        as_of_date=args.as_of.isoformat(),
        observation_start=min(result.observation_start for result in results),
        observation_end=max(result.observation_end for result in results),
        mature_label_cutoff=max(cutoffs),
        paper_contract_hash=contract_hash,
        policy_hash=policy_hash,
        evaluator_hash=args.evaluator_hash,
        sources=tuple(result.source for result in results),
        metrics=tuple(metric for result in results for metric in result.metrics),
    )
    destination = snapshot.write(args.output_root)
    print(json.dumps({"snapshot_id": snapshot.snapshot_id, "path": str(destination),
                      "statuses": [result.status for result in results]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Run deterministic synthetic Level 0-3 checks")
    collect = subparsers.add_parser("collect-stage2", help="Create a read-only Stage 2 snapshot")
    collect.add_argument("--stock-id", default="2330")
    collect.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    collect.add_argument("--predictions-dir", type=Path)
    collect.add_argument("--output-root", type=Path,
                         default=REPO_ROOT / "artifacts" / "monitoring" / "snapshots")
    collect.add_argument("--evaluator-hash", default="stage2-classification-v1")
    collect.add_argument("--contract", type=Path)
    collect.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        return run_smoke()
    return collect_stage2(args)


if __name__ == "__main__":
    raise SystemExit(main())