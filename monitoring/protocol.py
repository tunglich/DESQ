"""Validated I/O for adjacent-window monitoring decisions."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .decision import DecisionReport, DiagnosticWindow, decide
from .planner import build_plan
from .schemas import (
    AlarmMemory,
    CandidatePlan,
    MetricDeltaMemory,
    ResearchMemoryRecord,
    canonical_json,
    content_hash,
)


SCHEMA_VERSION = "1.0"
FEATURE_GROUPS = {"fundamental", "trend", "momentum", "trade", "macro"}
BATCH_REQUIRED_FIELDS = {
    "schema_version", "monitoring_anchor_index", "observation_start", "observation_end",
    "paper_contract_hash", "policy_hash", "windows",
}
BATCH_OPTIONAL_FIELDS = {"portfolio_window"}
WINDOW_REQUIRED_FIELDS = {
    "stock_id", "sample_count", "precision_gap", "return_gap", "sharpe",
    "information_ratio", "disagreement", "training_disagreement_q90",
    "flooding_upper_fraction", "max_psi", "affected_groups",
}
WINDOW_OPTIONAL_FIELDS = {"regime_signature"}
NUMERIC_WINDOW_FIELDS = WINDOW_REQUIRED_FIELDS - {"stock_id", "sample_count", "affected_groups"}


@dataclass(frozen=True)
class DiagnosticBatch:
    monitoring_anchor_index: int
    observation_start: str
    observation_end: str
    paper_contract_hash: str
    policy_hash: str
    windows: tuple[DiagnosticWindow, ...]
    portfolio_window: DiagnosticWindow | None = None
    schema_version: str = SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def batch_id(self) -> str:
        return content_hash(self.payload())


@dataclass(frozen=True)
class ProtocolEvaluation:
    previous_batch_id: str
    current_batch_id: str
    decision: DecisionReport
    candidate_plan: CandidatePlan
    research_memory: ResearchMemoryRecord
    paper_claim_scope: str = "reference monitoring trigger"
    update_policy_scope: str = "repository operational policy"
    schema_version: str = "1.1"

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def evaluation_id(self) -> str:
        return content_hash(self.payload())

    def write(self, root: Path) -> Path:
        destination = root / self.evaluation_id / "evaluation.json"
        serialized = canonical_json(self.payload()) + "\n"
        memory_destination = destination.with_name("research_memory.json")
        memory_serialized = canonical_json(self.research_memory.payload()) + "\n"
        if destination.exists():
            if destination.read_text(encoding="ascii") != serialized:
                raise RuntimeError(f"immutable evaluation collision: {destination}")
            if (not memory_destination.exists()
                    or memory_destination.read_text(encoding="ascii") != memory_serialized):
                raise RuntimeError(f"immutable research-memory collision: {memory_destination}")
            return destination
        destination.parent.mkdir(parents=True, exist_ok=False)
        destination.write_text(serialized, encoding="ascii")
        memory_destination.write_text(memory_serialized, encoding="ascii")
        return destination


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _parse_window(value: Any, context: str) -> DiagnosticWindow:
    fields = set(value) if isinstance(value, dict) else set()
    if (not isinstance(value, dict) or not WINDOW_REQUIRED_FIELDS <= fields
            or not fields <= WINDOW_REQUIRED_FIELDS | WINDOW_OPTIONAL_FIELDS):
        missing = sorted(WINDOW_REQUIRED_FIELDS - fields)
        extra = sorted(fields - WINDOW_REQUIRED_FIELDS - WINDOW_OPTIONAL_FIELDS)
        raise ValueError(f"{context} has invalid fields; missing={missing}, extra={extra}")
    if not isinstance(value["stock_id"], str) or not value["stock_id"].strip():
        raise ValueError(f"{context} stock_id must be a non-empty string")
    sample_count = value["sample_count"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError(f"{context} sample_count must be a non-negative integer")
    for field in NUMERIC_WINDOW_FIELDS:
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{context} {field} must be a finite number")
    if not -1.0 <= value["precision_gap"] <= 1.0:
        raise ValueError(f"{context} precision_gap must be in [-1, 1]")
    for field in ("disagreement", "training_disagreement_q90", "flooding_upper_fraction"):
        if not 0.0 <= value[field] <= 1.0:
            raise ValueError(f"{context} {field} must be in [0, 1]")
    if value["max_psi"] < 0.0:
        raise ValueError(f"{context} max_psi must be non-negative")
    groups = value["affected_groups"]
    if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
        raise ValueError(f"{context} affected_groups must be an array of strings")
    unknown_groups = set(groups) - FEATURE_GROUPS
    if unknown_groups or len(groups) != len(set(groups)):
        raise ValueError(f"{context} has invalid affected_groups: {sorted(unknown_groups)}")
    regime = value.get("regime_signature")
    if regime is not None and (not isinstance(regime, str) or not regime.strip()):
        raise ValueError(f"{context} regime_signature must be null or a non-empty string")
    return DiagnosticWindow(**{**value, "affected_groups": tuple(groups)})


def _build_research_memory(current: DiagnosticBatch, previous: DiagnosticBatch,
                           report: DecisionReport, plan: CandidatePlan,
                           promotion_status: str) -> ResearchMemoryRecord:
    previous_by_stock = {window.stock_id: window for window in previous.windows}
    current_by_stock = {window.stock_id: window for window in current.windows}
    deltas = tuple(
        MetricDeltaMemory(
            stock_id=stock_id,
            precision=previous_by_stock[stock_id].precision_gap
            - current_by_stock[stock_id].precision_gap,
            sharpe=current_by_stock[stock_id].sharpe - previous_by_stock[stock_id].sharpe,
            disagreement=(current_by_stock[stock_id].disagreement
                          - previous_by_stock[stock_id].disagreement),
            drift=current_by_stock[stock_id].max_psi - previous_by_stock[stock_id].max_psi,
        )
        for stock_id in sorted(current_by_stock)
    )
    promoted_statuses = {"threshold_promoted", "weights_promoted", "promoted"}
    failed_statuses = {"threshold_failed", "weights_failed", "failed"}
    promoted = (True if promotion_status in promoted_statuses
                else False if promotion_status in failed_statuses else None)
    return ResearchMemoryRecord(
        previous_batch_id=previous.batch_id,
        current_batch_id=current.batch_id,
        window_start=current.observation_start,
        window_end=current.observation_end,
        alarms=tuple(AlarmMemory(decision.stock_id, decision.previous_alarms,
                                 decision.current_alarms)
                     for decision in report.stock_decisions),
        regime_signatures=tuple(sorted({window.regime_signature for window in current.windows
                                        if window.regime_signature is not None})),
        candidate_plan_id=plan.plan_id,
        candidate_parameters=tuple(sorted({parameter for step in plan.steps
                                           for parameter in step.changed_parameters})),
        promotion_status=promotion_status,
        promoted=promoted,
        metric_deltas=deltas,
    )


def load_diagnostic_batch(path: Path) -> DiagnosticBatch:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError(f"diagnostic batch must be an object: {path}")
    fields = set(value)
    if not BATCH_REQUIRED_FIELDS <= fields or not fields <= BATCH_REQUIRED_FIELDS | BATCH_OPTIONAL_FIELDS:
        raise ValueError(
            f"invalid diagnostic batch fields: missing={sorted(BATCH_REQUIRED_FIELDS - fields)}, "
            f"extra={sorted(fields - BATCH_REQUIRED_FIELDS - BATCH_OPTIONAL_FIELDS)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported diagnostic schema_version: {value['schema_version']}")
    if not isinstance(value["windows"], list):
        raise ValueError(f"diagnostic batch must contain a windows array: {path}")
    anchor_index = value["monitoring_anchor_index"]
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, int) or anchor_index < 0:
        raise ValueError("monitoring_anchor_index must be a non-negative integer")
    try:
        windows = tuple(_parse_window(row, f"windows[{index}]")
                        for index, row in enumerate(value["windows"]))
        portfolio_value = value.get("portfolio_window")
        portfolio_window = (_parse_window(portfolio_value, "portfolio_window")
                            if portfolio_value is not None else None)
        batch = DiagnosticBatch(
            monitoring_anchor_index=anchor_index,
            observation_start=value["observation_start"],
            observation_end=value["observation_end"],
            paper_contract_hash=value["paper_contract_hash"],
            policy_hash=value["policy_hash"],
            windows=windows,
            portfolio_window=portfolio_window,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid diagnostic batch {path}: {error}") from error
    if not all(isinstance(value, str) and value for value in
               (batch.observation_start, batch.observation_end,
                batch.paper_contract_hash, batch.policy_hash)):
        raise ValueError(f"diagnostic batch dates and hashes must be non-empty strings: {path}")
    start = date.fromisoformat(batch.observation_start)
    end = date.fromisoformat(batch.observation_end)
    if start > end:
        raise ValueError(f"diagnostic batch start is after end: {path}")
    stock_ids = [window.stock_id for window in batch.windows]
    if not stock_ids or len(stock_ids) != len(set(stock_ids)):
        raise ValueError(f"diagnostic batch requires one window per stock: {path}")
    return batch


def evaluate_batches(current: DiagnosticBatch, previous: DiagnosticBatch,
                     contract: dict[str, Any], contract_hash: str,
                     policy_hash: str, policy: dict[str, Any],
                     recalibration_status: str = "not_evaluated") -> ProtocolEvaluation:
    for name, batch in (("current", current), ("previous", previous)):
        if batch.paper_contract_hash != contract_hash:
            raise ValueError(f"{name} batch reference contract hash does not match active contract")
        if batch.policy_hash != policy_hash:
            raise ValueError(f"{name} batch policy hash does not match active policy")
    if date.fromisoformat(previous.observation_end) >= date.fromisoformat(current.observation_start):
        raise ValueError("previous and current mature windows must be ordered and non-overlapping")
    expected_gap = int(contract["monitoring_lookback_trading_days"])
    if current.monitoring_anchor_index - previous.monitoring_anchor_index != expected_gap:
        raise ValueError(f"monitoring anchor indices must differ by exactly {expected_gap}")
    current_stocks = {window.stock_id for window in current.windows}
    previous_stocks = {window.stock_id for window in previous.windows}
    if current_stocks != previous_stocks:
        raise ValueError("previous and current batches must contain the same stocks")
    mature_count = int(contract["mature_anchor_count"])
    all_windows = (*current.windows, *previous.windows,
                   *(() if current.portfolio_window is None else (current.portfolio_window,)),
                   *(() if previous.portfolio_window is None else (previous.portfolio_window,)))
    if any(window.sample_count != mature_count for window in all_windows):
        raise ValueError(f"every diagnostic window must contain exactly {mature_count} mature anchors")
    report = decide(
        current.windows,
        previous.windows,
        policy,
        recalibration_status,
        mature_count,
        current.portfolio_window,
        previous.portfolio_window,
    )
    plan = build_plan(report, policy)
    memory = _build_research_memory(current, previous, report, plan, recalibration_status)
    return ProtocolEvaluation(previous.batch_id, current.batch_id, report, plan, memory)