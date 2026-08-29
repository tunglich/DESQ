"""Paper Appendix-F alarm, escalation, and promotion decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


ALARM_NAMES = ("precision", "return", "risk", "disagreement", "flooding", "feature_drift")


@dataclass(frozen=True)
class DiagnosticWindow:
    stock_id: str
    sample_count: int
    precision_gap: float
    return_gap: float
    sharpe: float
    information_ratio: float
    disagreement: float
    training_disagreement_q90: float
    flooding_upper_fraction: float
    max_psi: float
    affected_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class StockDecision:
    stock_id: str
    eligible: bool
    current_alarms: tuple[str, ...]
    previous_alarms: tuple[str, ...]
    update_triggered: bool
    affected_groups: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DecisionReport:
    level: int
    action: str
    dry_run: bool
    stock_decisions: tuple[StockDecision, ...]
    affected_stocks: tuple[str, ...]
    affected_groups: tuple[str, ...]
    threshold_source: str = "operational_policy"

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def compute_alarms(window: DiagnosticWindow, policy: dict[str, Any]) -> tuple[str, ...]:
    thresholds = policy["alarm_thresholds"]
    alarms = {
        "precision": window.precision_gap > float(thresholds["precision_gap"]),
        "return": window.return_gap > float(thresholds["return_gap"]),
        "risk": window.sharpe < 0.0 or window.information_ratio < 0.0,
        "disagreement": window.disagreement > window.training_disagreement_q90,
        "flooding": window.flooding_upper_fraction > float(thresholds["flooding_upper_fraction"]),
        "feature_drift": window.max_psi > 0.25,
    }
    return tuple(name for name in ALARM_NAMES if alarms[name])


def decide_stock(current: DiagnosticWindow, previous: DiagnosticWindow,
                 policy: dict[str, Any]) -> StockDecision:
    minimum = int(policy["minimum_mature_anchors"])
    if current.stock_id != previous.stock_id:
        raise ValueError("current and previous windows must refer to the same stock")
    if current.sample_count < minimum or previous.sample_count < minimum:
        return StockDecision(current.stock_id, False, (), (), False, (),
                             f"insufficient_data: need {minimum} mature anchors in both windows")
    current_alarms = compute_alarms(current, policy)
    previous_alarms = compute_alarms(previous, policy)
    triggered = len(current_alarms) >= 2 and len(previous_alarms) >= 2
    groups = current.affected_groups if triggered else ()
    reason = ("Eq. (22): at least two alarms fired in two consecutive mature windows"
              if triggered else "Eq. (22) did not fire")
    return StockDecision(current.stock_id, True, current_alarms, previous_alarms,
                         triggered, groups, reason)


def decide(current: Iterable[DiagnosticWindow], previous: Iterable[DiagnosticWindow],
           policy: dict[str, Any], recalibration_status: str = "not_evaluated") -> DecisionReport:
    previous_by_stock = {window.stock_id: window for window in previous}
    stock_decisions = tuple(
        decide_stock(window, previous_by_stock[window.stock_id], policy)
        for window in sorted(current, key=lambda item: item.stock_id)
        if window.stock_id in previous_by_stock
    )
    triggered = tuple(item for item in stock_decisions if item.update_triggered)
    stocks = tuple(item.stock_id for item in triggered)
    groups = tuple(sorted({group for item in triggered for group in item.affected_groups}))
    if not triggered:
        return DecisionReport(0, "no_update", True, stock_decisions, (), ())
    if recalibration_status in {"not_evaluated", "promoted"}:
        action = "evaluate_level_1_recalibration" if recalibration_status == "not_evaluated" else "deploy_level_1"
        return DecisionReport(1, action, True, stock_decisions, stocks, groups)
    if recalibration_status != "failed":
        raise ValueError("recalibration_status must be not_evaluated, promoted, or failed")
    broad = policy["broad_drift"]
    if len(stocks) >= int(broad["minimum_stocks"]) or len(groups) >= int(broad["minimum_feature_groups"]):
        return DecisionReport(3, "full_retraining", True, stock_decisions, stocks, groups)
    return DecisionReport(2, "local_fine_tuning_and_des_refit", True,
                          stock_decisions, stocks, groups)


def promotion_allowed(delta_precision: float, delta_sharpe: float,
                      delta_turnover: float, delta_drawdown: float,
                      changed_parameters: set[str], contract: dict[str, Any],
                      policy: dict[str, Any], immutable_unchanged: bool) -> bool:
    limits = policy["promotion"]
    allowed = set(contract["theta_allow"])
    return (
        delta_precision > float(limits["minimum_precision_improvement"])
        and delta_sharpe > -float(limits["maximum_sharpe_decrease"])
        and delta_turnover < float(limits["maximum_turnover_increase"])
        and delta_drawdown < float(limits["maximum_drawdown_increase"])
        and changed_parameters <= allowed
        and immutable_unchanged
    )