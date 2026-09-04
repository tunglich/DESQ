"""Reference alarms plus repository-defined update planning policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


ALARM_NAMES = (
    "precision_gap",
    "return_gap",
    "risk",
    "disagreement",
    "flooding_saturation",
    "feature_drift",
)


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
    regime_signature: str | None = None


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
        "precision_gap": window.precision_gap > float(thresholds["precision_gap"]),
        "return_gap": window.return_gap > float(thresholds["return_gap"]),
        "risk": window.sharpe < 0.0 or window.information_ratio < 0.0,
        "disagreement": window.disagreement > window.training_disagreement_q90,
        "flooding_saturation": (
            window.flooding_upper_fraction
            > float(thresholds["flooding_upper_fraction"])
        ),
        "feature_drift": window.max_psi > 0.25,
    }
    return tuple(name for name in ALARM_NAMES if alarms[name])


def decide_stock(current: DiagnosticWindow, previous: DiagnosticWindow,
                 policy: dict[str, Any], minimum_mature_anchors: int = 40) -> StockDecision:
    minimum = minimum_mature_anchors
    if current.stock_id != previous.stock_id:
        raise ValueError("current and previous windows must refer to the same stock")
    if current.sample_count != minimum or previous.sample_count != minimum:
        return StockDecision(current.stock_id, False, (), (), False, (),
                             f"invalid_window: require exactly {minimum} mature anchors in both windows")
    current_alarms = compute_alarms(current, policy)
    previous_alarms = compute_alarms(previous, policy)
    triggered = len(current_alarms) >= 2 and len(previous_alarms) >= 2
    groups = current.affected_groups if triggered else ()
    reason = ("Appendix F trigger: at least two alarms fired in adjacent mature windows"
              if triggered else "Reference mature-window trigger did not fire")
    return StockDecision(current.stock_id, True, current_alarms, previous_alarms,
                         triggered, groups, reason)


def decide(current: Iterable[DiagnosticWindow], previous: Iterable[DiagnosticWindow],
           policy: dict[str, Any], recalibration_status: str = "not_evaluated",
           minimum_mature_anchors: int = 40,
           portfolio_current: DiagnosticWindow | None = None,
           portfolio_previous: DiagnosticWindow | None = None) -> DecisionReport:
    current_windows = tuple(current)
    previous_by_stock = {window.stock_id: window for window in previous}
    stock_decisions = [
        decide_stock(window, previous_by_stock[window.stock_id], policy,
                     minimum_mature_anchors)
        for window in sorted(current_windows, key=lambda item: item.stock_id)
        if window.stock_id in previous_by_stock
    ]
    triggered = tuple(item for item in stock_decisions if item.update_triggered)
    affected_stocks = {item.stock_id for item in triggered}
    affected_groups = {group for item in triggered for group in item.affected_groups}
    if (portfolio_current is None) != (portfolio_previous is None):
        raise ValueError("portfolio current and previous windows must be supplied together")
    if portfolio_current is not None and portfolio_previous is not None:
        portfolio_decision = decide_stock(portfolio_current, portfolio_previous, policy,
                                          minimum_mature_anchors)
        stock_decisions.append(portfolio_decision)
        if portfolio_decision.update_triggered:
            portfolio_groups = set(portfolio_decision.affected_groups)
            matched_by_group = {
                window.stock_id for window in current_windows
                if portfolio_groups.intersection(window.affected_groups)
            }
            matched_by_regime = {
                window.stock_id for window in current_windows
                if portfolio_current.regime_signature
                and window.regime_signature == portfolio_current.regime_signature
            }
            matched = matched_by_group | matched_by_regime
            if len(matched) >= 2:
                affected_stocks.update(matched)
                affected_groups.update(portfolio_groups)
    stocks = tuple(sorted(affected_stocks))
    groups = tuple(sorted(affected_groups))
    decisions = tuple(stock_decisions)
    if not stocks:
        return DecisionReport(0, "no_update", True, decisions, (), ())
    if recalibration_status == "not_evaluated":
        return DecisionReport(1, "evaluate_level_1_threshold", True, decisions, stocks, groups)
    if recalibration_status in {"threshold_promoted", "promoted"}:
        return DecisionReport(1, "record_level_1_threshold_promotion", True,
                              decisions, stocks, groups)
    if recalibration_status == "threshold_failed":
        return DecisionReport(1, "evaluate_level_1_des_weights", True, decisions, stocks, groups)
    if recalibration_status == "weights_promoted":
        return DecisionReport(1, "record_level_1_des_weights_promotion", True,
                              decisions, stocks, groups)
    if recalibration_status not in {"weights_failed", "failed"}:
        raise ValueError(
            "recalibration_status must be not_evaluated, threshold_promoted, "
            "threshold_failed, weights_promoted, or weights_failed"
        )
    broad = policy["broad_drift"]
    if len(stocks) >= int(broad["minimum_stocks"]) or len(groups) >= int(broad["minimum_feature_groups"]):
        return DecisionReport(3, "full_retraining", True, decisions, stocks, groups)
    return DecisionReport(2, "local_fine_tuning_and_des_refit", True,
                          decisions, stocks, groups)


def promotion_allowed(delta_precision: float, delta_sharpe: float,
                      delta_turnover: float, delta_drawdown: float,
                      changed_parameters: set[str], policy: dict[str, Any],
                      immutable_unchanged: bool) -> bool:
    limits = policy["promotion"]
    allowed = set(policy["repository_update_extension"]["theta_allow"])
    return (
        delta_precision > float(limits["minimum_precision_improvement"])
        and delta_sharpe > -float(limits["maximum_sharpe_decrease"])
        and delta_turnover < float(limits["maximum_turnover_increase"])
        and delta_drawdown < float(limits["maximum_drawdown_increase"])
        and changed_parameters <= allowed
        and immutable_unchanged
    )