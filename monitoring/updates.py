"""Pure Appendix F recalibration equations for dry-run candidate evaluation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


OBJECTIVE_TERMS = ("precision", "f1", "sharpe", "information_ratio", "turnover", "drawdown")
COMPETENCE_TERMS = ("precision", "information_coefficient", "sharpe", "drawdown", "turnover", "psi")


def cost_aware_objective(metrics: Mapping[str, float], coefficients: Mapping[str, float]) -> float:
    """Equation (25): reward predictive/risk metrics and penalize cost and drawdown."""
    _require_terms(metrics, coefficients, OBJECTIVE_TERMS)
    return (
        coefficients["precision"] * metrics["precision"]
        + coefficients["f1"] * metrics["f1"]
        + coefficients["sharpe"] * metrics["sharpe"]
        + coefficients["information_ratio"] * metrics["information_ratio"]
        - coefficients["turnover"] * metrics["turnover"]
        - coefficients["drawdown"] * metrics["drawdown"]
    )


def select_threshold(candidate_metrics: Mapping[float, Mapping[str, float]],
                     coefficients: Mapping[str, float]) -> tuple[float, float]:
    """Equation (26): select the deterministic maximum-objective threshold."""
    if not candidate_metrics:
        raise ValueError("threshold candidates must be non-empty")
    scored = [(cost_aware_objective(metrics, coefficients), threshold)
              for threshold, metrics in candidate_metrics.items()]
    score, threshold = max(scored, key=lambda item: (item[0], -item[1]))
    return threshold, score


def competence_score(metrics: Mapping[str, float], coefficients: Mapping[str, float]) -> float:
    """Equation (29): score one specialist's mature-window competence."""
    _require_terms(metrics, coefficients, COMPETENCE_TERMS)
    return (
        coefficients["precision"] * metrics["precision"]
        + coefficients["information_coefficient"] * metrics["information_coefficient"]
        + coefficients["sharpe"] * metrics["sharpe"]
        - coefficients["drawdown"] * metrics["drawdown"]
        - coefficients["turnover"] * metrics["turnover"]
        - coefficients["psi"] * metrics["psi"]
    )


def softmax_weights(competence: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Equation (30): numerically stable temperature-controlled DES weights."""
    if not competence or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("competence must be non-empty and temperature must be positive")
    if any(not math.isfinite(value) for value in competence):
        raise ValueError("competence values must be finite")
    scaled = [value / temperature for value in competence]
    offset = max(scaled)
    exponentials = [math.exp(value - offset) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def deployable_weights(incumbent: Sequence[float], candidate: Sequence[float],
                       shrinkage: float, minimum_weight: float = 0.0) -> tuple[float, ...]:
    """Equations (31)-(32): shrink to incumbent, apply a floor, and renormalize."""
    if not incumbent or len(incumbent) != len(candidate):
        raise ValueError("incumbent and candidate weights must be non-empty and equally sized")
    if not 0.0 < shrinkage <= 1.0 or not 0.0 <= minimum_weight < 1.0:
        raise ValueError("shrinkage must be in (0, 1] and minimum_weight in [0, 1)")
    if minimum_weight * len(incumbent) >= 1.0:
        raise ValueError("minimum_weight leaves no feasible simplex")
    for name, values in (("incumbent", incumbent), ("candidate", candidate)):
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"{name} weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise ValueError(f"{name} weights must sum to one")
    shrunk = [(1.0 - shrinkage) * old + shrinkage * new
              for old, new in zip(incumbent, candidate)]
    floored = [max(minimum_weight, value) for value in shrunk]
    total = sum(floored)
    return tuple(value / total for value in floored)


def _require_terms(metrics: Mapping[str, float], coefficients: Mapping[str, float],
                   terms: Sequence[str]) -> None:
    missing = set(terms) - set(metrics) | (set(terms) - set(coefficients))
    if missing:
        raise ValueError(f"missing equation terms: {sorted(missing)}")
    values = [metrics[term] for term in terms] + [coefficients[term] for term in terms]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("equation inputs must be finite")
    if any(coefficients[term] < 0.0 for term in terms):
        raise ValueError("equation coefficients must be non-negative")