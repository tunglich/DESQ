"""Pure metric functions for the reference monitoring equations."""
from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Sequence


EPSILON = 1e-12


def _paired(left: Sequence[float], right: Sequence[float]) -> None:
    if not left or len(left) != len(right):
        raise ValueError("metric inputs must be non-empty and have equal length")


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    _paired(left, right)
    left_mean, right_mean = fmean(left), fmean(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator > EPSILON else None


def matured_direction_label(anchor_price: float, forward_prices: Sequence[float]) -> int:
    """Equation (10): label whether the mean forward price exceeds the anchor."""
    if not math.isfinite(anchor_price) or anchor_price <= 0.0 or len(forward_prices) != 20:
        raise ValueError("Eq. (10) requires a positive anchor and exactly 20 forward prices")
    if any(not math.isfinite(price) or price <= 0.0 for price in forward_prices):
        raise ValueError("forward prices must be positive")
    return int(fmean(forward_prices) > anchor_price)


def forward_return(anchor_price: float, terminal_price: float) -> float:
    """Equation (12): endpoint return over the matured forecast horizon."""
    if (not math.isfinite(anchor_price) or not math.isfinite(terminal_price)
            or anchor_price <= 0.0 or terminal_price <= 0.0):
        raise ValueError("anchor and terminal prices must be finite and positive")
    return (terminal_price - anchor_price) / anchor_price


def rolling_precision(signals: Sequence[int], labels: Sequence[int],
                      epsilon: float = EPSILON) -> float:
    """Equation (11): positive-signal precision over a mature window."""
    _paired(signals, labels)
    if any(value not in (0, 1) for value in (*signals, *labels)):
        raise ValueError("signals and labels must be binary")
    true_positives = sum(signal == label == 1 for signal, label in zip(signals, labels))
    return true_positives / (sum(signals) + epsilon)


def information_coefficient(probabilities: Sequence[float],
                            forward_returns: Sequence[float]) -> float | None:
    """Equation (12): correlation between DES probabilities and forward returns."""
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if any(not math.isfinite(value) for value in forward_returns):
        raise ValueError("forward returns must be finite")
    return _correlation(probabilities, forward_returns)


def annualized_sharpe(returns: Sequence[float], epsilon: float = EPSILON) -> float:
    """Equation (13): annualized rolling Sharpe using population volatility."""
    if not returns:
        raise ValueError("returns must be non-empty")
    mean_return = fmean(returns)
    volatility = math.sqrt(fmean((value - mean_return) ** 2 for value in returns))
    return mean_return / (volatility + epsilon) * math.sqrt(252.0)


def annualized_information_ratio(strategy_returns: Sequence[float],
                                 benchmark_returns: Sequence[float],
                                 epsilon: float = EPSILON) -> float:
    """Equation (13): annualized information ratio of active returns."""
    _paired(strategy_returns, benchmark_returns)
    active = [strategy - benchmark
              for strategy, benchmark in zip(strategy_returns, benchmark_returns)]
    return annualized_sharpe(active, epsilon)


def precision_gap(validation_precision: float, live_precision: float) -> float:
    """Equation (14): validation-to-live precision degradation."""
    return validation_precision - live_precision


def implied_net_return(action: int, probability: float, neighborhood_magnitude: float,
                       transaction_cost: float) -> float:
    """Equation (15): DES-implied net return for one action."""
    if action not in (-1, 0, 1):
        raise ValueError("action must be -1, 0, or 1")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if neighborhood_magnitude < 0.0 or transaction_cost < 0.0:
        raise ValueError("magnitude and transaction cost must be non-negative")
    return action * (2.0 * probability - 1.0) * neighborhood_magnitude - transaction_cost


def live_return_gap(implied_returns: Sequence[float], live_returns: Sequence[float]) -> float:
    """Equation (16): mean DES-implied return minus realized live return."""
    _paired(implied_returns, live_returns)
    return fmean(implied - live for implied, live in zip(implied_returns, live_returns))


def capital_weighted_return(weights: Sequence[float], returns: Sequence[float]) -> float:
    """Equation (17): one-period capital-weighted portfolio return."""
    _paired(weights, returns)
    if any(weight < 0.0 for weight in weights) or not math.isclose(sum(weights), 1.0,
                                                                   abs_tol=1e-8):
        raise ValueError("portfolio weights must be non-negative and sum to one")
    return sum(weight * value for weight, value in zip(weights, returns))


def des_disagreement(specialist_probabilities: Sequence[float]) -> float:
    """Equation (18): mean absolute deviation among five specialist probabilities."""
    if len(specialist_probabilities) != 5:
        raise ValueError("DES disagreement requires five specialist probabilities")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
           for value in specialist_probabilities):
        raise ValueError("specialist probabilities must be finite and in [0, 1]")
    mean_probability = fmean(specialist_probabilities)
    return fmean(abs(value - mean_probability) for value in specialist_probabilities)


def flooding_upper_fraction(levels: Iterable[float], upper_bound: float,
                            tolerance: float = EPSILON) -> float:
    """Equation (19): fraction of recent epochs at the Dynamic-Flooding upper bound."""
    values = list(levels)
    if not values:
        raise ValueError("flooding levels must be non-empty")
    return sum(math.isclose(value, upper_bound, abs_tol=tolerance) for value in values) / len(values)


def population_stability_index(expected_shares: Sequence[float],
                               actual_shares: Sequence[float],
                               epsilon: float = EPSILON) -> float:
    """Equation (20): population stability index for one feature group."""
    _paired(expected_shares, actual_shares)
    if any(value < 0.0 for value in (*expected_shares, *actual_shares)):
        raise ValueError("PSI shares must be non-negative")
    if not math.isclose(sum(expected_shares), 1.0, abs_tol=1e-8):
        raise ValueError("expected PSI shares must sum to one")
    if not math.isclose(sum(actual_shares), 1.0, abs_tol=1e-8):
        raise ValueError("actual PSI shares must sum to one")
    return sum((actual - expected) * math.log((actual + epsilon) / (expected + epsilon))
               for expected, actual in zip(expected_shares, actual_shares))