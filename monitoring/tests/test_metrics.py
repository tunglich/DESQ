from __future__ import annotations

import math
import unittest

from monitoring.metrics import (
    annualized_information_ratio,
    annualized_sharpe,
    capital_weighted_return,
    des_disagreement,
    flooding_upper_fraction,
    forward_return,
    implied_net_return,
    information_coefficient,
    live_return_gap,
    matured_direction_label,
    population_stability_index,
    precision_gap,
    rolling_precision,
)


class MetricsTest(unittest.TestCase):
    def test_predictive_and_return_equations(self) -> None:
        self.assertEqual(matured_direction_label(100.0, [101.0] * 20), 1)
        self.assertAlmostEqual(forward_return(100.0, 110.0), 0.1)
        self.assertAlmostEqual(rolling_precision([1, 1, 0], [1, 0, 1]), 0.5)
        self.assertAlmostEqual(information_coefficient([0.1, 0.5, 0.9], [-0.2, 0.0, 0.2]), 1.0)
        self.assertAlmostEqual(precision_gap(0.72, 0.63), 0.09)
        implied = implied_net_return(1, 0.75, 0.10, 0.005)
        self.assertAlmostEqual(implied, 0.045)
        self.assertAlmostEqual(live_return_gap([implied, 0.02], [0.01, 0.01]), 0.0225)

    def test_risk_and_portfolio_equations(self) -> None:
        returns = [0.01, 0.02, 0.03]
        benchmark = [0.0, 0.01, 0.02]
        self.assertGreater(annualized_sharpe(returns), 0.0)
        self.assertGreater(annualized_information_ratio(returns, benchmark), 0.0)
        self.assertAlmostEqual(capital_weighted_return([0.25, 0.75], [0.1, 0.2]), 0.175)

    def test_stability_and_drift_equations(self) -> None:
        self.assertAlmostEqual(des_disagreement([0.1, 0.2, 0.3, 0.4, 0.5]), 0.12)
        self.assertAlmostEqual(flooding_upper_fraction([0.1, 0.4, 0.4, 0.2], 0.4), 0.5)
        psi = population_stability_index([0.5, 0.5], [0.25, 0.75])
        self.assertAlmostEqual(psi, 0.25 * math.log(3.0))

    def test_invalid_shapes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            matured_direction_label(100.0, [101.0] * 19)
        with self.assertRaises(ValueError):
            information_coefficient([1.1, 0.2], [0.1, 0.2])
        with self.assertRaises(ValueError):
            des_disagreement([0.1, 0.2])
        with self.assertRaises(ValueError):
            population_stability_index([0.5, 0.5], [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()