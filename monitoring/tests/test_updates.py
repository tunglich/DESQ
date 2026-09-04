from __future__ import annotations

import unittest

from monitoring.updates import (
    competence_score,
    cost_aware_objective,
    deployable_weights,
    select_threshold,
    softmax_weights,
)


class UpdatesTest(unittest.TestCase):
    def test_threshold_objective_and_selection(self) -> None:
        coefficients = {
            "precision": 2.0, "f1": 1.0, "sharpe": 1.0,
            "information_ratio": 1.0, "turnover": 1.0, "drawdown": 1.0,
        }
        baseline = {
            "precision": 0.6, "f1": 0.5, "sharpe": 0.4,
            "information_ratio": 0.3, "turnover": 0.2, "drawdown": 0.1,
        }
        self.assertAlmostEqual(cost_aware_objective(baseline, coefficients), 2.1)
        better = {**baseline, "precision": 0.7}
        threshold, score = select_threshold({0.50: baseline, 0.55: better}, coefficients)
        self.assertEqual(threshold, 0.55)
        self.assertAlmostEqual(score, 2.3)

    def test_competence_softmax_shrinkage_and_floor(self) -> None:
        metrics = {
            "precision": 0.7, "information_coefficient": 0.2, "sharpe": 0.4,
            "drawdown": 0.1, "turnover": 0.2, "psi": 0.05,
        }
        coefficients = {name: 1.0 for name in metrics}
        self.assertAlmostEqual(competence_score(metrics, coefficients), 0.95)
        candidate = softmax_weights([2.0, 1.0], 1.0)
        self.assertAlmostEqual(sum(candidate), 1.0)
        deployed = deployable_weights([0.5, 0.5], candidate, 0.5, 0.2)
        self.assertAlmostEqual(sum(deployed), 1.0)
        self.assertGreater(deployed[0], deployed[1])

    def test_invalid_weight_parameters_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            softmax_weights([1.0], 0.0)
        with self.assertRaises(ValueError):
            deployable_weights([0.5, 0.5], [0.5, 0.5], 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()