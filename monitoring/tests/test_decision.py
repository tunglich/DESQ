from __future__ import annotations

import unittest

from monitoring.config import load_contract, load_policy
from monitoring.decision import DiagnosticWindow, decide, promotion_allowed
from monitoring.planner import build_plan


def window(stock: str, *, alarms: bool, groups: tuple[str, ...] = (), samples: int = 40) -> DiagnosticWindow:
    return DiagnosticWindow(
        stock_id=stock,
        sample_count=samples,
        precision_gap=0.06 if alarms else 0.01,
        return_gap=0.01 if alarms else -0.01,
        sharpe=1.0,
        information_ratio=1.0,
        disagreement=0.1,
        training_disagreement_q90=0.2,
        flooding_upper_fraction=0.1,
        max_psi=0.3 if alarms else 0.1,
        affected_groups=groups,
    )


class DecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.contract_hash = load_contract()
        cls.policy, cls.policy_hash = load_policy()

    def test_stable_windows_are_level_zero(self) -> None:
        report = decide([window("2330", alarms=False)], [window("2330", alarms=False)], self.policy)
        self.assertEqual((report.level, report.action), (0, "no_update"))

    def test_two_consecutive_alarm_windows_queue_level_one(self) -> None:
        current = [window("2330", alarms=True, groups=("macro",))]
        report = decide(current, current, self.policy)
        self.assertEqual((report.level, report.action), (1, "evaluate_level_1_threshold"))

    def test_failed_recalibration_escalates_local_and_broad_drift(self) -> None:
        local = [window("2330", alarms=True, groups=("macro",))]
        self.assertEqual(decide(local, local, self.policy, "weights_failed").level, 2)
        broad = [window(str(2300 + index), alarms=True,
                        groups=("fundamental", "trend", "macro")) for index in range(10)]
        self.assertEqual(decide(broad, broad, self.policy, "weights_failed").level, 3)

    def test_insufficient_data_never_triggers(self) -> None:
        short = [window("2330", alarms=True, samples=39)]
        report = decide(short, short, self.policy)
        self.assertEqual(report.level, 0)
        self.assertFalse(report.stock_decisions[0].eligible)
        long = [window("2330", alarms=True, samples=41)]
        self.assertFalse(decide(long, long, self.policy).stock_decisions[0].eligible)

    def test_alarm_thresholds_are_strict(self) -> None:
        exact = DiagnosticWindow("2330", 40, 0.05, 0.0, 0.0, 0.0,
                                 0.2, 0.2, 0.25, 0.25, ("macro",))
        report = decide([exact], [exact], self.policy)
        self.assertEqual(report.stock_decisions[0].current_alarms, ())

    def test_appendix_trigger_uses_six_named_alarm_types(self) -> None:
        risk = DiagnosticWindow("2330", 40, 0.01, 0.0, -0.1, 0.1,
                                0.1, 0.2, 0.3, 0.1, ("macro",))
        report = decide([risk], [risk], self.policy)
        self.assertEqual(
            report.stock_decisions[0].current_alarms,
            ("risk", "flooding_saturation"),
        )
        self.assertTrue(report.stock_decisions[0].update_triggered)

    def test_level_one_threshold_then_weights_before_escalation(self) -> None:
        local = [window("2330", alarms=True, groups=("macro",))]
        threshold_plan = build_plan(decide(local, local, self.policy), self.policy)
        self.assertEqual(threshold_plan.steps[0].name, "sealed_threshold_recalibration")
        self.assertEqual(
            decide(local, local, self.policy, "threshold_failed").action,
            "evaluate_level_1_des_weights",
        )
        weight_plan = build_plan(
            decide(local, local, self.policy, "threshold_failed"), self.policy
        )
        self.assertEqual(weight_plan.steps[0].name, "sealed_des_weight_recalibration")
        self.assertEqual(
            decide(local, local, self.policy, "weights_promoted").action,
            "record_level_1_des_weights_promotion",
        )

    def test_repository_promotion_gate(self) -> None:
        self.assertTrue(promotion_allowed(0.01, 0.01, -0.01, -0.01,
                                          {"des_threshold"}, self.policy, True))
        self.assertFalse(promotion_allowed(0.01, 0.0, 0.0, 0.0,
                           {"des_threshold"}, self.policy, True))
        self.assertFalse(promotion_allowed(0.01, 0.01, -0.01, -0.01,
                                           {"label_definition"}, self.policy, True))
        self.assertFalse(promotion_allowed(0.01, 0.01, -0.01, -0.01,
                                           {"des_threshold"}, self.policy, False))

    def test_candidate_plan_is_dry_run_and_within_theta_allow(self) -> None:
        windows = [window("2330", alarms=True, groups=("macro",))]
        report = decide(windows, windows, self.policy, "weights_failed")
        plan = build_plan(report, self.policy)
        self.assertEqual(plan.decision_level, 2)
        self.assertTrue(plan.dry_run)
        self.assertFalse(plan.executable)
        changed = {parameter for step in plan.steps for parameter in step.changed_parameters}
        self.assertLessEqual(
            changed, set(self.policy["repository_update_extension"]["theta_allow"]))

    def test_portfolio_alarm_can_batch_shared_group(self) -> None:
        stocks = [
            DiagnosticWindow(stock, 40, 0.06, 0.0, 1.0, 1.0, 0.1, 0.2,
                             0.1, 0.1, ("macro",))
            for stock in ("2330", "2454")
        ]
        portfolio = DiagnosticWindow("TW50_PORTFOLIO", 40, 0.06, 0.0, 1.0, 1.0,
                                     0.1, 0.2, 0.1, 0.3, ("macro",))
        report = decide(stocks, stocks, self.policy,
                        portfolio_current=portfolio, portfolio_previous=portfolio)
        self.assertEqual(report.level, 1)
        self.assertEqual(report.affected_stocks, ("2330", "2454"))

    def test_portfolio_alarm_can_batch_shared_regime(self) -> None:
        stocks = [
            DiagnosticWindow(stock, 40, 0.01, 0.0, 1.0, 1.0, 0.1, 0.2,
                             0.1, 0.1, (), "high-volatility")
            for stock in ("2330", "2454")
        ]
        portfolio = DiagnosticWindow("TW50_PORTFOLIO", 40, 0.06, 0.01, 1.0, 1.0,
                                     0.1, 0.2, 0.1, 0.1, (), "high-volatility")
        report = decide(stocks, stocks, self.policy,
                        portfolio_current=portfolio, portfolio_previous=portfolio)
        self.assertEqual(report.affected_stocks, ("2330", "2454"))


if __name__ == "__main__":
    unittest.main()