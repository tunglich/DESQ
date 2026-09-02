"""Translate a monitoring decision into a non-executing candidate plan."""
from __future__ import annotations

from .decision import DecisionReport
from .schemas import CandidatePlan, CandidateStep, content_hash


def build_plan(report: DecisionReport, policy: dict) -> CandidatePlan:
    allowed = set(policy["repository_update_extension"]["theta_allow"])
    stocks = report.affected_stocks
    aspects = report.affected_groups
    if report.level == 0:
        steps: tuple[CandidateStep, ...] = ()
    elif report.level == 1:
        steps = (CandidateStep(
            "sealed_des_recalibration", (3, 4), stocks, aspects,
            ("des_threshold", "des_competence_window", "temperature_scaling"),
        ),)
    elif report.level == 2:
        steps = (
            CandidateStep("local_specialist_fine_tuning", (1, 2), stocks, aspects,
                          ("fine_tuning_learning_rate", "fine_tuning_epochs",
                           "dynamic_flooding_bounds", "freeze_policy")),
            CandidateStep("associated_des_refit", (3, 4), stocks, aspects,
                          ("des_threshold", "des_competence_window", "temperature_scaling")),
        )
    elif report.level == 3:
        steps = (
            CandidateStep("full_walk_forward_automl_rebuild", (1, 2), stocks, aspects,
                          ("automl_ranges", "dynamic_flooding_bounds")),
            CandidateStep("portfolio_wide_des_refit", (3, 4), stocks, aspects,
                          ("des_threshold", "des_competence_window", "temperature_scaling")),
        )
    else:
        raise ValueError(f"unsupported decision level: {report.level}")
    changed = {parameter for step in steps for parameter in step.changed_parameters}
    if not changed <= allowed:
        raise ValueError(f"candidate changes parameters outside repository policy: {sorted(changed - allowed)}")
    return CandidatePlan(report.level, content_hash(report.payload()), steps)