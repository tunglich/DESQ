# Appendix F monitoring contract

Authority: `Paper2_IEEEAccess_appendixD_added.pdf`, pages 24-28, SHA-256
`7b474ee437690126d5474c696faf15a25a1d586861a090abf3a123ee1fc4f91a`.

This is a post-deployment governance protocol. The paper explicitly states
that it was not activated for the reported empirical performance.

## Implemented decision rules

- Eq. (10): labels mature after the 20-trading-day forecast horizon.
- Eq. (21): precision-gap, return-gap, negative Sharpe/IR, DES-disagreement,
  Dynamic-Flooding upper-bound, and PSI alarms. PSI uses the paper threshold
  `0.25`; the paper leaves `delta_p`, `delta_r`, and `delta_b` unspecified, so
  they live in versioned operational policy.
- Eq. (22): an update is queued only when at least two alarms fire in each of
  two consecutive mature windows.
- Eqs. (23)-(31): Level 1 first evaluates threshold and DES-weight
  recalibration while specialist parameters remain fixed.
- Eq. (32): failed recalibration escalates persistent localized drift to Level
  2 and broad drift to Level 3.
- Eqs. (33)-(38): Level 2 updates affected stock/group specialists and refits
  DES; Level 3 expands to the complete stock/group universe.
- Eqs. (39)-(40): candidate configuration is limited to `Theta_allow`, while
  labels, split, transaction costs, holdout evaluator, benchmark, and backtest
  rule remain immutable.
- Eq. (41): promotion requires material validation-precision improvement and
  bounded Sharpe, turnover, and drawdown deterioration under the same sealed
  validation protocol.

Numerical values not supplied by the paper are labeled `operational_policy`
in `config/policy_v1.json`. DES weights are treated as derived recalibration
outputs of Eqs. (27)-(31), not an extra tunable member of Eq. (39).