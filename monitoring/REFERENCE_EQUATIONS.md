# Monitoring equation contract

This post-deployment governance protocol was not activated for the recorded
empirical performance.

## Implemented decision rules

- Eq. (10): 20-day matured direction label.
- The lookback is `L=60`; because labels through `t-h` are usable and `h=20`,
  each mature window contains `L-h=40` anchors.
- Eqs. (11)-(12): rolling precision and probability/forward-return IC.
- Eq. (13): annualized rolling Sharpe and information ratio.
- Eqs. (14)-(16): precision degradation, DES-implied net return, and live
  return shortfall.
- Eq. (17): capital-weighted portfolio and benchmark returns.
- Eqs. (18)-(20): DES disagreement, Dynamic-Flooding upper-bound frequency,
  and feature-group PSI.
- Eq. (21) is the paper's illustrative five-alarm case. The formal operational
  trigger in Eqs. (22)-(23) uses precision gap, return gap, combined negative
  Sharpe/IR risk, DES disagreement, Flooding saturation, and PSI drift. At
  least two must fire at monitoring anchors `t` and `t-L`.
- Eqs. (24)-(27): threshold grid, cost-aware objective, maximization, and
  materiality/risk gate.
- Eqs. (28)-(32): convex DES aggregation, specialist competence,
  temperature-softmax weights, incumbent shrinkage, and diversity floor.
- Eqs. (33)-(39): threshold-first/weight-second Level 1, mature-only localized
  Level 2 with sealed DES refit, and full-universe Level 3 rebuild.
- Eqs. (40)-(42): allowed/forbidden parameter sets and strict sealed-validation
  promotion gate.
- Eq. (43): immutable research memory containing window, alarms, regime,
  candidate, promotion result, and diagnostic changes.
- A capital-weighted portfolio alarm may batch stocks that share an affected
  feature group even when no stock triggers alone. A shared regime signature
  is also allowed through the optional `regime_signature` input field.

## Repository update extension

The Level 0-3 mechanism is specified by Appendix F, but this repository emits
non-executable plans only. Candidate parameters are constrained by
`theta_allow`; labels, splits, costs, evaluator, benchmark, and backtest rules
remain immutable. Coefficients and tolerances not numerically fixed by the
paper remain repository operational policy.

Additional numerical values and procedures are labeled `operational_policy`
in `config/policy_v1.json`.

See `monitoring/README.md` for executable commands and input schemas.
