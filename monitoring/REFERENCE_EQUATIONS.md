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
- The explicit trigger set is precision, Sharpe, information ratio, PSI drift,
  and DES disagreement. At least two must fire at monitoring anchors `t` and
  `t-L`. Return shortfall and Flooding saturation are diagnostics, not members
  of this five-alarm set.
- A capital-weighted portfolio alarm may batch stocks that share an affected
  feature group even when no stock triggers alone. A shared regime signature
  is also allowed, but the current input schema does not represent regime
  signatures and therefore does not route on them.

## Repository update extension

A hypothetical case describes localized Level 2 fine-tuning, sealed DES
refitting, and promotion after a cost-aware validation-objective gain while
turnover and drawdown remain within limits. It is an illustration, not an
update observed during holdout evaluation.

The general Level 0-3 router and its numerical thresholds remain a repository
extension: Level 1 recalibrates, Level 2 updates localized specialists, and
Level 3 retrains broadly. Candidate parameters are constrained by `theta_allow`;
labels, splits, costs, evaluator, benchmark, and backtest rules remain
immutable. The repository also applies a Sharpe non-degradation guard.

Additional numerical values and procedures are labeled `operational_policy`
in `config/policy_v1.json`.

See `monitoring/README.md` for executable commands and input schemas.
