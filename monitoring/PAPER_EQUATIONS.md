# Appendix F monitoring contract

Authority: `Paper2_Highlighted_PDF_consistency_fixed.pdf`, pages 23-26,
SHA-256 `a92e3ea2148d6bbe0d802976b5e6b46e7f879a270fe67c885215d977ec9dfd14`.

This is a post-deployment governance protocol. The paper explicitly states
that it was not activated for the reported empirical performance.

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
  feature group even when no stock triggers alone. The paper also permits a
  shared regime signature, but the current input schema does not represent
  regime signatures and therefore does not route on them.

## Repository update extension

The paper's hypothetical case describes localized Level 2 fine-tuning, sealed
DES refitting, and promotion after a cost-aware validation-objective gain while
turnover and drawdown remain within limits. It is an illustration, not an
update observed during holdout evaluation.

The general Level 0-3 router and its numerical thresholds remain a repository
extension: Level 1 recalibrates, Level 2 updates localized specialists, and
Level 3 retrains broadly. Candidate parameters are constrained by `theta_allow`;
labels, splits, costs, evaluator, benchmark, and backtest rules remain
immutable. The repository also applies a Sharpe non-degradation guard.

Numerical values and procedures not supplied by the paper are labeled
`operational_policy` in `config/policy_v1.json`; they are not paper claims.

See `monitoring/README.md` for executable commands and input schemas.