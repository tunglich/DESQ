# Appendix F monitoring contract

Authority: `Paper2_Highlighted_PDF_consistency_fixed.pdf`, pages 23-26,
SHA-256 `a92e3ea2148d6bbe0d802976b5e6b46e7f879a270fe67c885215d977ec9dfd14`.

This is a post-deployment governance protocol. The paper explicitly states
that it was not activated for the reported empirical performance.

## Implemented decision rules

- Eqs. (10)-(20): define label maturity, diagnostic windows, precision and
  return gaps, risk, DES disagreement, Dynamic-Flooding saturation, feature
  drift, and the adjacent-window alarm trigger.
- An update is queued only when at least two alarms fire in each of two
  consecutive mature windows.

## Repository update extension

The paper no longer defines candidate selection, Level 0-3 update procedures,
or promotion guards. The dry-run planner retains these controls as a repository
extension: Level 1 recalibrates, Level 2 updates localized specialists, and
Level 3 retrains broadly. Candidate parameters are constrained by
`theta_allow`; labels, splits, costs, evaluator, benchmark, and backtest rules
remain immutable. Promotion requires improved validation precision without
breaching configured Sharpe, turnover, or drawdown limits.

Numerical values and procedures not supplied by the paper are labeled
`operational_policy` in `config/policy_v1.json`; they are not paper claims.