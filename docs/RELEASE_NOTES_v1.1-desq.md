# v1.1-desq - Revised-paper implementation

Release date: 2026-08-29

This release aligns the active repository with the authoritative 30-page
revised IEEE Access paper, `Paper2_IEEEAccess_appendixD_added.pdf` (SHA-256
`7b474ee437690126d5474c696faf15a25a1d586861a090abf3a123ee1fc4f91a`).

## Highlights

- Adds Signal-Conditioned Double DQN with prioritized replay and paper-defined
  learning defaults.
- Encodes five rolling validation windows and the 50-anchor effective gap:
  20-day label horizon plus a separate 30-trading-day purge.
- Applies Taiwan costs of 0.1425% on buys and 0.4425% on sells.
- Adds canonical Tables 3-10, generated outputs, validation reports, and
  explicit `reported_only` versus reproducible evidence status.
- Adds Appendix F monitoring governance with immutable snapshots,
  two-alarm/two-window triggering, Level 0-3 dry-run plans, and promotion
  guards. Monitoring was not active in the paper experiments.
- Makes table and US figure generation portable from a clean clone and runs
  monitoring/table regression tests on every push.

The historical `v1.0-desq` release remains unchanged. Shipped DES/CUSUM and
signal-pattern backtests remain available only as clearly labeled legacy
diagnostics; they are not relabeled as DDQN evidence.