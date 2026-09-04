# Evaluation tables and audits

Generate and audit every reference table with:

```bash
python evaluation/paper/generate_tables.py
```

Outputs are written to `tables/` as CSV, Markdown, and LaTeX. Validation reports are written to `validation/`, and `tables_manifest.json` records source hashes, row counts, and evidence status counts.

## Evidence status

| Table | Subject | Current evidence |
| ---: | --- | --- |
| 3 | Walk-forward precision | `reported_only`: corresponding fold predictions are not shipped |
| 4 | Cumulative module ablation | `reported_only`: six DDQN experiment bundles are not shipped |
| 5 | Horizon sensitivity | `reported_only`: horizon-specific retraining bundles are not shipped |
| 6 | Cross-market statistics | Peer and benchmark rows reproduce from their shipped NAV; DESQ rows are `reported_only` because matching DDQN NAV is not shipped |
| 7 | Nine-seed/statistical reliability | `reported_only` except benchmark returns; seed returns, bootstrap draws, DM inputs, and SPA inputs are not shipped |
| 8 | Regime performance | The regime set, day partition, and excess-return arithmetic are audited; matching DDQN NAV is not shipped |
| A1 (compatibility filename: 9) | 78-feature taxonomy | Reproduced from all 250 feature CSV headers; six Fundamental files contain an extra `CMDTY` field recorded in `table9_schema_drift.csv` |
| C1 (compatibility filename: 10) | Top-50 flooding ablations | Reference arithmetic reproduces; raw DDQN/flooding NAV is not shipped. Reference ticker `2324.TT` conflicts with repo constituent `3231` |

`reported_only` means the artifact is a deterministic, machine-readable reference transcription, not an empirical rerun. These rows must not be promoted to `reproduced` until their raw predictions, NAV series, seeds, configuration, and checkpoint hashes are present.

## Validation conventions

- Table 6 uses 562 common-date observations and an annual 3M T-bill rate of 4.42%. Core NAV metrics reproduce within 0.02. Several reported Sortino values do not follow the identifiable shipped-NAV formula and remain visible as failed audit rows.
- Table 8 treats each correction as peak-to-trough return intervals: 16/31 NAV points correspond to 15/30 daily return intervals. Pooled up-trend days are the remaining 495 of 540 observations.
- Appendix C Table C1 transaction cost is checked as `round_trips * 0.585` percentage points with half-up rounding; all return/excess identities are checked to 0.011 percentage points.
