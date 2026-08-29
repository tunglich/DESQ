# Revised-paper Tables 3-10

This bundle targets the authoritative 30-page PDF `Paper2_IEEEAccess_appendixD_added.pdf` with SHA-256:

```text
7b474ee437690126d5474c696faf15a25a1d586861a090abf3a123ee1fc4f91a
```

Generate and audit every table with:

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
| 6 | Cross-market statistics | Peer and benchmark rows reproduce from shipped NAV; DESQ rows are `reported_only` because shipped DESQ is legacy DES+CUSUM |
| 7 | Nine-seed/statistical reliability | `reported_only` except benchmark returns; seed returns, bootstrap draws, DM inputs, and SPA inputs are not shipped |
| 8 | Regime performance | Benchmark dates/days/returns reproduce; shipped strategy NAV is a legacy rule trader and differs from DDQN |
| 9 | 78-feature taxonomy | Reproduced from all 250 feature CSV headers; six Fundamental files contain an extra `CMDTY` field recorded in `table9_schema_drift.csv` |
| 10 | Top-50 flooding ablations | PDF arithmetic reproduces; raw DDQN/flooding NAV is not shipped. PDF ticker `2324.TT` conflicts with repo constituent `3231` |

`reported_only` means the artifact is a deterministic, machine-readable transcription of the paper, not an empirical rerun. These rows must not be promoted to `reproduced` until their raw predictions, NAV series, seeds, configuration, and checkpoint hashes are present.

## Validation conventions

- Table 6 uses 562 common-date observations and an annual 3M T-bill rate of 4.42%. Core NAV metrics reproduce within 0.02. Several reported Sortino values do not follow the identifiable shipped-NAV formula and remain visible as failed audit rows.
- Table 8 treats each correction as peak-to-trough return intervals: 16/31 NAV points correspond to 15/30 daily return intervals. Pooled up-trend days are the remaining 495 of 540 observations.
- Table 10 transaction cost is checked as `round_trips * 0.585` percentage points with half-up rounding; all return/excess identities are checked to 0.011 percentage points.
