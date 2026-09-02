# v1.2 paper revision ledger

This ledger compares the previous 30-page authority with the complete new
28-page authority. Yellow highlighting is a review aid, not a scope filter.

| Role | Document | Pages | SHA-256 |
| --- | --- | ---: | --- |
| Previous baseline | `Paper2_IEEEAccess_appendixD_added.pdf` | 30 | `7b474ee437690126d5474c696faf15a25a1d586861a090abf3a123ee1fc4f91a` |
| Current authority | `Paper2_Highlighted_PDF_consistency_fixed.pdf` | 28 | `a92e3ea2148d6bbe0d802976b5e6b46e7f879a270fe67c885215d977ec9dfd14` |

The PDFs are local review inputs and are not distributed in this repository.
`tools/paper_audit.py` records their hashes, normalized page text, flattened
yellow highlights, and page-aware text differences.

## Substantive changes

| New location | Previous claim | Current claim | Repository resolution |
| --- | --- | --- | --- |
| Appendix F, pp. 23-26, Eqs. (10)-(20) | Appendix F Eqs. (10)-(42) defined monitoring plus a full Level 0-3 update and promotion protocol. | Appendix F defines monitoring diagnostics and the mature-window alarm trigger. Candidate selection, recalibration, retraining, and promotion are maintained separately in the repository. | Paper contract covers Appendix F Eqs. (10)-(20). The existing non-executing Level 0-3 planner remains a clearly labeled repository extension whose thresholds live in operational policy. |
| Appendix G, pp. 26-27, Eqs. (21)-(29) | KNORA-E and DDQN details were interleaved with the monitoring equation sequence. | Appendix G separately defines KNORA-E competence and probability aggregation, the fixed 0.5 reference mapping, costs, and the Double-DQN target. | Existing `K=30`, state/action, costs, PER, and Double-DQN behavior agree; no architecture change or retraining is required. |
| Section IV.H and Table 7, p. 19 | Representative results were described as a median-return portfolio agent. | Each stock trains nine DDQN agents; the agent with the median validation return is selected per stock, then selected stock NAVs are aggregated. No separate portfolio agent is trained. | Add a deterministic Stage-4 median-agent selector. Existing Stage-3 seed sweep remains a variability diagnostic and is not DDQN selection evidence. |
| Table 4, p. 16 | Static Flooding return 104.4%; No Flooding return 89.9%. | Static Flooding return 111.1%; No Flooding return 96.6%. | Update reported source rows and regenerate all table formats and manifests. |
| Section IV.F, p. 16 | Passive Top-50 119.1%; DESQ excess +9.9 percentage points. | Passive Top-50 108.64%; DESQ excess +20.35 percentage points. | Canonical Appendix C Table C1 already contains 108.64% and +20.35 pp; align prose and references without inventing a NAV path. |
| Appendices A/C/E/F | Tables 9/10/11/12 and Figures 18/20 used sequential numbering. | These are Tables A1/C1/E1/F1 and Figure B1; the bootstrap illustration is Figure D1. | Preserve compatibility filenames where useful, but use current paper labels and page references in generated content and documentation. |
| Appendix B, Figure B1, p. 21 | The repository diagram extended the paper flow with an eighth DDQN box. | Figure B1 contains seven stages, a five-split/30-sample-gap walk-forward node, and ends with dynamic ensemble selection and back-test. | Regenerate `docs/training_pipeline.png` from the seven-stage renderer; document DDQN separately rather than inserting it into Figure B1. |

## Unchanged executable contract

- Five feature groups with 78 paper features.
- Five dated walk-forward folds, a 20-day label horizon, and a separate
  30-trading-day purge.
- Dynamic Flooding search/repeats/top-3 averaging and KNORA-E with `K=30`.
- Signal-conditioned Double DQN state/action architecture, PER,
  `gamma=0.99`, 5,000-step target synchronization, and Taiwan costs.
- Reported returns: TSMC 202.5%, MediaTek 101.2%, Top-50 129.0%, Dow 30
  67.4%, S&P 100 82.8%, and NASDAQ 100 83.5%.

These unchanged reported values remain `reported_only` where checkpoints,
action paths, seed manifests, or NAV series are not shipped.

## Editorial-only changes

The abstract removes two precision figures, revises scalability wording, and
uses “Top-50 portfolio” rather than “Top-50 agent.” References are corrected
and the KNORA-E source is added. Author and affiliation metadata are unchanged.
These edits do not alter code.

## Known paper issue

Section IV.H refers to “Figure 20,” while the corresponding current caption is
“Figure D1.” Repository documentation uses Figure D1 and records this mismatch
rather than silently treating it as a second figure.
