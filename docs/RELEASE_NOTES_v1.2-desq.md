# v1.2-desq - Current-paper alignment

Release-candidate date: 2026-09-03

This release aligns the active repository with the complete 28-page
`Paper2_Highlighted_PDF_consistency_fixed.pdf` authority (SHA-256
`a92e3ea2148d6bbe0d802976b5e6b46e7f879a270fe67c885215d977ec9dfd14`).
The PDF is an audit input and is not included in the release.

## Changed

- Adds a deterministic PDF audit utility for whole-document and flattened
  yellow-highlight comparison.
- Implements reproducible Stage-4 seeded training, isolated seed directories,
  candidate manifests, and fifth-of-nine validation-return selection per stock.
- Updates Table 4 Static Flooding return from 104.4% to 111.1% and No Flooding
  return from 89.9% to 96.6%, preserving `reported_only` provenance.
- Uses current Appendix A Table A1 and Appendix C Table C1 labels while
  preserving `table9_*` and `table10_*` compatibility filenames.
- Regenerates Appendix B Figure B1 as the current seven-stage training and
  back-test flow, including the five-split/30-sample-gap walk-forward node.
- Realigns monitoring to Appendix F Eqs. (10)-(20) and Appendix G Eqs.
  (21)-(29). Level 0-3 candidate planning and promotion controls are explicitly
  repository-defined operational policy.
- Updates README, citation, Zenodo, monitoring, and paper-artifact documentation.

## Evidence boundary

The repository still does not ship the nine-seed DDQN checkpoints, action
paths, or NAV series used for reported returns. Headline DDQN results therefore
remain `reported_only`. Existing Taiwan rule-trader and U.S. DES/CUSUM curves
remain legacy diagnostics; this release does not synthesize replacement return
paths from paper endpoints.

## Compatibility

The ATT/Flooding -> Dynamic Flooding -> KNORA-E -> Double DQN architecture and
historical `v1.0-desq` and `v1.1-desq` releases are unchanged.
