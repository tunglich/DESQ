# Monitoring and controlled updates

The `monitoring` package implements the reference diagnostics and trigger.
It consumes existing pipeline outputs without modifying them, writes
content-addressed snapshots, applies the six-alarm/two-window trigger in
Appendix F, emits non-executing candidate plans, and records Eq. (43) research
memory.

This mechanism was not activated in the recorded experiments. It does not
change the evidence status or evaluation values.

## Current commands

```bash
python -m monitoring smoke
python -m monitoring show-config
python -m monitoring collect-stage2 --stock-id 2330 --as-of 2026-03-31
python -m monitoring evaluate --previous previous.json --current current.json
```

`smoke` exercises Levels 0-3 using synthetic windows. `collect-stage2` reads
the five current Dynamic-Flooding prediction CSVs, excludes the final 20
trading anchors whose labels are not mature at the requested date, and writes
an immutable snapshot under `artifacts/monitoring/snapshots/`.

The lifecycle first evaluates threshold recalibration, then DES-weight
recalibration. Both must fail before escalation to mature-only local specialist
fine-tuning plus sealed DES refit (Level 2), or a full walk-forward/AutoML
rebuild plus portfolio DES refit (Level 3). The mechanism was not observed
during holdout evaluation. The planner is
deliberately `dry_run=true` and `executable=false`. Execution stays disabled
until every Stage 1-4 command can write to an isolated candidate root and
approval, sealed validation, atomic promotion, and rollback are implemented.

The full JSON schema, equation API, and operational workflow are documented in
the [monitoring package README](../monitoring/README.md).

## Configuration boundary

- `monitoring/config/reference_contract.json` contains the revised-paper hash,
  fixed reference invariants, and trigger rules.
- `monitoring/config/policy_v1.json` contains thresholds and update/promotion
  procedures that the fixed contract does not specify. These values are
  operational bootstrap policy.

Daily and weekly diagnostics may produce risk reports or watch lists. A model
update can be queued only after 20-day labels mature. With `L=60` and `h=20`,
each eligible window contains `L-h=40` mature anchors, and the two monitoring
anchor indices must differ by exactly 60 trading days.

Every evaluation writes `evaluation.json` and `research_memory.json`. Missing
candidate-validation deltas such as drawdown or turnover remain `null` until a
sealed candidate evaluator supplies them.