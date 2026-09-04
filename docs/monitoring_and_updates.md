# Monitoring and controlled updates

The `monitoring` package implements the reference diagnostics and trigger.
It consumes existing pipeline outputs without modifying them, writes
content-addressed snapshots, applies the two-alarm/two-window trigger,
and emits non-executing candidate plans.

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

Candidate selection and Level 0-3 procedures are repository extensions, not
part of the fixed monitoring equations. A hypothetical localized Level-2
example uses a cost-aware promotion condition;
the example was not observed during holdout evaluation. The planner is
deliberately `dry_run=true` and `executable=false`. Execution stays disabled
until every Stage 1-4 command can write to an isolated candidate root and
approval, sealed validation, atomic promotion, and rollback are implemented.

The full JSON schema, equation API, and operational workflow are documented in
the [monitoring package README](../monitoring/README.md).

## Configuration boundary

- `monitoring/config/reference_contract.json` contains fixed reference invariants,
  allowed parameters, forbidden parameters, and fixed trigger rules.
- `monitoring/config/policy_v1.json` contains thresholds and update/promotion
  procedures that the fixed contract does not specify. These values are
  operational bootstrap policy.

Daily and weekly diagnostics may produce risk reports or watch lists. A model
update can be queued only after 20-day labels mature. With `L=60` and `h=20`,
each eligible window contains `L-h=40` mature anchors, and the two monitoring
anchor indices must differ by exactly 60 trading days.