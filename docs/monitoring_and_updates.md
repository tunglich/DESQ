# Monitoring and controlled updates

The `monitoring` package implements the diagnostics and trigger in Appendix F,
Eqs. (10)-(20).
It consumes existing pipeline outputs without modifying them, writes
content-addressed snapshots, applies the paper's two-alarm/two-window trigger,
and emits non-executing candidate plans.

This mechanism was not activated in the paper's reported experiments. It does
not change the evidence status or values of Tables 3-10.

## Current commands

```bash
python -m monitoring smoke
python -m monitoring collect-stage2 --stock-id 2330 --as-of 2026-03-31
```

`smoke` exercises Levels 0-3 using synthetic windows. `collect-stage2` reads
the five current Dynamic-Flooding prediction CSVs, excludes the final 20
trading anchors whose labels are not mature at the requested date, and writes
an immutable snapshot under `artifacts/monitoring/snapshots/`.

Candidate selection and Level 0-3 procedures are repository extensions, not
claims about the current paper. The planner is deliberately `dry_run=true` and
`executable=false`. Execution stays disabled until every Stage 1-4 command can
write to an isolated candidate root and approval, sealed validation, atomic
promotion, and rollback are implemented.

## Configuration boundary

- `monitoring/config/paper_contract.json` contains paper-defined invariants,
  allowed parameters, forbidden parameters, and fixed trigger rules.
- `monitoring/config/policy_v1.json` contains thresholds and update/promotion
  procedures that Appendix F does not specify. These values are operational
  bootstrap policy, not paper claims.

Daily and weekly diagnostics may produce risk reports or watch lists. A model
update can be queued only after 20-day labels mature and at least 60 mature
anchors are available in each of two consecutive windows.