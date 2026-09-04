# Self-improving monitoring protocol

This package implements Appendix F of `paper_extention_momemtum_titles.pdf`
(SHA-256 `0bf0c4ed7caba508ef96cfc4dd8ddcd8a7f725d0ba7ab38291d3330ea663914e`).
Monitoring was not active during the reported experiments, so these outputs do
not reproduce or alter the recorded performance results.

The implementation is fail-closed and read-only:

- metric and recalibration functions implement Eqs. (10)-(20), (25)-(26), and
  (29)-(32);
- two mature, non-overlapping windows are required for an update trigger;
- inputs must match the active reference-contract and operational-policy hashes;
- evaluations, plans, and Eq. (43) research memory are content-addressed JSON;
- every candidate plan has `dry_run=true` and `executable=false`.

## Reference contract

Monitoring uses a 60-trading-day lookback and a 20-day label horizon. Therefore
each evaluation window contains 40 matured anchors: `L - h = 60 - 20`. The
formal trigger in Eqs. (22)-(23) contains six alarm types:

1. validation-to-live precision gap;
2. DES-implied versus realized return gap;
3. negative rolling Sharpe or information ratio (one combined risk alarm);
4. DES disagreement above its training-period 90th percentile;
5. Dynamic-Flooding upper-bound saturation;
6. maximum feature-group PSI above 0.25.

An update is queued only when at least two alarm types fire in both the current
and immediately preceding mature windows. Eq. (21)'s five-alarm expression is
the preceding illustrative case; Eqs. (22)-(23) govern the formal lifecycle.

Level 1 first evaluates threshold recalibration and then DES-weight
recalibration. Only if both fail can persistent local drift reach Level 2 or
broad drift reach Level 3. Level 2 fine-tunes affected stock-group specialists
using mature labels only and refits DES on sealed validation; Level 3 reruns the
original walk-forward/AutoML path and performs a portfolio-wide DES refit.
Unspecified coefficients, tolerances, and broad-drift cutoffs remain labeled
repository `operational_policy`.

## Components

| Path | Responsibility |
| --- | --- |
| `metrics.py` | Pure implementations of Eqs. (10)-(20) |
| `adapters.py` | Read-only Stage-2 prediction adapter with label maturity filtering |
| `decision.py` | Six Appendix F alarms, adjacent-window trigger, and Level 0-3 routing |
| `updates.py` | Pure threshold-objective and DES-weight recalibration equations |
| `protocol.py` | Validated diagnostic batches and immutable evaluation artifacts |
| `planner.py` | Non-executing Level 0-3 candidate plans |
| `config/reference_contract.json` | Reference invariants and trigger structure |
| `config/policy_v1.json` | Repository-defined thresholds and update controls |

No extra package is required beyond the repository environment.

## Quick start

Run the deterministic checks and print the active hashes:

```bash
python -m monitoring smoke
python -m monitoring show-config
```

Collect a read-only Stage-2 snapshot for one stock:

```bash
python -m monitoring collect-stage2 --stock-id 2330 --as-of 2026-03-31
```

This reads the five files under `artifacts/dflood/pred/`, excludes the final 20
anchors whose labels are not mature, and writes an immutable snapshot under
`artifacts/monitoring/snapshots/`.

## Evaluate two mature windows

Upstream monitoring jobs calculate the full diagnostic values with
`monitoring.metrics` and write one JSON file per mature window. Use the hashes
printed by `show-config` in both files:

```json
{
  "schema_version": "1.0",
  "monitoring_anchor_index": 1060,
  "observation_start": "2026-04-01",
  "observation_end": "2026-06-30",
  "paper_contract_hash": "<from show-config>",
  "policy_hash": "<from show-config>",
  "windows": [
    {
      "stock_id": "2330",
      "sample_count": 40,
      "precision_gap": 0.09,
      "return_gap": 0.01,
      "sharpe": 0.42,
      "information_ratio": 0.18,
      "disagreement": 0.30,
      "training_disagreement_q90": 0.20,
      "flooding_upper_fraction": 0.10,
      "max_psi": 0.30,
      "affected_groups": ["trade"]
    }
  ]
}
```

`monitoring_anchor_index` is the current monitoring date's zero-based position
in the upstream sealed trading calendar. The previous file must use index
`1000` in this example: evaluation requires an exact difference of `L = 60`,
not merely two files in chronological order. Each window must report exactly 40
matured anchors to be eligible. The evaluator verifies the declared indices;
the upstream collector remains responsible for binding indices to calendar
dates and preserving that calendar as a hashed source artifact.

Each window may also include a `regime_signature`. An optional top-level
`portfolio_window` accepts the same fields. A portfolio alarm can batch stocks
when at least two windows share an affected feature group or the triggered
portfolio regime signature.

Evaluate the current and previous batches:

```bash
python -m monitoring evaluate \
  --previous diagnostics/2026-Q1.json \
  --current diagnostics/2026-Q2.json
```

The default queues threshold recalibration. Re-evaluate with
`--recalibration-status threshold_failed` to queue DES-weight recalibration.
Use `weights_failed` only after both sealed Level-1 candidates fail; this routes
local drift to Level 2 or broad drift to Level 3. The corresponding
`threshold_promoted` and `weights_promoted` values record successful gates.
Each evaluation directory contains `evaluation.json` and
`research_memory.json`.

## Metric API

The functions in `monitoring.metrics` accept already aligned mature-window
arrays. For example:

```python
from monitoring.metrics import (
    annualized_information_ratio,
    annualized_sharpe,
    des_disagreement,
    forward_return,
    information_coefficient,
    population_stability_index,
    rolling_precision,
)

precision = rolling_precision(signals, matured_labels)
forward_returns = [forward_return(anchor, terminal) for anchor, terminal in prices]
ic = information_coefficient(des_probabilities, forward_returns)
sharpe = annualized_sharpe(net_strategy_returns)
information_ratio = annualized_information_ratio(
    net_strategy_returns, benchmark_returns
)
disagreement = des_disagreement(five_specialist_probabilities)
psi = population_stability_index(training_bin_shares, live_bin_shares)
```

Callers are responsible for preserving date alignment, using only labels known
by `t - 20`, and deriving PSI bins and disagreement baselines from sealed
training data. Invalid shapes and probability/share constraints raise errors.

## Safety boundary

The package never launches training, mutates an incumbent model, or promotes a
candidate. Before execution can be enabled, all stages must support isolated
candidate roots plus explicit approval, sealed validation, atomic promotion,
and rollback. Until then, an evaluation is evidence for review, not permission
to deploy.