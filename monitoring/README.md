# Self-improving monitoring protocol

This package implements the post-deployment monitoring contract in Appendix F,
Eqs. (10)-(20), of the authoritative 28-page paper. Monitoring was not active
during the reported experiments, so these outputs do not reproduce or alter
the paper's performance tables.

The implementation is fail-closed and read-only:

- metric functions implement Eqs. (10)-(20);
- two mature, non-overlapping windows are required for an update trigger;
- inputs must match the active paper-contract and operational-policy hashes;
- decisions and plans are content-addressed immutable JSON;
- every candidate plan has `dry_run=true` and `executable=false`.

## Paper contract

Monitoring uses a 60-trading-day lookback and a 20-day label horizon. Therefore
each paper window contains 40 matured anchors: `L - h = 60 - 20`. The paper's
explicit trigger contains five alarm types:

1. validation-to-live precision degradation;
2. rolling Sharpe below its prescribed lower limit;
3. rolling information ratio below its prescribed lower limit;
4. maximum feature-group PSI above its threshold;
5. DES disagreement above its training-period 90th percentile.

An update is queued only when at least two alarm types fire in both the current
and immediately preceding mature windows. Return shortfall and Dynamic-Flooding
upper-bound frequency remain recorded diagnostics, but they are not members of
the paper's explicit five-alarm trigger set.

Appendix F also gives a hypothetical localized-degradation example: Level 2
fine-tunes only affected stock-group specialists on matured observations,
refits DES on sealed validation data, and promotes only when a cost-aware
validation objective improves without violating turnover or drawdown limits.
The general Level 0-3 routing, numerical thresholds, candidate budget, and
additional promotion guards in this package are repository operational policy.

## Components

| Path | Responsibility |
| --- | --- |
| `metrics.py` | Pure implementations of Eqs. (10)-(20) |
| `adapters.py` | Read-only Stage-2 prediction adapter with label maturity filtering |
| `decision.py` | Five paper alarms, adjacent-window trigger, and policy routing |
| `protocol.py` | Validated diagnostic batches and immutable evaluation artifacts |
| `planner.py` | Non-executing Level 0-3 candidate plans |
| `config/paper_contract.json` | Paper-defined invariants and trigger structure |
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

An optional top-level `portfolio_window` accepts the same fields as an item in
`windows`. A portfolio alarm can batch stocks only when at least two stock
windows name a feature group also listed in the triggered portfolio window.

Evaluate the current and previous batches:

```bash
python -m monitoring evaluate \
  --previous diagnostics/2026-Q1.json \
  --current diagnostics/2026-Q2.json
```

Use `--recalibration-status failed` only after a separately sealed Level-1
candidate evaluation has failed. This allows repository policy to route local
drift to Level 2 or broad drift to Level 3. Evaluation output is written under
`artifacts/monitoring/evaluations/<evaluation_id>/evaluation.json`.

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