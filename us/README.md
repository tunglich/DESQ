# DESQ US evaluation: Dow 30 / S&P 100 / NASDAQ 100

The same four-stage architecture is applied to three U.S.
universes: Attention specialists, Dynamic Flooding, probability-valued
KNORA-E aggregation, and Signal-Conditioned Double DQN execution.

## Evaluation results

The reference bundle records the sealed 2024-01-02 to 2026-03-31 evaluation:

| Universe | DESQ DDQN return | Benchmark return | Excess return | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dow 30 | **67.40%** | 19.89% | 47.51 pp | 1.80 | -10.08% |
| S&P 100 | **82.80%** | 38.99% | 43.81 pp | 2.13 | -10.40% |
| NASDAQ 100 | **83.50%** | 38.74% | 44.76 pp | 1.92 | -11.60% |

The complete peer-method comparison, including annualized return, volatility,
Sortino, Calmar, DSR, DRL Ensemble, and MACE rows, is available in:

- [Table 6 CSV](../evaluation/paper/tables/table6_cross_market.csv)
- [Table 6 Markdown](../evaluation/paper/tables/table6_cross_market.md)
- [Table 6 LaTeX](../evaluation/paper/tables/table6_cross_market.tex)

The DESQ rows are deterministic transcriptions of the reference results and
remain `reported_only`: matching DDQN checkpoints, selected nine-seed manifests,
action paths, and NAV series are not shipped. Peer and benchmark rows retain
their independent shipped-NAV audits.

## Architecture

Each stock uses four U.S. feature groups (`fundamental`, `moment`,
`tech_trend`, and `macro`). KNORA-E produces the probability-valued DES signal;
the execution layer consumes 10 DES signals, 10 OHLC bars, position state, and
running P&L, then chooses `Skip`, `Buy`, or `Close`.

The executable DDQN implementation and nine-agent median-selection protocol
are shared with the Taiwan workflow under [dqn/](../dqn/). The complete
seven-stage supervised training flow is shown in the root
[training-flow diagram](../docs/training_pipeline.png).

## Rebuild evaluation tables

```bash
python evaluation/paper/generate_tables.py
```

This regenerates the Table 6 CSV, Markdown, LaTeX, validation report, and
manifest. It does not infer a DESQ return path from reported endpoints.

## Data windows

| Split | Range |
| --- | --- |
| Training | up to 2023-12-31 |
| Test / trade | 2024-01-02 → 2026-03-31 |
| Validation | rolling last 20 % of train (5 folds) |

## Feature aspects (4)

`fundamental`, `moment`, `tech_trend`, `macro`
(no `trade` aspect; US trade data via Alpha Vantage TIME_SERIES_DAILY_ADJUSTED).

## License

Inherits the repo-level `LICENSE`.
