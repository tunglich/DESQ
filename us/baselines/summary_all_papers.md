# Portfolio-Backtest Baselines — Full 4×3 Paper Reproduction Matrix

Trade window **2024-01-02 → 2026-03-30**, initial capital **$1,000,000**,
three universes (Dow 30 / S&P 100 / NASDAQ 100), four methods each.

**Main artefacts**

- Combined 1×3 comparison chart:
  [baselines/combined/four_methods_1x3.png](baselines/combined/four_methods_1x3.png)
- Per-universe standalone charts:
  [dow30](baselines/combined/dow30_comparison.png) ·
  [sp100](baselines/combined/sp100_comparison.png) ·
  [ndx100](baselines/combined/ndx100_comparison.png)
- Full backtest report (executive summary + analysis):
  [baselines/backtest_report.md](baselines/backtest_report.md)
- Extended statistics table:
  [baselines/combined/combined_stats.csv](baselines/combined/combined_stats.csv) ·
  [baselines/combined/combined_stats.md](baselines/combined/combined_stats.md)
- Cumulative-return data:
  [dow30_comparison.csv](baselines/combined/dow30_comparison.csv) ·
  [sp100_comparison.csv](baselines/combined/sp100_comparison.csv) ·
  [ndx100_comparison.csv](baselines/combined/ndx100_comparison.csv)

---

## Naming convention

Our method is displayed as **DTE** (*Dynamic-flooding Transformer
Ensembles*) in the combined charts, extended stats, and backtest
report. The underlying driver, source files, and per-ticker prediction
CSVs are all still named **DES** on disk (unchanged from earlier work).

## Transaction cost applied to each method

| Method | Buy | Sell | Roundtrip |
|---|---:|---:|---:|
| **DTE (ours)**                             | 0.10 % | 0.34 % | **0.44 %** |
| Dynamic Stock Recommendation (Yang 2018)   | 0.05 % | 0.05 % | 0.10 % |
| DRL Ensemble (Yang 2020)                   | 0.05 % | 0.05 % | 0.10 % |
| MACE-baseline (Abbade 2026)                | 0.05 % | 0.05 % | 0.10 % |
| MACE-AC (Abbade 2026, paper model)         | Almgren-Chriss (α=0.5, β=1.0, ε=5 bps, τ½=5 d) |

DTE carries the highest transaction cost of the four methods. The three
peer baselines all share 0.10 % roundtrip so their comparison is
like-for-like.

## Snapshot — total return by universe (%)

| Method \\ Universe | Dow 30 | S&P 100 | NASDAQ 100 |
|---|---:|---:|---:|
| **DTE (ours)**                | **+68.62** | **+89.50** | **+92.06** |
| Dynamic Stock Recommendation  | +11.78     | +64.45     | +79.45     |
| DRL Ensemble                  | +28.57     | +25.82     | +51.60     |
| MACE                          | +65.53     | +34.25     | +40.96     |
| Benchmark index               | +19.89 (^DJI) | +38.99 (^OEX) | +38.74 (^NDX) |

Sharpe, Sortino, MaxDD, Calmar, annualised return / volatility, and
excess return vs index are in
[combined_stats.md](baselines/combined/combined_stats.md).

---

## Per-paper details

### Yang et al. 2020 — DRL Ensemble (63-day rolling Sharpe)
*Paper*: arXiv 2511.12120 (Yang, Liu, Zhong, Walid; ICAIF 2020).
*Method*: 5 SB3 agents (A2C/PPO/DDPG/TD3/SAC) trained on 2015-2023,
per-quarter rolling-Sharpe (63d) agent selection. First 63 days =
equal-weight warm-up.
*Adaptation*: agents are frozen (trained once); paper retrains each quarter.

Aggregator: [FinRL/ensemble_aggregator.py](FinRL/ensemble_aggregator.py) —
`python ensemble_aggregator.py {dow30|sp100|ndx100}`.
Per-quarter chosen agent in
`FinRL/backtest_{universe}_*_2024_20260331/ensemble_selection.csv`.

- Dow 30: [FinRL/backtest_dow30_2024_20260331/](FinRL/backtest_dow30_2024_20260331/)
- S&P 100: [FinRL/backtest_sp100_variantA_2024_20260331/](FinRL/backtest_sp100_variantA_2024_20260331/)
- NDX 100: [FinRL/backtest_ndx100_variantA_2024_20260331/](FinRL/backtest_ndx100_variantA_2024_20260331/)

### Yang, Liu, Wu 2018 IEEE — Dynamic Stock Recommendation (DSR)
*Paper*: arXiv 2511.12129 / IEEE TrustCom/BigDataSE 2018.
*Method*: quarterly panel of fundamental indicators × 5 ML regressors
(Linear / Ridge / Lasso / RandomForest / GBM); 16Q rolling training + 4Q
validation → pick lowest-MSE model → hold **top 20 %** with EW / MVO /
MinVar allocation and quarterly rebalance.
*Adaptation*: 18 features (14 in-house fundamentals + 4 AV-computed TTM
ratios ROA/ROE/NPM/DE) instead of the paper's 20 Compustat
X-indicators. Combined chart reports the best of EW/MVO/MinVar per
universe by final return.

Driver: [baselines/dsr_yang/dsr_backtest.py](baselines/dsr_yang/dsr_backtest.py) —
`python dsr_backtest.py {dow30|sp100|ndx100}`.

- Dow 30 (30 tickers, top-6 selected): [baselines/dsr_yang/backtest_dow30_2024_20260330/](baselines/dsr_yang/backtest_dow30_2024_20260330/)
- S&P 100 (99 tickers, top-20 selected): [baselines/dsr_yang/backtest_sp100_2024_20260330/](baselines/dsr_yang/backtest_sp100_2024_20260330/)
- NDX 100 (95 tickers kept, top-19 selected): [baselines/dsr_yang/backtest_ndx100_2024_20260330/](baselines/dsr_yang/backtest_ndx100_2024_20260330/)

### Abbade & Costa 2026 — MACE (Almgren-Chriss Market Impact)
*Paper*: arXiv 2603.29086v2 (FinRL-Meta market-impact env).
*Method*: same 5 SB3 agents, re-backtested under two cost models:
- **Baseline**: flat 5 bps buy + 5 bps sell
- **AC (paper)**: Almgren-Chriss non-linear impact with permanent-impact
  exponential decay:
  $C_\text{perm} = \tfrac12\alpha\sigma(|x|/V)|x|P$,
  $C_\text{spread} = \varepsilon|x|P$,
  $C_\text{temp}  = \beta\sigma(|x|/V)|x|P$
  with $\alpha=0.5$, $\beta=1.0$, $\varepsilon=5$ bps, $\tau_{1/2}=5$ days.

Env: [baselines/mi_abbade/env_mace.py](baselines/mi_abbade/env_mace.py) —
Driver: [baselines/mi_abbade/mace_backtest.py](baselines/mi_abbade/mace_backtest.py) —
`python mace_backtest.py {dow30|sp100|ndx100}`.

The combined chart shows the **best-AC agent** per universe (A2C for
Dow 30, TD3 for S&P 100, PPO for Nasdaq 100).

**Cost-model impact — trades reduced by AC (baseline → AC)**

| Universe | Best AC agent | Baseline trades | AC trades | Reduction |
|---|---|---:|---:|---:|
| Dow 30    | A2C  | 7,266  | 1,403 | 5.2×  |
| S&P 100   | TD3  | 33,160 |    52 | 638×  |
| NDX 100   | PPO  | 20,574 | 1,907 | 10.8× |

- Dow 30: [baselines/mi_abbade/backtest_dow30_2024_20260330/](baselines/mi_abbade/backtest_dow30_2024_20260330/)
- S&P 100: [baselines/mi_abbade/backtest_sp100_2024_20260330/](baselines/mi_abbade/backtest_sp100_2024_20260330/)
- NDX 100: [baselines/mi_abbade/backtest_2024_20260330/](baselines/mi_abbade/backtest_2024_20260330/)

### DTE (ours) — Dynamic-flooding Transformer Ensembles
*Method*: each ticker trades on its own frozen DES probability signal +
CUSUM direction filter. Position sizes fixed at market-cap weight at t0.
Transaction cost = 0.10 % buy + 0.34 % sell (0.44 % roundtrip).

*Underlying signal*: DES probabilities in
`model_pred_DES_US/DES_pred_<TKR>_2019-12-31.csv` (one per ticker,
unchanged); CUSUM direction filter in `cumSum_prob_12/cusum_<TKR>.csv`.
Portfolio-level aggregation driver:
[Backtest_Portfolio_US.py](Backtest_Portfolio_US.py) (source-level name
remains DES; DTE is a display label only).

Skill spec: [.github/skills/backtesting-US/SKILL.md](.github/skills/backtesting-US/SKILL.md)

DTE equity CSVs (refreshed 2026-07-02 with the new 0.10 % / 0.34 % fees):

| Universe | Final $ (full CSV, 2024-01-02 → 2026-03-31) | Return | Sharpe |
|---|---:|---:|---:|
| Dow 30  | $1,703,533 | +70.35% | 2.96 |
| S&P 100 | $1,908,878 | +90.89% | 3.81 |
| NDX 100 | $1,934,929 | +93.49% | 3.15 |

Slight discrepancy vs the 4×3 matrix in the report because the matrix
truncates at 2026-03-30 (last common trading day across all methods)
whereas the DES CSV runs through 2026-03-31.

- Dow 30: [backtest_portfolio_US/equity_dow30_market_2024-01-02_2026-03-31.csv](backtest_portfolio_US/equity_dow30_market_2024-01-02_2026-03-31.csv)
- S&P 100: [backtest_portfolio_US/equity_sp100_market_2024-01-02_2026-03-31.csv](backtest_portfolio_US/equity_sp100_market_2024-01-02_2026-03-31.csv)
- NDX 100: [backtest_portfolio_US/equity_ndx100_market_2024-01-02_2026-03-31.csv](backtest_portfolio_US/equity_ndx100_market_2024-01-02_2026-03-31.csv)

---

## Reproduction commands

```powershell
# 1. DTE (DES) refresh — all 3 universes
powershell -File d:\US_stock\_refresh_des_3univ.ps1

# 2. Yang 2018 DSR
conda run -n finlabUS python d:\US_stock\baselines\dsr_yang\dsr_backtest.py dow30
conda run -n finlabUS python d:\US_stock\baselines\dsr_yang\dsr_backtest.py sp100
conda run -n finlabUS python d:\US_stock\baselines\dsr_yang\dsr_backtest.py ndx100

# 3. Yang 2020 backtest + ensemble aggregator (needs SB3, DRL env)
conda run -n DRL python d:\US_stock\FinRL\dow30_2024_20260331_backtest.py
conda run -n DRL python d:\US_stock\FinRL\sp100_variantA_backtest.py
conda run -n DRL python d:\US_stock\FinRL\ndx100_variantA_backtest.py
conda run -n finlabUS python d:\US_stock\FinRL\ensemble_aggregator.py dow30
conda run -n finlabUS python d:\US_stock\FinRL\ensemble_aggregator.py sp100
conda run -n finlabUS python d:\US_stock\FinRL\ensemble_aggregator.py ndx100

# 4. Abbade 2026 MACE (needs SB3, DRL env)
conda run -n DRL python d:\US_stock\baselines\mi_abbade\mace_backtest.py dow30
conda run -n DRL python d:\US_stock\baselines\mi_abbade\mace_backtest.py sp100
conda run -n DRL python d:\US_stock\baselines\mi_abbade\mace_backtest.py ndx100

# 5. Combined 1x3 chart + stats CSV + stats MD
conda run -n finlabUS python d:\US_stock\baselines\combined\combined_comparison.py

# One-shot rerun of steps 2-5 with the current cost settings
powershell -File d:\US_stock\_rerun_new_costs.ps1
```

## Environments

- `finlabUS`: DTE refresh, Yang 2018, Yang 2020 aggregator, combined chart
  (numpy / pandas / sklearn / yfinance only)
- `DRL`: SB3 2.3.2 + FinRL — required for Yang 2020 SB3 backtest and
  Abbade 2026 MACE (agent inference)

## SB3 trained agent locations

| Universe | Directory | Notes |
|---|---|---|
| Dow 30     | `FinRL/trained_models/` and `FinRL/trained_models_train_to_2023/` | 30 tickers × 20k timesteps |
| S&P 100    | `FinRL/sp100_variantA_trained_models/`                            | 98 tickers × 20k timesteps (added 2026-07-02) |
| NASDAQ 100 | `FinRL/ndx100_variantA_trained_models/`                           | 85 tickers × 20k timesteps |
