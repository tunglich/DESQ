# TW-50 Attention + Flooding + DES Pipeline

[![CI](https://github.com/tunglich/DESQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tunglich/DESQ/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/pending.svg)](https://doi.org/pending) <!-- replace with real DOI after Zenodo mints it -->

> **Paper snapshot**: this repository accompanies the IEEE Access submission *"Dynamic-Flooding Transformer Ensembles for Reinforcement-Learning-Based Equity Market Timing"*. The exact code state used in the paper is tagged as [`v1.0-desq`](https://github.com/tunglich/DESQ/releases/tag/v1.0-desq); that tag is archived on Zenodo with a persistent DOI (see [How to cite](#how-to-cite)). The `main` branch may contain post-submission changes; check out the tag for byte-exact reproduction.

A market-timing framework for the TWSE Top-50 constituents that stacks:

1. **Attention (ATT) sequence classifier** with static Flooding regularization (Bayesian hyperparameter search).
2. **Dynamic Flooding** retraining with the best `flooding_b` per aspect.
3. **Dynamic Ensemble Selection (KNORA-E, K=30)** across the 5 per-aspect ATT predictions.
4. **Signal-Conditioned Double DQN** using the DES signal, OHLC history, position state, and running P&L to choose `Skip / Buy / Close`.

The pipeline uses the revised paper's five **walk-forward rolling validation**
windows (4:1 train:validation ratio) with a 20-day label horizon and a further
30-trading-day purge on five feature aspects.

## End-to-end training pipeline

The eight stages below are the complete per-stock workflow, annotated with the exact script in this repository that implements each stage. The `next aspect` loop repeats stages 2–5 for each of the five feature aspects; after all five aspects are trained, stages 6–8 run once.

![End-to-end training pipeline (7 stages, with code annotations)](docs/training_pipeline.png)

Regenerate the figure with `python docs/_render_training_pipeline.py`.

## Experimental results

The revised paper reports the following sealed-holdout results. These are the
canonical headline values for this branch; their current evidence status is
`reported_only` because the corresponding DDQN checkpoints and action paths
are not shipped. See [evaluation/paper/README.md](evaluation/paper/README.md).

| Revised-paper result | DESQ DDQN return | Benchmark return | Canonical source |
| --- | ---: | ---: | --- |
| TSMC (2330.TT) | **+202.50%** | +201.82% | [evaluation/paper/tables/table10_top50_flooding.csv](evaluation/paper/tables/table10_top50_flooding.csv) |
| MediaTek (2454.TT) | **+101.20%** | +62.69% | [evaluation/paper/tables/table10_top50_flooding.csv](evaluation/paper/tables/table10_top50_flooding.csv) |
| TWSE Top-50 portfolio | **+129.0%** | +88.07% | [evaluation/paper/tables/table8_regime.csv](evaluation/paper/tables/table8_regime.csv) |

### Legacy diagnostic

The shipped chart below evaluates KNORA-E with the earlier signal-pattern
trader over 2024-01-02 to 2026-03-31. It is retained only for audit comparison
and does not override the revised-paper values above.

![Out-of-sample back-tests](evaluation/figure_backtest_overview.png)

| Panel | Legacy rule-trader return | Buy-and-hold cumulative return | Source CSV |
| --- | ---: | ---: | --- |
| TSMC (2330.TT) | **+202.53 %** | +201.82 % | [evaluation/backtest_2330.csv](evaluation/backtest_2330.csv) |
| MediaTek (2454.TT) | **+103.42 %** | +62.69 % | [evaluation/backtest_2454.csv](evaluation/backtest_2454.csv) |
| TW-50 Model Portfolio (vs TWA02) | **+131.37 %** | +88.07 % | [evaluation/backtest_portfolio_tw50.csv](evaluation/backtest_portfolio_tw50.csv) |

Regenerate the figure directly from the shipped CSVs with:

```bash
python evaluation/render_figure_backtest.py
```

The CSV schemas are:

- Single-stock panels (`backtest_2330.csv`, `backtest_2454.csv`): `Date, Model_Return_Pct, Stock_Return_Pct, Model_Return_Ratio, Stock_Return_Ratio`.
- Portfolio panel (`backtest_portfolio_tw50.csv`): `Date, Model_CumRet, Benchmark_CumRet, Model_CumRet_Pct, Benchmark_CumRet_Pct`.

### US extension — four-method paper reproduction

The shipped US comparison is also **legacy DES+CUSUM evidence**. It has not yet been regenerated through the revised paper's Double-DQN execution layer and is retained as a diagnostic comparison only. See the [US extension README](us/README.md) and [alignment ledger](docs/paper_alignment.md).

![Four methods across three US universes](us/baselines/combined/four_methods_1x3.png)

| Universe | Legacy DES+CUSUM | DSR — Yang 2018 | DRL Ensemble — Yang 2020 | MACE — Abbade 2026 | Benchmark index |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dow 30     | **+68.6 %** | +11.8 % | +28.6 % | +65.5 % | +19.9 % (^DJI) |
| S&P 100    | **+89.5 %** | +64.5 % | +25.8 % | +34.3 % | +39.0 % (^OEX) |
| NASDAQ 100 | **+92.1 %** | +79.5 % | +51.6 % | +41.0 % | +38.7 % (^NDX) |

Per-day cumulative-return CSVs (columns: `date, DESQ, DRL Ensemble, Dynamic Stock Recommendation, MACE, <benchmark>`):

- [us/baselines/combined/dow30_comparison.csv](us/baselines/combined/dow30_comparison.csv)
- [us/baselines/combined/sp100_comparison.csv](us/baselines/combined/sp100_comparison.csv)
- [us/baselines/combined/ndx100_comparison.csv](us/baselines/combined/ndx100_comparison.csv)

Full stats (Sharpe / Sortino / MaxDD / Calmar): [us/baselines/combined/combined_stats.md](us/baselines/combined/combined_stats.md) · [combined_stats.csv](us/baselines/combined/combined_stats.csv). Regenerate the figure with `python us/baselines/combined/combined_comparison.py`.

### Training time (single stock)

Measured on RTX 5090 and RTX 5080 (WSL2 Ubuntu-24.04, TF 2.21, `finlab` conda env). Wall-clock time is essentially the same on both cards for this workload.

| Stage | Script | Time per stock |
| --- | --- | ---: |
| Phase 1 — hyperparameter search (Bayesian) | `ATT+Flood.py` | ~3 h |
| Phase 2 — Dynamic Flooding retraining (18 repeats, top-3 kept) | `ATT+Dflooding.py` | ~2 h |
| DES ensemble (RF + KNORA-E + CUSUM + backtest) | `DES_update_ATT-sentiment.py` | ~5 min |
| **Total per stock** (6 aspects, ATT only) | Batch_training agent | **~5 h** |

Reference guides for the internal end-to-end training pipeline that produced the above experimental results (uses the parent workspace's ATT scripts, not the compact `tw50_flood.py` / `tw50_dflood.py` / `tw50_des.py` demos in this repo):

- [docs/att_batch_training/README_Batch_training.md](docs/att_batch_training/README_Batch_training.md) — Batch_training agent (RTX 5090)
- [docs/att_batch_training/README_Batch_training_5080.md](docs/att_batch_training/README_Batch_training_5080.md) — Batch_training agent (RTX 5080)
- [docs/att_batch_training/README_DES.md](docs/att_batch_training/README_DES.md) — DES ensemble execution guide

## Data window

| Split                | Range                        |
| -------------------- | ---------------------------- |
| ATT training         | 2010-01-01 ~ **2023-12-31**  |
| DES (KNORA-E) train  | 2020-01-01 ~ **2023-12-31**  |
| Test (held-out)      | 2024-01-01 ~ **2026-03-31**  |
| Validation           | last 20% of the train window (rolling, 5 folds) |

## Feature aspects (5)

The pipeline uses five attribute-grouped feature aspects. The names in the accompanying IEEE Access paper differ slightly from the identifiers used in the code and on disk; the table below is the canonical mapping.

| Paper name    | Code identifier | On-disk file pattern           | Description                                                        |
| ------------- | --------------- | ------------------------------ | ------------------------------------------------------------------ |
| Fundamental   | `fundamental`   | `features/fundamental_<id>.csv` | Valuation ratios, revenue / EPS / margin growth at MoM/QoQ/YoY.    |
| Float         | `trade`         | `features/trade_<id>.csv`       | Chip-flow / smart-money: foreign & institutional holdings, margin, short-balance, net-buy. |
| Price-Trend   | `tech_trend`    | `features/tech_trend_<id>.csv`  | Trend-following technicals: SMA/HullMA deviations, Aroon, MACD, Bollinger, OHLCV. |
| Momentum      | `moment`        | `features/moment_<id>.csv`      | Oscillator technicals: RSI, KD, Williams %R, CCI, ADX, acceleration, rolling beta. |
| Macro         | `macro`         | `features/macro_<id>.csv`       | Rates, commodities, global equity indices, VIX, FX, TAIEX futures/options. |

Each CSV follows the layout `<aspect>_<stock_id>.csv`, with the last 4 columns being labels `y_10, y_20, y_40, y_60`.

## Universe

TWSE Top-50 constituents by market cap on 2023-12-29 — see [tw50_top50.csv](tw50_top50.csv).

## Pipeline

```
                    ┌─────────────────┐
features (5 aspects)│  tw50_flood.py  │  Bayesian tuning over ATT hyperparams
                    │   (Stage 1)     │  + static Flooding grid b in {0.00..0.40}
                    └────────┬────────┘  Saves best trial per (stock, aspect)
                             ▼
                    ┌─────────────────┐
                    │ tw50_dflood.py  │  Fixed-HP retraining + Dynamic Flooding
                    │   (Stage 2)     │  Default: DES-train (in-sample 2020..2023)
                    │                 │  With --des-oof: DES-train is OOF from
                    │                 │  an inner ATT (leakage-free for Stage 3)
                    └────────┬────────┘  + test (OOS 2024..2026) probabilities
                             ▼
                    ┌─────────────────┐
                    │  tw50_des.py    │  KNORA-E (K=30) over 5 ATT probabilities
                    │   (Stage 3)     │  rule-based backtest retained as diagnostic
                    │                 │  --strict-oof aborts if any aspect's
                    │                 │  DES-train rows are not source='oof'
                    └────────┬────────┘  Writes DES probabilities
                             ▼
                    ┌─────────────────┐
                    │      dqn/       │  Signal-Conditioned Double DQN
                    │   (Stage 4)     │  DES + 10 OHLC bars + position + P&L
                    └─────────────────┘  Skip / Buy / Close
```

## Repository layout

```
tw50_pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── tw50_top50.csv               # 50 stock IDs + market cap weights
├── features/                    # 5 aspects x 50 stocks = 250 CSVs
│   ├── fundamental_<id>.csv
│   ├── trade_<id>.csv
│   ├── tech_trend_<id>.csv
│   ├── moment_<id>.csv
│   └── macro_<id>.csv
├── prices/                      # user-supplied OHLCV per stock (git-ignored)
│   └── <id>.csv                 # populated by fetch_prices.py
├── fetch_prices.py              # yfinance -> prices/<id>.csv helper
├── tw50_flood.py                # Stage 1: hyperparameter + flooding-b search
├── tw50_dflood.py               # Stage 2: Dynamic Flooding retrain + predict (--des-oof)
├── tw50_des.py                  # Stage 3: KNORA-E ensemble + diagnostic backtest
├── Makefile                     # one-command recipes (Linux/WSL/macOS)
├── run.ps1                      # equivalent PowerShell task runner (Windows)
├── artifacts/                   # generated at runtime (git-ignored)
│   ├── flood/{hyperbayes,feature_selection,feature_scaler,experiments}/
│   ├── dflood/{feature_selection,feature_scaler,models,pred}/
│   └── des/{pred,models,backtest}/
├── evaluation/                  # shipped back-test CSVs + figure regen script
└── dqn/                         # Stage 4: paper-aligned Double DQN execution
```

## Install

Python 3.11 (tested), 3.10 also works.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt        # tested compatible ranges
# or, to reproduce the paper's exact environment:
pip install -r requirements-lock.txt   # exact versions (pip freeze snapshot)
```

```bash
# Linux / WSL / macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt        # tested compatible ranges
# or, to reproduce the paper's exact environment:
pip install -r requirements-lock.txt   # exact versions (pip freeze snapshot)
```

If you want the price fetcher (Stage 3 backtest), install `yfinance` in addition:

```powershell
pip install yfinance
```

TensorFlow 2.21 uses the GPU on Linux/WSL if CUDA is available; on Windows it will fall back to CPU (which is fine for smoke testing).

## Quick start — end-to-end smoke test (one stock, ~5 minutes on CPU)

The commands below reproduce the smoke test that was executed on 2026-07-30 in this repo. Elapsed times were measured on Windows / CPU-only TF 2.21.

### One-command targets

Use the shipped task runners to avoid copy-pasting stage-by-stage:

```bash
# Linux / WSL / macOS
make smoke        # in-sample DES-fit (legacy behaviour)
make smoke-oof    # OOF DES-fit (leakage-free; recommended)
make full-2330    # production settings for TSMC (~20 min on GPU, uses --des-oof)
make seed-sweep   # multi-seed Stage 3 sweep -> mean +/- std CSV (§IV.H evidence)
```

```powershell
# Windows PowerShell (no `make` required)
.\run.ps1 smoke
.\run.ps1 smoke-oof
.\run.ps1 full-2330
.\run.ps1 seed-sweep
```

Run `make help` or `.\run.ps1 help` for the full target list, and `make preflight` / `.\run.ps1 preflight` for environment sanity checks.

### Seed reproducibility

All three stages accept `--seed N` (default `42`, or set `DESQ_SEED` env var). The
global RNG seed is threaded into `PYTHONHASHSEED`, `random`, `numpy`,
`tf.keras.utils.set_random_seed`, `tf.config.experimental.enable_op_determinism()`,
`kt.oracles.BayesianOptimizationOracle(seed=...)`, and
`RandomForestClassifier(random_state=...)`. Use `scripts/run_seed_sweep.py` to
regenerate the multi-seed mean ± std evidence CSV expected in §IV.H:

```bash
python scripts/run_seed_sweep.py --stock-ids 2330,2454 \
    --seeds 42,123,456,789,2024 --stages 3
# -> artifacts/seed_sweep/per_run.csv + aggregate.csv
```

Pass `--stages 23` or `--stages 123` to also retrain Stage 2 / Stages 1+2 per seed
(slower; used when reviewers question tuner determinism).

### Baseline reproducibility (US market: DSR-Yang, MACE, combined)

The `us/baselines/` tree ships CSV outputs from the DSR-Yang and MACE baselines
across `dow30 / sp100 / ndx100`, plus the joint `combined_comparison.py`
summary. Reviewers with the required US price data (default `d:\US_stock`) can
regenerate every CSV and diff it against what we ship:

```bash
# 1) One-shot: snapshot shipped CSVs, rerun all baselines, diff (tol=1e-6).
make rerun-baselines
# or:  bash us/baselines/run_all_baselines.sh

# 2) Diff-only (uses an existing us/baselines/_shipped_snapshot/):
make verify-baselines
```

`verify_baselines.py` walks `_shipped_snapshot/` recursively, matches every
`metrics.csv`, `predictions.csv`, `selections.csv`, `equity_*.csv`,
`combined_stats.csv`, `*_comparison.csv`, prints a per-file `PASS/FAIL` with the
worst-numerical-column diff, and exits `1` on any drift. This gives reviewers a
one-command falsifiable check that the shipped baselines are not hand-tuned
snapshots.

### Reviewer reproducibility kit (public data only, 10 min CPU)

`reproducibility/` gives a reviewer without Cmoney access a falsifiable
end-to-end check on the TW50 pipeline. See
[`reproducibility/README.md`](reproducibility/README.md) for the full
walkthrough; the one-command path is:

```bash
make repro   # hash-shipped + verify-prices + smoke-oof, seed=42
```

Individual pieces:

* `make hash-shipped` — prints SHA-256 of the shipped `tw50_top50.csv`,
  `prices/2330.csv`, and 5 aspect features. Cross-check against the fingerprints
  pinned in [`reproducibility/EXPECTED_OUTPUT.md`](reproducibility/EXPECTED_OUTPUT.md).
* `make verify-prices` — downloads TWSE OHLCV from Yahoo Finance for the
  shipped tickers and asserts the shipped `prices/*.csv` differ only by a
  clean split multiplier (proves the prices are not fabricated).
* `EXPECTED_OUTPUT.md` — pins the expected `summary.csv` columns, tolerance
  bands for `total_ret_model`, and the multi-seed `aggregate.csv` values.

### Explicit commands (equivalent to `make smoke`)

```powershell
# 0. Fetch OHLCV for 2330 (needed by Stage 3 backtest).
python fetch_prices.py --stock-ids 2330

# 1. Stage 1 — Bayesian tuning + static Flooding (all 5 aspects, ~4 min).
python tw50_flood.py --stock-ids 2330 --aspect all --trials 2 --epochs 3 --batch-size 128

# 2. Stage 2 — Dynamic Flooding retrain + predict (~30 s).
python tw50_dflood.py --stock-ids 2330 --aspect all --epochs 5 --batch-size 128

# 3. Stage 3 — KNORA-E ensemble + backtest (~12 s).
python tw50_des.py --stock-ids 2330 --no-show
```

**Expected smoke-test output (Stage 3 tail):**

```text
=== DES: 2330 ===
[2330] fitting RandomForest ...
[2330] fitting KNORA-E ...
[2330] cum_model=0.217, cum_stock=2.002, excess=-1.785, buys=18, sells=18
[SUMMARY] 1 rows -> artifacts/des/backtest/summary.csv
```

`cum_model=0.217` means +21.7% cumulative return over the test window with only 2 tuner trials and 3-epoch ATT models — this is a **plumbing smoke test**, not a production result. See "Production settings" below for realistic numbers.

## Full run — one stock (production settings)

Uses the tuner budget the code was designed for.

```powershell
python fetch_prices.py --stock-ids 2330

python tw50_flood.py  --stock-ids 2330 --aspect all --trials 12 --epochs 80
python tw50_dflood.py --stock-ids 2330 --aspect all --epochs 120
python tw50_des.py    --stock-ids 2330 --no-show
```

Expected wall time on a mid-range GPU (Linux/WSL, TF 2.21 with CUDA): roughly 20 minutes per stock end-to-end. On Windows CPU it is much slower — prefer WSL for full runs.

## Batch — full TW-50

```powershell
# Fetch all 50 tickers up front (yfinance can throttle; --sleep controls pacing).
python fetch_prices.py --top50 --sleep 0.4

python tw50_flood.py  --top50 --aspect all
python tw50_dflood.py --top50 --aspect all
python tw50_des.py    --top50 --no-show
```

Results land in `artifacts/des/backtest/summary.csv`, one row per stock.

## Preflight checklist

Run these once before your first batch to make sure your environment is ready:

```powershell
# 1. Deps import
python -c "import tensorflow as tf, keras_tuner, deslib, sklearn, joblib; print('tf', tf.__version__, 'sklearn', sklearn.__version__, 'deslib', deslib.__version__)"

# 2. GPU visible (Linux/WSL only; expect an empty list on Windows).
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Features present.
python -c "from pathlib import Path; n = len(list(Path('features').glob('*.csv'))); print(f'features: {n} CSVs')"

# 4. Prices present for stocks you plan to backtest.
python -c "from pathlib import Path; ids = ['2330','2454']; missing = [s for s in ids if not (Path('prices')/f'{s}.csv').exists()]; print('missing prices:', missing)"
```

## Notes and caveats

- **Prices are user-supplied.** `tw50_des.py` reads `prices/<stock_id>.csv` with columns `Date,Open,High,Low,Close,Volume`. Use `fetch_prices.py` (yfinance) or bring your own source. Without a price CSV, Stage 3 still saves the DES/RF probability files but skips the backtest.
- **Walk-forward rolling constants**: `WF_N_SPLITS=5`, `WF_VAL_RATIO=0.2`, and `WF_GAP=50` anchor intervals. The effective gap implements the paper equation as a 20-day label horizon followed by a separate 30-trading-day purge. `WF_GAP` remains overridable for diagnostics, but values below 50 are not the revised-paper protocol.
- **Stage 2 emits both train and test predictions.** By default the DES-train window (2020-01-01..2023-12-31) is predicted by the same ATT that was trained on 2010-2023, so those rows are *in-sample* w.r.t. the ATT. The reported out-of-sample metrics still come from the strictly held-out 2024-01-01..2026-03-31 window, but the KNORA-E meta-learner does see in-sample ATT probabilities during its fit.
  - Pass `--des-oof` to `tw50_dflood.py` to instead train an inner ATT on `TRAIN_START..(DES_TRAIN_START - WF_GAP)` and use it to predict the DES-train window. The resulting rows are tagged `source='oof'` in the CSV. Test-window rows are always tagged `source='test'` and produced by the final ATT trained on the full 2010-2023 window.
  - Pass `--strict-oof` to `tw50_des.py` to abort the run if any aspect's DES-train rows are not `source='oof'` (a leakage guard for reproducibility scripts).
  - The Makefile / `run.ps1` targets `smoke-oof`, `full-2330`, `full-flagships`, and `full-top50` all use `--des-oof` + `--strict-oof` by default.
- **deslib 0.3.7 + scikit-learn 1.7 compat.** `tw50_des.py` monkey-patches `BaseEstimator._validate_data` so deslib's `KNORAE.fit(...)` keeps working. Nothing else in your environment is affected.
- **Environment override variables** (all optional):
  - `FEATURE_ROOT` — where to look for `<aspect>_<id>.csv` (default: `./features`).
  - `MODEL_ROOT` — where Stage 1 stores tuning artifacts (default: `./artifacts/flood`).
  - `DFLOOD_ROOT` — where Stage 2 stores retrained model + preds (default: `./artifacts/dflood`).
  - `DES_ROOT` — where Stage 3 stores KNORA-E artifacts (default: `./artifacts/des`).
  - `PRICES_DIR` — where Stage 3 reads OHLCV (default: `./prices`).

## Troubleshooting

| Symptom                                                              | Fix                                                                                                              |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: tensorflow` / `keras_tuner` / `deslib`         | `pip install -r requirements.txt` in the active venv.                                                            |
| `AttributeError: 'KNORAE' object has no attribute '_validate_data'`  | Pull latest `tw50_des.py` — it applies the deslib+sklearn-1.7 compat shim automatically.                         |
| Stage 3 fails with `DES train slice too short (0)`                   | Re-run Stage 2 with the current `tw50_dflood.py` (older versions only wrote the test window).                    |
| `[stock] no price CSV at prices/<id>.csv; skipping backtest`         | Run `python fetch_prices.py --stock-ids <id>`.                                                                   |
| yfinance returns EMPTY for a ticker                                  | The Yahoo symbol is `<id>.TW`. Delisted or newly listed stocks may lack coverage; check on finance.yahoo.com.    |
| TF logs `Cannot dlopen some GPU libraries` in WSL                    | Export `LD_LIBRARY_PATH` to include the pip nvidia lib dirs before launching Python.                             |

## Stage 4 — Double DQN execution

The `dqn/` subfolder contains the revised paper's execution layer, adapted from `tunglich/Market-Timing-DQN` and driven by the Stage 3 `<DES>` feature. Its defaults now use Double-DQN targets, prioritised replay, $\gamma=0.99$, a 5,000-step hard target update, and Taiwan buy/sell costs of 0.1425%/0.4425%. See [dqn/README.md](dqn/README.md).

## How to cite

Please cite the specific release DOI rather than the head of the main branch,
so reviewers can reproduce the exact bytes you ran against. The `v1.0-desq`
tag is the reviewer snapshot; every future release will mint a fresh Zenodo
DOI while the "concept DOI" links them all.

**BibTeX** (replace `10.5281/zenodo.XXXXXXX` with the DOI printed on the
Zenodo record once the release is minted):

```bibtex
@software{chen_desq_2026,
  author    = {Chen, Tung-Li},
  title     = {DESQ: Dynamic Ensemble Selection for Quantitative
               Stock Selection (v1.0-desq)},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0-desq},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX},
  note      = {Reviewer reproducibility kit for the IEEE Access
               submission; see reproducibility/README.md}
}
```

Machine-readable metadata lives in [CITATION.cff](CITATION.cff) (parsed by
GitHub's "Cite this repository" button) and [.zenodo.json](.zenodo.json)
(consumed by Zenodo when the GitHub release is archived). Update the DOI
placeholder in both files after Zenodo mints the identifier.

## License

MIT for the source code — see [LICENSE](LICENSE).

### Data licensing

The repository distributes three categories of data with different licensing terms; be sure you comply with the applicable terms before redistributing anything derived from them.

| Location                                    | Origin                                     | License / redistribution status                                                                                                                                        |
| ------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `features/*.csv` (TW-50 aspect features)    | Derived from licensed CMoney fundamental / chip-flow data. | **Derived features only**; raw CMoney data is *not* included. Redistributed here for academic reproducibility of the paper. Commercial re-use requires a CMoney licence. |
| `evaluation/*.csv`, `us/baselines/**/*.csv` | Produced by the scripts in this repository. | MIT, same as the source code.                                                                                                                                          |
| `prices/*.csv`                              | User-supplied (e.g. yfinance via `fetch_prices.py`). | Subject to the source provider's terms; git-ignored, never committed.                                                                                                  |
| `us/features/**.csv` (Dow30 / SP100 / NDX100 aspects) | Derived from public sources (yfinance + FRED). | Redistributable under MIT; independently reproducible via the scripts in `us/`.                                                                                        |

If you only need to verify the framework end-to-end without a CMoney licence, use the US extension (`us/`) which relies solely on public data.
