---
name: us-stock-pipeline
description: >-
  Operate the US (Dow 30) stock-prediction pipeline in this workspace: generate
  the four feature families (fundamental, moment, tech_trend, macro), run the
  ATT+Flood hyper-parameter search, and run ATT+Dflooding final training. USE
  WHEN the user wants to (re)generate features, add/refresh a ticker, run or
  smoke-test training, debug the finlabUS conda env, Alpha Vantage fundamentals,
  or the feature/hyper/scalar/selection/experiment outputs. DO NOT USE for
  unrelated Python tasks or the original Taiwan scripts (FeatureUS.py,
  ATT+Flood.py, ATT+Dflooding.py).
---

# US Stock (Dow 30) Pipeline Skill

This workspace ports a Taiwan stock-prediction architecture to the Dow 30.
Always operate inside the **`finlabUS`** conda environment and stream output
with `--no-capture-output` (plain `conda run` buffers until exit, which looks
like "no response").

## Golden rules

1. Run everything via `conda run -n finlabUS --no-capture-output python <script>`.
2. The `_US` scripts are the live ones. Never edit the Taiwan originals
   (`FeatureUS.py`, `ATT+Flood.py`, `ATT+Dflooding.py`) — they are reference.
3. Feature families are exactly: `fundamental, moment, tech_trend, macro`.
   Never reintroduce `sentiment` or `trade`.
4. Generated CSVs live in `feature/`, indexed by `Date`; the **last 4 columns
   are labels** `y_10,y_20,y_40,y_60`. Keep columns identical across tickers.
5. Validate with a single ticker/family before launching the full 120-job run.

## Components

| File | Role | Writes to |
|------|------|-----------|
| `feature/_us_data.py` | data layer: yfinance OHLCV + `^DJI`, Alpha Vantage fundamentals, dividends; caches to `feature/_raw/` | — |
| `FeatureUS_US.py` | feature generator | `feature/<family>_<ticker>.csv` |
| `ATT+Flood_US.py` | stage-1/2 hyper-parameter search | `hyper/`, `scalar/` |
| `ATT+Dflooding_US.py` | final training (dynamic flooding) | `experiment/`, `selection/` |

## Common tasks

### Generate features
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python FeatureUS_US.py            # all 30
conda run -n finlabUS --no-capture-output python FeatureUS_US.py AAPL       # one ticker
```
Expect ~5344 rows; tech_trend 25 cols, moment 17, fundamental 18, macro 19.

### Hyper-parameter search (smoke → full)
```powershell
$env:FAST_DEBUG='1'; $env:STOCK_IDS='AAPL'; $env:MODEL_TYPES='tech_trend'; $env:CUDA_VISIBLE_DEVICES='-1'
conda run -n finlabUS --no-capture-output python "ATT+Flood_US.py"
```
Then unset the smoke vars for the full run.

### Final training (smoke → full)
```powershell
$env:STOCK_IDS='AAPL'; $env:MODEL_TYPES='tech_trend'; $env:CUDA_VISIBLE_DEVICES='-1'; $env:MAX_EPOCHS='6'
conda run -n finlabUS --no-capture-output python "ATT+Dflooding_US.py"
```

## Key env vars

`STOCK_IDS`, `MODEL_TYPES`, `CUDA_VISIBLE_DEVICES` (`-1`=CPU), `FAST_DEBUG`,
`FIT_VERBOSE`, `MAX_EPOCHS`, `VALIDATION_MODE` (blocking/walk_forward_*),
`FEATURE_PREPROCESS`, `STAGE1/2_MAX_TRIALS`, `STAGE1/2_EPOCHS`,
`ISOLATE_STOCK_MODEL_RUNS`, `AV_API_KEY`. Path roots overridable via
`ATT_*_DIR` / `*_ROOT`.

## Gotchas & fixes (learned)

- **"No response" from `conda run`** → it buffers; add `--no-capture-output`.
- **Alpha Vantage** free tier ≈ 25 req/day, 3 req/ticker. Hitting the limit is
  fine — `feature/_raw/` cache makes runs resumable. Delete a specific cache
  file to force a refresh.
- **FMP is unusable for fundamentals** (free tier caps at 5 quarters). Do not
  switch back; Alpha Vantage provides deep history.
- **DY column**: tickers without dividends (e.g. AMZN) must still emit `DY=0`
  so columns stay consistent across all 30 tickers.
- **fundamental qoq/acc metrics**: income statements (~81q) and EPS (~121q)
  arrive on different report dates — compute growth on each clean index
  separately, then outer-join. Never diff across the interleaved union.
- **Dividends**: normalize the yfinance dividend index to midnight
  (`.normalize()`) or DY misaligns to trading dates and goes all-zero.
- **Missing matplotlib** → `conda run -n finlabUS pip install matplotlib seaborn`.
  `pyodbc`/`tensorflow_addons` are optional (guarded / dead code).
- **Non-TTY prompts** auto-resolve: validation → `blocking`, preprocess → on.
- **PowerShell + conda output**: redirect to a file
  (`... > out.txt 2>&1; Get-Content out.txt`) when piped output is swallowed.

## WSL2 GPU training (RTX 5090 / Blackwell)

Native Windows TF ≥ 2.11 has **no GPU**. GPU runs under **WSL2 Ubuntu-24.04**
in a separate `finlabUS` env (py3.11, `tensorflow[and-cuda] 2.21`). The Windows
`finlabUS` stays CPU-only.

- **Always** `wsl -d Ubuntu-24.04 -- bash -lic "..."`. The default distro
  `docker-desktop` has no bash and will fail.
- **PowerShell mangles inline bash** (`|`, `$()`, `tr`, quotes). Put every WSL
  command in a `.sh` file and run `bash <script>.sh`. Never inline pipes.
- **GPU libs don't load by default** ("Cannot dlopen some GPU libraries" →
  `GPU devices: []`). Fix: put the pip `nvidia-*-cu12` lib dirs on
  `LD_LIBRARY_PATH`. Use the wrapper `wsl_gpu_run.sh`; verify with
  `bash wsl_gpu_run.sh python _gputest.py` → `RESULT: GPU OK`.
- **Blackwell sm_120**: TF 2.21 has no prebuilt kernels → first GPU op
  JIT-compiles PTX (slow once, then cached). The warning is expected.
- **conda run --no-capture-output gives a TTY** → the search script prompts for
  validation mode and feature preprocess. Export `VALIDATION_MODE` and
  `FEATURE_PREPROCESS=1` to skip (done in `wsl_train_first15.sh`).
- **Fresh full-budget run**: keras-tuner resumes from `hyper/ATT_<fam>_<tkr>/`.
  Delete those dirs first (`wsl_clear_first15.sh`) or a prior smoke/FAST_DEBUG
  run will be resumed instead of a clean 12/24-trial search.

Helper scripts (repo root): `wsl_gpu_run.sh` (LD_LIBRARY_PATH wrapper),
`wsl_smoke_gpu.sh` (1-ticker smoke), `wsl_clear_first15.sh`,
`wsl_train_first15.sh` (full first-15, logs to `logs/`), `wsl_checklog.sh`.
