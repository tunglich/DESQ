# Batch_training Agent (RTX 5080 profile)

An automated batch-training agent that runs ATT+Flood (AutoML
hyperparameter search) followed by ATT+Dflooding (fixed-parameter
repeated training) over a list of tickers, with **live per-epoch
monitoring** and **automatic WSL restart** on GPU hangs. This profile is
tuned for RTX 5080.

---

## Architecture

```
Windows                           WSL (Ubuntu / finlab env)
┌──────────────────┐              ┌────────────────────────┐
│ Batch_training   │─── wsl ────▶│ Batch_training.sh        │
│ .bat             │              │  ├─ ATT+Flood.py   (P1)  │
│                  │◀── exit 42 ──│  ├─ ATT+Dflooding.py(P2) │
│ wsl --terminate  │              │  └─ monitor_log()        │
│ wait 10s         │              └────────────────────────┘
│ re-launch ───────│─── wsl ────▶ (resume from state file)
└──────────────────┘
```

**Why two layers?**
`wsl --terminate` kills the whole WSL instance, including the monitor
script itself. The outer `.bat` therefore lives on the Windows side and
is responsible for detecting exit code 42, restarting WSL, and
re-invoking `.sh`.

---

## Quick start

### 1. Run from Windows CMD / PowerShell

```bat
Batch_training.bat 3293
```

Multiple tickers:

```bat
Batch_training.bat 3293,2330,2317
```

Set the per-epoch timeout in seconds (default 10 s):

```bat
Batch_training.bat 3293 --epoch-timeout 15
```

Train only specific factors:

```bat
Batch_training.bat 3293 --model-types fundamental,trade,moment
```

Choose validation strategy (default: walk-forward rolling):

```bat
Batch_training.bat 3293 --validation walk_forward_rolling
Batch_training.bat 3293 --validation walk_forward_expanding
Batch_training.bat 3293 --validation blocking
```

Without `--validation` and with no env override, the WSL side prompts
interactively (`1`/`2`/`3`, 10-second timeout, default 3). See
[SKILL.md](SKILL.md#6-validation-strategy).

### 2. Run directly inside WSL (no auto-restart layer)

```bash
bash Batch_training.sh 3293
bash Batch_training.sh 3293,2330 --epoch-timeout 12
```

### 3. Reset the state file and start over

```bat
Batch_training.bat --reset
```

---

## Execution flow

> **Important**: every agent invocation runs **Phase 1 → Phase 2 back
> to back** with no human intervention. Once Phase 1 AutoML finishes,
> Phase 2 Dynamic Flooding starts automatically (guaranteed by the
> `main()` loop in `Batch_training.sh` / `run_att_agent.sh`). Both
> phases are recorded in the `completed` list of
> `.batch_training_state`, and a WSL restart mid-run resumes from the
> interruption point.

> **Estimated wall time per ticker (measured on RTX 5090 / 5080,
> comparable)**
> Phase 1 hyperparameter search ≈ 3 h, Phase 2 Dflooding training ≈ 2 h,
> and the downstream DES ensemble ≈ 5 min. A single stock across all 6
> factors takes about 5 h (DES not included).

1. **Phase 1 — AutoML (`ATT+Flood.py`)**
   For each `(stock_id, model_type)` pair, run Bayesian hyperparameter
   optimization (stage1: 12 trials / 80 epochs → stage2: 24 trials /
   120 epochs). Output is written to
   `D:/hyperbayes_ATT/ATT_{model_type}_{stock_id}/best_trial_summary.json`.

2. **Phase 2 — Dynamic Flooding (`ATT+Dflooding.py`)**
   Load the best hyperparameters produced by Phase 1 and repeat training
   18 times with fixed settings, keeping the top-3 models under
   `D:/experiments_ATT/`. Both phases receive the stock list through
   the same `STOCK_IDS` env var (do not rename it; see SKILL.md §7).

3. **Per-epoch monitoring**
   A background `tail -f` on the log parses each epoch's wall time. If
   two consecutive epochs exceed the threshold (default 10 s), the GPU
   is considered stuck:
   - kill the current training process,
   - exit WSL with code 42,
   - the Windows `.bat` sees exit code 42, runs `wsl --terminate`, waits
     10 s, and re-launches.

4. **Resume from checkpoint**
   The `.batch_training_state` file records which `(phase, stock,
   model)` combinations have completed. After a WSL restart, completed
   items are skipped and training resumes from the interruption point.
   The state file is only cleared once the very last Phase 2 job
   finishes.

---

## Argument reference

| Argument | Default | Description |
|------|--------|------|
| `<stock_ids>` | *(required)* | Comma-separated stock IDs, e.g. `3293,2330` |
| `--model-types` | `fundamental,trade,moment,sentiment,tech_trend,macro` | Comma-separated factor types |
| `--epoch-timeout` | `10` | Per-epoch timeout in seconds |
| `--validation` | *(interactive prompt)* | `blocking` / `walk_forward_expanding` / `walk_forward_rolling` |
| `--wf-splits` | `5` | Number of walk-forward folds |
| `--wf-val-ratio` | `0.2` | Per-fold validation ratio |
| `--wf-gap` | `20` | Train/val purge gap (>= 20-day label horizon) |
| `--reset` | — | Clear the state file and start over |

---

## Environment variables

The agent sets the following environment variables internally (each can
be overridden by an outer `export`):

| Env var | Default | Description |
|----------|--------|------|
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | TF GPU memory allocator |
| `GPU_MEMORY_LIMIT_MB` | `12288` | GPU memory cap (MB); 0 = unlimited |
| `ENABLE_TF32` | `1` | Enable TF32 acceleration |
| `ENABLE_MIXED_PRECISION` | `0` | Mixed precision (recommended off under RTX 5080 + WSL for stability) |
| `ENABLE_XLA` | `0` | XLA compilation (recommended off to avoid slow first epoch) |
| `TRAIN_MODE` | `speed` | Training mode |
| `ISOLATE_STOCK_MODEL_RUNS` | `0` | The agent controls the outer loop; no in-script subprocess isolation needed |
| `FIT_VERBOSE` | `2` | Force one-line-per-epoch formatting so epoch times are easy to parse |
| `FEATURE_PREPROCESS` | `0` | Feature preprocessing off by default (pass-through) so batch runs are not blocked by an interactive prompt; set `1` to enable |
| `VALIDATION_MODE` | `walk_forward_rolling` | `blocking` / `walk_forward_expanding` / `walk_forward_rolling` (agent overrides the Python default) |
| `WF_N_SPLITS` | `5` | Walk-forward fold count |
| `WF_VAL_RATIO` | `0.2` | Per-fold validation ratio |
| `WF_GAP` | `20` | Train/val purge gap (>= 20-day label horizon) |
| `VENV_ACTIVATE` | `$HOME/venvs/finlab/bin/activate` | Path to the Python venv activate script (preferred) |
| `CONDA_ENV_NAME` | `finlab` | Conda env name (only used when the venv is missing) |
| `CONDA_SH_PATH` | `$HOME/miniconda3/etc/profile.d/conda.sh` | Conda init script (fallback) |

---

## File layout

```
docs/att_batch_training/
├── Batch_training.bat        # Windows launcher (WSL restart loop)
├── Batch_training.sh         # WSL training agent (core logic)
├── ATT+Flood.py              # Phase 1: AutoML hyperparameter search
├── ATT+Dflooding.py          # Phase 2: fixed-parameter repeated training
├── .batch_training_state     # Runtime state file (auto-created / cleared)
├── logs/                     # Training log directory
└── README_Batch_training.md  # This document
```

---

## Common scenarios

### GPU repeatedly times out and restarts

If WSL restarts hit the retry cap (default 20), the `.bat` gives up.
Things to try:
- lower `GPU_MEMORY_LIMIT_MB`
- set `TRAIN_MODE=safe`
- check GPU cooling

### Restart from a specific phase / model

Edit the `completed=` field in `.batch_training_state` and add the
markers you want to skip.

### Only run Phase 2 (AutoML results already exist)

Mark every `automl:*` entry in `.batch_training_state` as completed and
set `current_phase=dflooding`.

---

## Dependencies

- Windows 10/11 + WSL2 (default distro name `Ubuntu`; if different,
  edit `WSL_DISTRO` in `Batch_training.bat`)
- WSL Python environment: prefer the `~/venvs/finlab` venv; fall back
  to the `finlab` conda env if the venv does not exist
- RTX 5080 (or any other CUDA GPU)
- Python packages: see [requirements.txt](requirements.txt) (produced
  by `pip freeze` in the finlab venv).
  - Install:
    `python -m venv ~/venvs/finlab && source ~/venvs/finlab/bin/activate && pip install -r requirements.txt`
  - Note: TensorFlow is a self-built wheel
    (`tensorflow==2.20.0-dev0+selfbuilt`, with sm_120 support for
    Blackwell RTX 5080/5090); it is not on PyPI and must be installed
    from a local wheel or built from source.
