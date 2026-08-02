# Batch_training Agent

An automated batch-training agent that runs ATT+Flood (AutoML
hyperparameter search) followed by ATT+Dflooding (fixed-parameter
repeated training) over a list of tickers, with **live per-epoch
monitoring** and **automatic WSL restart** on GPU hangs.

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

Pick a validation mode (applies to the whole batch):

```bat
Batch_training.bat 3293 --validation-mode traditional
Batch_training.bat 3293 --validation-mode walk_forward_expanding
Batch_training.bat 3293 --validation-mode walk_forward_rolling
```

> `traditional` is a synonym for `blocking` (the legacy split strategy).

If `--validation-mode` is not supplied and this is a new batch,
`Batch_training.sh` will prompt interactively:
- `1) traditional`
- `2) walk-forward expanding`
- `3) walk-forward rolling` (default)

**Batch-level feature-preprocessing decision** (asked once, inherited by
all sub-tasks):

```bat
Batch_training.bat 2317,2301,3231 --no-preprocess   # skip feature preprocessing
Batch_training.bat 2317,2301,3231 --preprocess      # force feature preprocessing on
Batch_training.bat 2317,2301,3231                    # prompt once (Y/n)
```

> `ATT+Flood.py` / `ATT+Dflooding.py` originally prompted "run feature
> preprocessing?" once per `(stock, model_type)` pair. The agent now
> **asks once for the whole batch**, writes the decision into
> `.batch_training_state`, and passes it to every child process via the
> `FEATURE_PREPROCESS` env var. The decision is preserved across WSL
> restarts.

### 2. Run directly inside WSL (no auto-restart layer)

```bash
bash Batch_training.sh 3293
bash Batch_training.sh 3293,2330 --epoch-timeout 12
```

### 3. Reset the state file and start over

```bat
Batch_training.bat --reset
```

> ⚠️ **State-file priority rule (important)**
> If `.batch_training_state` already exists, `Batch_training.sh` treats
> it as the source of truth: the `stock_ids`, `--no-preprocess /
> --preprocess`, and `--validation-mode` passed on the command line are
> ignored (only a `[WARN]` line is printed), and neither the feature-
> preprocess nor the validation-mode prompt is re-asked.
>
> **Common symptom**: you passed new `stock_ids` (e.g. `8299,5347`) but
> training runs on the old tickers (e.g. `2330`) and you are never asked
> about feature preprocessing.
>
> **Agent handling**: when the user issues a batch-training command and
> the `stock_ids` in `.batch_training_state` differ from the command-line
> arguments, the agent **must stop and ask the user** whether to:
> 1. abort the previous unfinished batch (`--reset` and re-run with the
>    new tickers), or
> 2. continue the previous unfinished batch (ignore the new arguments).
>
> Only proceed after confirmation; never silently overwrite or reuse the
> old state.

---

## Execution flow

> **Estimated wall time per ticker (measured on RTX 5090 / 5080,
> comparable)**
> Phase 1 hyperparameter search ≈ 3 h, Phase 2 Dflooding training ≈ 2 h,
> and the downstream DES ensemble ≈ 5 min. A single stock across all 6
> factors takes about 5 h (DES not included).

1. **Phase 1 — AutoML (`ATT+Flood.py`)**
   For each `(stock_id, model_type)` pair, run Bayesian hyperparameter
   optimization (stage1: 12 trials / 80 epochs → stage2: 24 trials /
   120 epochs).

2. **Phase 2 — Dynamic Flooding (`ATT+Dflooding.py`)**
   Load the best hyperparameters produced by Phase 1 and repeat training
   18 times with fixed settings, keeping the top-3 models.

3. **Per-epoch monitoring**
   A background `tail -f` on the log file parses the per-epoch wall
   time. If two consecutive epochs exceed the threshold (default 10 s),
   the GPU is considered stuck:
   - kill the current training process,
   - exit WSL with code 42,
   - the Windows `.bat` sees exit code 42, runs `wsl --terminate`, waits
     10 s, and re-launches.

   **Special rules (to avoid false positives)**:
   - Whenever `Search: Running Trial #N` or `[Trial N] start:` is
     detected, the epoch counter and the "consecutive slow" counter are
     **reset**.
   - **Epoch 1 of every trial is always ignored** (XLA JIT compile is
     normally >10 s and is not a GPU hang); the log shows
     `(ignored: first epoch / XLA compile)`.
   - `consecutive_slow` only accumulates from epoch 2 onward; two
     consecutive >10 s epochs are required to trigger a restart.
   - This prevents the failure mode "new trial → first epoch XLA
     compile → restart triggered → recompile again", i.e. an infinite
     restart loop.

4. **Resume from checkpoint**
   The `.batch_training_state` file records which `(phase, stock, model)`
   combinations have already finished. After a WSL restart, completed
   items are skipped automatically and training resumes from the
   interruption point.

---

## Argument reference

| Argument | Default | Description |
|------|--------|------|
| `<stock_ids>` | *(required)* | Comma-separated stock IDs, e.g. `3293,2330` |
| `--model-types` | `fundamental,trade,moment,sentiment,tech_trend,macro` | Comma-separated factor types |
| `--epoch-timeout` | `10` | Per-epoch timeout in seconds |
| `--feature-preprocess` | *(interactive prompt)* | `yes` / `no`: batch-wide decision on whether to run feature preprocessing |
| `--validation-mode` | *(interactive prompt, default 3)* | `traditional` / `blocking` / `walk_forward_expanding` / `walk_forward_rolling` |
| `--no-preprocess` | — | Equivalent to `--feature-preprocess no` |
| `--preprocess` | — | Equivalent to `--feature-preprocess yes` |
| `--reset` | — | Clear the state file and start over |

---

## Environment variables

The agent sets the following environment variables internally (each can
be overridden by an outer `export`):

| Env var | Default | Description |
|----------|--------|------|
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | TF GPU memory allocator |
| `GPU_MEMORY_LIMIT_MB` | `24576` | GPU memory cap (MB) |
| `ENABLE_TF32` | `1` | Enable TF32 acceleration |
| `ENABLE_MIXED_PRECISION` | `0` | Mixed precision (recommended off under RTX 5090 + WSL for stability) |
| `ENABLE_XLA` | `0` | XLA compilation (recommended off to avoid slow first epoch) |
| `TRAIN_MODE` | `speed` | Training mode |
| `ISOLATE_STOCK_MODEL_RUNS` | `0` | The agent controls the outer loop; no in-script subprocess isolation needed |
| `FIT_VERBOSE` | `2` | Force one-line-per-epoch formatting so epoch times are easy to parse |
| `FEATURE_PREPROCESS` | *(injected by the batch decision)* | `1` = run feature preprocessing; `0` = pass-through. The agent exports this after `--feature-preprocess` / the interactive prompt so that `ATT+Flood.py` / `ATT+Dflooding.py` do not re-ask inside each subprocess |
