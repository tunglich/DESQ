---
name: att-validation
description: >-
  Run the validation-aware ATT flooding ablation pipeline on 2330: re-search
  hyperparameters with the F1 objective (val_fbeta_score), execute the
  none/static/dynamic FLOOD_MODE comparison via ATT+Dflooding_floodexp.py,
  save train/val/test predictions with correct dates, then build the DES
  stage with KNORAE on top via DES_update_ATT_floodexp.py. USE WHEN the user
  wants to re-tune flooding b, change repeats, adjust the walk-forward
  validation window, debug PR-AUC / sample-weight / date-alignment issues,
  or generate the flood comparison and DES comparison plots. DO NOT USE for
  the US Dow 30 pipeline (that is the us-stock-pipeline skill) or the
  unmodified Taiwan originals (ATT+Flood.py, ATT+Dflooding.py,
  DES_update_ATT-sentiment.py).
---

# ATT Validation + Flooding Ablation Skill

This skill covers the 2330-specific flooding-ablation workflow. The relevant
files are **`ATT+Dflooding_floodexp.py`**, **`DES_update_ATT_floodexp.py`**,
**`plot_flood_compare.py`**, **`wsl_flood_hpo.sh`**, **`wsl_flood_exp.sh`**.
The originals (`ATT+Dflooding.py`, `DES_update_ATT-sentiment.py`) are
reference and must stay unchanged.

> **Script location**: the **executed copies** live at the workspace root
> (`D:/US_stock/`) — that is what the WSL launchers and any running
> training are using. A **reference snapshot** of the same scripts is
> bundled under this skill folder
> (`.github/skills/att-validation/`). Always edit the workspace-root copy
> (`D:/US_stock/`), then copy the changed file to
> `.github/skills/att-validation/` to keep the snapshot current. Never edit
> the skill-folder copy directly. Do not run the bundled copies directly —
> paths inside the launchers assume the workspace-root layout.

## Golden rules

1. Conda is unavailable in non-interactive WSL bash. Always use the absolute
   interpreter path:
   `PYBIN=/home/tungl/miniconda3/envs/finlabUS/bin/python` (env is
   **finlabUS**, not finlab).
2. The VS Code terminal tool rewrites `&&` → `;` in WSL bash. Use `;` or
   pass the chain inside the quoted `bash -lc "..."` string.
3. HPO outputs live at `D:/hyperbayes_test_f1/ATT_<aspect>_2330/`. Never
   delete this directory when restarting the flood experiment — it is the
   source of best hyperparameters.
4. Flood experiment outputs live at `D:/experiment_flood/<mode>/`; aggregate
   plots at `D:/evaluation_plot/<mode>/` and `D:/evaluation_plot/_compare/`.
   DES outputs at `D:/DES_flood/<mode>/`.
5. Six aspects: `fundamental, trade, tech_trend, moment, sentiment, macro`.
6. Validation = walk-forward rolling, **last fold** (`wf_folds[-1]`),
   covering 2024-09-03 ~ 2025-12-31 for 2330. Test = 2026-01-01+.

## Pipeline

```text
HPO (F1)  ─►  Flood ablation (none/static/dynamic × 6 aspects × repeats)
   │              │
   │              └─► per repeat saves 4 CSVs:
   │                    experiment_result_<r>.csv          (legacy: train+test concat)
   │                    experiment_result_train_<r>.csv    (correct dates, ATT train segment)
   │                    experiment_result_val_<r>.csv      (correct dates, ATT val segment)
   │                    experiment_result_test_<r>.csv     (correct dates, 2026+)
   │
   └─► plot_flood_compare.py → side-by-side loss & metric curves
                                                   │
                                                   ▼
                                  DES_update_ATT_floodexp.py
                                     RF + KNORAE on 6-aspect stack
                                     DES train = ATT train preds
                                     DES val   = ATT val preds (2024-09 ~ 2025-12)
                                     DES test  = 2026+
                                     outputs to D:/DES_flood/<mode>/
                                     and       D:/evaluation_plot/_compare/des_compare_2330.png
```

> **Partial-mode runs**: if only a subset of modes have completed,
> `plot_flood_compare.py` and `DES_update_ATT_floodexp.py` will fail or
> silently skip missing modes. Set `FLOOD_MODES` to only the completed
> modes before running either script, e.g.
> `export FLOOD_MODES=none,static`.

## Components

| File | Role | Writes to |
|------|------|-----------|
| `wsl_flood_hpo.sh` | F1 hyper-parameter search per aspect (always invoke via this launcher; do not call the underlying Python script directly) | `D:/hyperbayes_test_f1/` |
| `ATT+Dflooding_floodexp.py` | Flood ablation training; saves 4 CSVs per repeat | `D:/experiment_flood/<mode>/`, `D:/evaluation_plot/<mode>/` |
| `plot_flood_compare.py` | Aggregate comparison across 3 modes | `D:/evaluation_plot/_compare/` |
| `DES_update_ATT_floodexp.py` | RF + KNORAE on 6-aspect ATT outputs | `D:/DES_flood/<mode>/`, `D:/evaluation_plot/_compare/des_compare_2330.{png,pdf}` |
| `wsl_flood_hpo.sh` | Launcher for full F1 HPO | logs/flood_hpo_*.log |
| `wsl_flood_exp.sh` | Launcher: 3 modes → plot | logs/flood_exp_*.log |

## Common tasks

### Full HPO (F1 objective) re-search
```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/US_stock ; bash wsl_flood_hpo.sh"
```
Objective is `kt.Objective('val_fbeta_score', 'max')`. Each aspect runs
~12 trials. Best trial summary at
`D:/hyperbayes_test_f1/ATT_<aspect>_2330/best_trial_summary.json`.

### Full flood ablation (3 modes × 6 aspects × 8 repeats)
```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/US_stock ; bash wsl_flood_exp.sh"
```
Long-running; launch in the background using `nohup` or inside a `tmux`
session so the terminal can be closed without killing the process. Example:
`nohup bash wsl_flood_exp.sh > logs/flood_exp_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
Reads HPO from `HYPER_ROOT=D:/hyperbayes_test_f1`.

Before launching `wsl_flood_exp.sh`, verify that
`D:/hyperbayes_test_f1/ATT_<aspect>_2330/best_trial_summary.json` exists
for all six aspects. If any file is missing, re-run HPO for that aspect
with `STOCK_IDS=2330 MODEL_TYPES=<aspect> bash wsl_flood_hpo.sh` before
proceeding.

### DES on top
```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/US_stock ; /home/tungl/miniconda3/envs/finlabUS/bin/python DES_update_ATT_floodexp.py"
```
Fast (~1-2 min per mode). RandomizedSearchCV picks RF; KNORAE selects per
test window.

### Clean restart for flood experiment
```powershell
Remove-Item -Recurse -Force "D:\experiment_flood"
Remove-Item -Recurse -Force "D:\evaluation_plot"
# DO NOT touch D:\hyperbayes_test_f1
```

## Key env vars

`STOCK_IDS=2330`, `MODEL_TYPES=fundamental,trade,moment,sentiment,tech_trend,macro`,
`NUM_REPEATS=8`, `MAX_EPOCHS=300`, `VALIDATION_MODE=rolling`,
`FEATURE_PREPROCESS=1`, `HYPER_ROOT=D:/hyperbayes_test_f1`,
`FLOOD_MODE` ∈ `{none, static, dynamic}`, `STATIC_FLOOD_B=0.3`,
`DISABLE_EARLY_STOPPING=1`, `CUDA_VISIBLE_DEVICES=0`.

For DES: `FLOOD_MODES=none,static,dynamic`, `EXP_ROOT=D:/experiment_flood`,
`OUTPUT_ROOT=D:/DES_flood`, `TRAIN_END=2024-09-02`, `VAL_START=2024-09-03`,
`VAL_END=2025-12-31`, `TEST_START=2026-01-01`, `RF_ITER=30`, `RF_CV=5`,
`KNORAE_K=10`, `THRESHOLD=0.5`.

## Tuning knobs (verified)

- **Static b**: `STATIC_FLOOD_B=0.3`. Below 0.25 the loss term is too small
  to bend the optimizer; above 0.5 training under-fits on 2330.
- **Dynamic b**: starting `b ∈ np.linspace(0.3, 0.5, NUM_REPEATS)`;
  `DynamicFloodingCallback(min_b=0.3, max_b=0.5, step_up=0.5,
  step_down=0.3, patience=4, min_delta=1e-4)`. Big step sizes are
  intentional — small steps barely moved the floor on this dataset.
- **Monitor for dynamic / EarlyStopping / ReduceLROnPlateau**:
  `val_pr_auc` (Keras key from `AUC(curve='PR', name='pr_auc')`).
- **Validation fold**: `wf_folds[-1]` from `WalkForwardSplit(n_splits=5,
  val_ratio=0.2, gap=10, rolling=True)`. Gives 324 windows for 2330
  spanning 2024-09-03 ~ 2025-12-31.

## Critical: window date alignment

Original `ATT+Dflooding.py` indexed predictions with
`data.index[-(len(X_train)+len(X_test)):]` — a hack that breaks when val
exists. The `_floodexp` script uses an explicit helper right after
`get_windows`:

```python
def window_indices_to_dates(window_indices, slc, n_steps_, data_index):
    min_idx = 1 * 250 - 1                    # floor used inside get_windows
    base = max(min_idx, slc.start - n_steps_ + 1)
    arr = np.asarray(window_indices, dtype=int)
    return data_index[base + arr + n_steps_ - 1]
```

This is the **only** correct way to map a `(slice, window_index)` pair to
a calendar date. Always feed `train_slice` / `test_slice` along with the
respective `train_indices` / `val_indices` / `np.arange(len(X_test))`.

## DES setup (sklearn 1.7 + deslib 0.3.7 compat)

deslib 0.3.7 imports a private sklearn helper that moved in 1.7. The
`DES_update_ATT_floodexp.py` script monkey-patches at import time:

```python
import sklearn.utils.validation as _skv
if not hasattr(_skv, '_check_pos_label_consistency'):
    from sklearn.metrics._classification import _check_pos_label_consistency as _cpl
    _skv._check_pos_label_consistency = _cpl
```

Without this shim, `from deslib.des import KNORAE` raises ImportError.

## Pitfalls & fixes (learned this session)

- **val accuracy ~0.24 < base rate**: caused by a `val_recall` HPO
  objective combined with class imbalance (model collapses to all-positive).
  → switched to `val_fbeta_score` (F1) and applied `class_weight` via
  per-sample weights in `tf.data` pipelines.
- **`val_pr_auc` returns NaN under HPO**: small val fold + single-class
  windows. → final HPO uses sklearn F1 (robust); flood training keeps Keras
  `pr_auc` only as the monitor (not the objective).
- **Old `experiment_result_<r>.csv` had wrong dates**: the legacy file is
  preserved for backward compatibility but `DES_update_ATT_floodexp.py`
  reads the new `_train_/_val_/_test_` files instead.
- **Custom `FloodingModel` save/load**: model uses
  `tf.keras.models.save_model(...keras)`. Reloading later requires
  `custom_objects={'FloodingModel': FloodingModel}`. Predictions only need
  `model.predict`, no recompile.
- **Walk-forward determinism**: with the same data, `VALIDATION_MODE`,
  `WF_N_SPLITS=5`, `val_ratio=0.2`, `gap=10`, the split is identical
  across modes and repeats. Safe to compare metrics across `FLOOD_MODE`.

## Outputs to expect

```
D:/hyperbayes_test_f1/ATT_<aspect>_2330/best_trial_summary.json
D:/experiment_flood/<mode>/ATT_<aspect>_2330/
    experiment_<r>.keras
    experiment_result_<r>.csv          (legacy)
    experiment_result_train_<r>.csv    (NEW)
    experiment_result_val_<r>.csv      (NEW)
    experiment_result_test_<r>.csv     (NEW)
    history_<r>.csv
D:/evaluation_plot/<mode>/ATT_<aspect>_2330/loss_<r>.png  metric_<r>.png
D:/evaluation_plot/_compare/loss_compare_<aspect>.png
D:/DES_flood/<mode>/
    rf_pred_2330.csv  des_pred_2330.csv
    rf_model_2330.pkl des_model_2330.pkl
    metrics_2330.json
D:/DES_flood/metrics_summary_2330.json
D:/evaluation_plot/_compare/des_compare_2330.{png,pdf}
```

## Reference: WSL terminal patterns

```powershell
# launch a long async run
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/US_stock ; bash wsl_flood_exp.sh"

# quick py_compile check
wsl -d Ubuntu-24.04 -- bash -lc "/home/tungl/miniconda3/envs/finlabUS/bin/python -m py_compile 'ATT+Dflooding_floodexp.py' ; echo OK"

# tail latest flood log
Get-ChildItem D:\US_stock\logs\flood_exp_*.log | Sort-Object LastWriteTime -Desc | Select-Object -First 1 | Get-Content -Tail 50
```
