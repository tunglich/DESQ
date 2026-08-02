# `DES_update_ATT-sentiment.py` — usage guide

## Overview

This script uses **Dynamic Ensemble Selection (DES)** to combine the
predictions of six per-aspect ATT models with CUSUM statistics, produce
buy/sell signals, and then backtest and plot the performance.

### Flow

```
ATT model predictions → combine 6 aspects → RF tuning → KNORAE ensemble
                     → CUSUM filter → backtest trades → performance report
```

---

## Environment

| Item | Version |
| --- | --- |
| Python | 3.10+ |
| tensorflow | 2.20.0-dev (self-built, sm_120) |
| deslib | latest |
| scikit-learn | latest |
| matplotlib | (TkAgg backend) |
| pandas / numpy | latest |

### Install dependencies

```bash
python -m venv ~/venvs/finlab
source ~/venvs/finlab/bin/activate
pip install -r requirements.txt
```

---

## Prerequisite data

The following inputs must exist under `D:/` before you run the script:

| Path | Description | Source |
| --- | --- | --- |
| `D:/experiments_df_test/ATT_{aspect}_{stock_id}/experiment_result_*.csv` | Per-aspect ATT model predictions | Produced by `ATT+Dflooding.py` |
| `D:/Feature_new/fundamental_{stock_id}.csv` | Fundamental features (includes the `y_20` label) | `Feature_Cmoney_update.py` |
| `D:/CmoneyFactor/Open.csv`, `Close.csv`, `High.csv`, `Low.csv`, `Volume.csv` | CMoney OHLCV | `Feature_Cmoney_update.py` |
| `D:/CmoneyFactor/Stock_name.csv` | Stock name lookup | CMoney |
| `./cumSum/cusum_{stock_id}.csv` | CUSUM statistics | `CUMSUM_feature_finlab.py` |
| `./cumSum_prob_6/cumsum_prob_{stock_id}.csv` | CUSUM probability series | `CUSUM_prob_multi_finlab.py` |

### Outputs

| Path | Description |
| --- | --- |
| `D:/DES_model_test/DES_{stock_id}_{period}.pkl` | Trained DES model |
| `D:/RF_model_test/RF_{stock_id}_{period}.pkl` | Trained RF base classifier |
| `D:/model_pred_DES_test/DES_pred_{stock_id}_{period}.csv` | DES predictions |
| `D:/model_pred_RF_test/RF_pred_{stock_id}_{period}.csv` | RF predictions |
| `D:/model_output/ensemble_{stock_id}.png` | Signal overview (10 subplots) |
| `./evaluation/backtest_{stock_id}_L1S1.png` | Backtest performance chart |

---

## Running the script

> **Estimated wall time per ticker**: the DES ensemble (RF tuning +
> KNORAE + CUSUM filter + backtest) takes about **5 minutes** (RTX 5090
> and 5080 are comparable). This does **not** include the two upstream
> ATT training phases: Phase 1 hyperparameter search ≈ 3 h and Phase 2
> Dflooding ≈ 2 h; see `README_Batch_training.md`.

### Interactive mode (default)

At launch the script uses `input()` to ask for a stock ID:

```text
Enter stock id: 2330
```

### Notes

1. **Must run in an environment with a display**: the script uses the
   `matplotlib` TkAgg backend and pops up windows.
2. **WSL users** need to configure X11 forwarding, or set
   `show_fig = False` (modify line 476).
3. If the model and prediction files already exist (`.pkl` / `.csv`),
   the script loads them directly and skips retraining.

---

## Global parameters

Key parameters you can tune (in the main block):

| Parameter | Default | Description |
| --- | --- | --- |
| `train_start` | `'2007-08-01'` | Training window start date |
| `train_end` | `'2024-06-30'` | Training window end date |
| `test_start` | `'2024-07-01'` | Test window start date |
| `period` | `['2019-12-31']` | Model retraining timestamps (multiple values → rolling updates) |
| `long` | `1` | Number of consecutive bullish days required for a buy signal |
| `short` | `1` | Number of consecutive bearish days required for a sell signal |
| `threshold` | `0.50` | Probability > threshold → bullish |
| `span` | `1` | EWM smoothing span (1 = no smoothing) |
| `show_fig` | `True` | Pop up figure windows |
| `save_fig` | `True` | Save figures to disk |

---

## Full pipeline order

This script is the last step (ensemble + backtest). The full pipeline:

```
1. Feature_Cmoney_update.py        → refresh CMoney OHLCV / features
2. CUMSUM_feature_finlab.py         → compute CUSUM statistics
3. CUSUM_prob_multi_finlab.py       → compute CUSUM probability series
4. ATT+Dflooding.py                 → train ATT models (AutoML + Flooding)
5. prediction_ATT_update.py         → batch-update ATT model predictions
6. DES_update_ATT-sentiment.py      → DES ensemble + CUSUM filter + backtest ← this script
```

---

## Example output

After completion the script prints a performance report:

```text
trades:         12
winning trades: 8
win rate:       0.67
gross profit:   15234567.89
avg profit:     1904320.99
gross loss:     -3456789.12
avg loss:       -864197.28
profit factor:  2.20
```

It also produces:
- **Signal overview chart** (10 subplots): price, DES raw / smoothed /
  blended signals, and per-aspect signals for the six aspects.
- **Backtest performance chart**: model cumulative return vs
  buy-and-hold, with buy/sell markers and model-update points annotated.
