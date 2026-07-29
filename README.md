# TW-50 Attention + Flooding + DES Pipeline

A market-timing framework for the TWSE Top-50 constituents that stacks:

1. **Attention (ATT) sequence classifier** with static Flooding regularization (hyperparameter search).
2. **Dynamic Flooding** re-training with the best `flooding_b` per aspect.
3. **Dynamic Ensemble Selection (KNORA-E)** across the 5 per-aspect ATT predictions.

The pipeline uses **walk-forward rolling validation** (4:1 train:val ratio) on 5 feature aspects (no sentiment, no CUSUM).

## Data window

| Split       | Range                        |
| ----------- | ---------------------------- |
| Training    | up to **2023-12-31**         |
| Test        | **2024-01-01 ~ 2026-03-31**  |
| Validation  | last 20% of the train window (rolling, 5 folds) |

## Feature aspects (5)

`fundamental`, `trade`, `tech_trend`, `moment`, `macro`.

Each CSV under `features/` follows the layout `<aspect>_<stock_id>.csv`, with the last 4 columns being labels `y_10, y_20, y_40, y_60`.

## Universe

TWSE Top-50 constituents by market cap on 2023-12-29 — see [tw50_top50.csv](tw50_top50.csv).

## Pipeline

```
                    ┌─────────────────┐
features (5 aspects)│  tw50_flood.py  │  Bayesian tuning over ATT hyperparams
                    │                 │  + static Flooding grid b ∈ {0.0..0.4}
                    └────────┬────────┘  Saves best trial per (stock, aspect)
                             ▼
                    ┌─────────────────┐
                    │ tw50_dflood.py  │  Fixed-hyperparam retraining with
                    │                 │  Dynamic Flooding callback (b∈[0,0.4])
                    └────────┬────────┘  Emits per-aspect prediction CSVs
                             ▼
                    ┌─────────────────┐
                    │  tw50_des.py    │  KNORA-E over 5 ATT probabilities
                    │                 │  No CUSUM filter, no sentiment
                    └─────────────────┘  Backtest + performance metrics
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Stage 1: hyperparameter + flooding-b search
python tw50_flood.py --stock-ids 2330 --aspect fundamental --trials 12 --epochs 80

# Stage 2: Dynamic Flooding retraining (reads best-b from stage 1)
python tw50_dflood.py --stock-ids 2330 --aspect fundamental --repeats 3 --epochs 120

# Stage 3: DES ensemble backtest
python tw50_des.py --stock-ids 2330 --start 2024-01-01 --end 2026-03-31 --no-show

# Batch over the full TW-50
python tw50_flood.py --top50 --aspect all
python tw50_dflood.py --top50 --aspect all
python tw50_des.py --top50 --start 2024-01-01 --end 2026-03-31 --no-show
```

## Layout

```
tw50_pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── tw50_top50.csv               # 50 stock IDs and market cap weights
├── features/                    # 5 aspects × 50 stocks = 250 CSVs
│   ├── fundamental_<id>.csv
│   ├── trade_<id>.csv
│   ├── tech_trend_<id>.csv
│   ├── moment_<id>.csv
│   └── macro_<id>.csv
├── prices/                      # user-supplied OHLCV per stock (not tracked)
│   └── <id>.csv
├── tw50_flood.py                # stage 1
├── tw50_dflood.py               # stage 2
├── tw50_des.py                  # stage 3
└── artifacts/                   # generated at runtime (git-ignored)
    ├── flood/{hyperbayes,feature_selection,feature_scaler,experiments}/
    ├── dflood/{feature_selection,feature_scaler,experiments,pred}/
    └── des/{pred,des_model,rf_model}/
```

## Notes

- **No CUSUM**: this pipeline does not use CUSUM statistics anywhere. Buy/sell signals are derived purely from the DES probability threshold.
- **No sentiment**: only the 5 quantitative aspects are used.
- **Prices**: `tw50_des.py` reads user-supplied OHLCV from `prices/<stock_id>.csv` with columns `Date,Open,High,Low,Close,Volume` (Date parseable). Bring your own price source (e.g. yfinance).
- **Walk-forward rolling**: constant `WF_N_SPLITS=5`, `WF_VAL_RATIO=0.2`, `WF_GAP=10` trading days between train and validation to reduce leakage.

## License

MIT — see [LICENSE](LICENSE).
