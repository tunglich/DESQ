---
name: backtesting-US
description: >-
  Run portfolio-level backtests on US equity universes (Dow 30, S&P 100, SOX 30,
  or custom ticker lists) where each constituent trades on its own pre-trained
  DES signal (`model_pred_DES_US/DES_pred_<TKR>_2019-12-31.csv` + CUSUM filter
  from `cumSum_prob_12/`). Period-start capital is allocated by **price-weighted**
  (DJIA-style) or **market-weighted** (S&P 500-style) scheme; daily portfolio
  equity is the sum of per-stock equity curves, plotted against a yfinance
  benchmark (default `^DJI` / `^OEX` / `^SOX`). USE WHEN the user asks for
  portfolio backtest, weighting comparison, alpha vs Dow/Nasdaq/SOX, or wants to
  rerun the same engine on a custom ticker subset. DO NOT USE for single-stock
  DES backtest (that already runs inside `DES_update_ATT_US.py`), for retraining
  DES/ATT, or for non-US universes.
---

# Portfolio Backtest (US) Skill

Driver: [`Backtest_Portfolio_US.py`](../../../Backtest_Portfolio_US.py).
Strategy: long-only, **each constituent trades independently** on its DES
probability (threshold + optional CUSUM-direction filter). Buy/sell executes at
**next-day Open** after a signal; flat in cash otherwise. Per-stock initial
capital = `weight_i × total_capital`; portfolio equity = `Σ_i equity_i(t)`.

## Golden rules

1. **Run in `finlabUS`** (Windows works for pure backtests because deslib isn't
   needed at backtest time; WSL also fine). Always with `--no-capture-output`.
2. **Inputs must already exist** for every ticker in the universe:
   - `model_pred_DES_US/DES_pred_<TKR>_2019-12-31.csv` (DES probability)
   - `cumSum_prob_12/cusum_<TKR>.csv` (CUSUM direction filter, when enabled)
   Missing tickers are **warning-skipped** and remaining weights are renormalised
   so Σweight = 1 and Σinitial_capital = `--capital`.
3. **Outputs** flat under `backtest_portfolio_US/`, filenames include
   `<universe>_<weight>_<start>_<end>`. Three files per run: `equity_*.csv`,
   `summary_*.csv`, `cum_return_*.png`.
4. **The backtest does not retrain anything.** It reads cached DES predictions.
   To refresh predictions, rerun `DES_update_ATT_US.py` first.
5. **Headless / CI**: set `MPLBACKEND=Agg`, `SHOW_FIG=0` and trigger
   non-interactive single-run mode by exporting at least `BT_UNIVERSE`.

## Components

| File | Role | Writes to |
|------|------|-----------|
| `Backtest_Portfolio_US.py` | interactive + env-driven backtest driver | `backtest_portfolio_US/` |
| `DES_update_ATT_US.py` (import only) | source of `DOW30_NAME`, single-stock trade-engine reference (`plot_backtest`); imported but its `main()` is **not** invoked | — |
| `feature/_us_data.py` | OHLCV (`load_price_frames`), benchmark Series (`load_market_index`), dividends (`_dividends`), universe constants (`DOW_30_TICKER`, `SOX_30_TICKER`) | `feature/_raw/` cache |
| `_sp100_diff.py` | exposes the 101-ticker `SP100` list (incl. GOOG + GOOGL) | — |
| `model_pred_DES_US/DES_pred_<TKR>_2019-12-31.csv` | per-stock DES probability series, columns `Date, 0` | — |
| `cumSum_prob_12/cusum_<TKR>.csv` | per-stock CUSUM direction filter (no header) | — |

## Common tasks

### 1. Dow 30, price-weighted, default 2024-01-01 ~ 2026-03-31, `^DJI` benchmark (interactive)
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python Backtest_Portfolio_US.py
# press Enter through every prompt → uses defaults
```

### 2. Dow 30, market-weighted, headless one-shot via env vars (PowerShell)
```powershell
$env:MPLBACKEND='Agg'; $env:SHOW_FIG='0'; $env:SAVE_FIG='1'
$env:BT_UNIVERSE='dow30'; $env:BT_WEIGHT='2'
$env:BT_START='2024-01-01'; $env:BT_END='2026-03-31'
conda run -n finlabUS --no-capture-output python Backtest_Portfolio_US.py
```

### 3. WSL detached batch (multiple runs)
Use `wsl_backtest_smoke.sh` as a template; pass `BT_WEIGHT=1` or `2` via env to
contrast price vs market weighting against the same universe.

### 4. Custom ticker subset, no CUSUM filter, threshold 0.55
```powershell
$env:BT_UNIVERSE='AAPL,MSFT,NVDA,GOOGL'
$env:BT_BENCHMARK='SPY'
$env:BT_CUSUM='2'
$env:BT_THRESHOLD='0.55'
conda run -n finlabUS --no-capture-output python Backtest_Portfolio_US.py
```

### 5. Ticker list from file
Create `my_universe.txt` (one ticker per line, `#` for comments), then:
```powershell
$env:BT_UNIVERSE='@my_universe.txt'
conda run -n finlabUS --no-capture-output python Backtest_Portfolio_US.py
```

## Key env vars

| Var | Default | Notes |
|-----|---------|-------|
| `BT_UNIVERSE` | `dow30` | `dow30` / `sp100` / `sox30` / comma list / `@file.txt`. Setting this triggers **non-interactive single-run mode** (script exits after one backtest). |
| `BT_WEIGHT` | `1` | `1`=price-weighted, `2`=market-weighted. |
| `BT_START` / `BT_END` | `2024-01-01` / `2026-03-31` | Inclusive ISO dates. |
| `BT_CAPITAL` | `100000000` | Total USD allocated at period start. |
| `BT_BENCHMARK` | universe default | Any yfinance symbol (`^DJI`, `^GSPC`, `^OEX`, `^SOX`, `SPY`, `DIA`, ...). Empty = universe default. |
| `BT_CUSUM` | `1` | `1`=apply CUSUM-direction filter on entries/exits, `2`=signal-only. |
| `BT_TOTAL_RETURN` | `2` | `1`=add cash dividends to cash on ex-date (price-return + div reinvest into cash), `2`=pure price return. |
| `BT_THRESHOLD` | `0.50` | DES probability threshold to flip `AGG_DES1` to 1. |
| `BT_NON_INTERACTIVE` | unset | Force single-run mode even when other `BT_*` not set. |
| `MPLBACKEND` | `TkAgg` | Set to `Agg` for headless / WSL without X. |
| `SHOW_FIG` | `1` | `0` to suppress `plt.show()`. |
| `SAVE_FIG` | `1` | `0` to suppress writing `cum_return_*.png`. |
| `CUDA_VISIBLE_DEVICES` | `-1` recommended | Backtest is pure CPU (sklearn + numpy); set to `-1` to keep GPU free for other work. |

## Output schema

`equity_<base>.csv` — daily wide table:
- `portfolio_equity` (USD), `portfolio_cum_return` (decimal, 0 at t0)
- `benchmark_cum_return` (decimal, when benchmark resolves)
- `<TKR>_asset` for each constituent (USD; missing leading days back-filled with `initial_capital_i`)

`summary_<base>.csv` — one row per ticker plus `TOTAL` and `BENCH:<symbol>` rows:
- `ticker, weight, initial_capital, final_asset, return_pct, n_buy, n_sell, sharpe, max_dd_pct`
- Sharpe = annualised (`rf=0`, `√252`), MaxDD computed on the equity series.

`cum_return_<base>.png` — single panel: portfolio (red) + benchmark (black), serif/300-dpi.

## Gotchas & fixes

- **`get_shares_full` fallback ladder for market-weighting**:
  1. `yfinance.Ticker(t).get_shares_full(start, end)` (true historical) — take
     the last sample with index `≤ start_date`;
  2. `Ticker(t).fast_info['shares']` (latest snapshot, biased for old start dates);
  3. `feature/_raw/av_BALANCE_SHEET_<t>.json` → `commonStockSharesOutstanding`;
  4. **Skip** the ticker (warning) and renormalise weights over the rest.
  The `summary_*.csv` row stays; `src` is printed in the Top-5 header line.
- **Missing DES_pred / cusum CSV** → ticker is skipped with a warning, weights
  are renormalised so Σweight = 1 and Σinit = `--capital`. Run
  `DES_update_ATT_US.py` (or `wsl_des_dow30.sh` for batch) to fill missing
  predictions before retrying.
- **Yfinance `^OEX` (S&P 100)** sometimes returns thin data outside extended
  hours; if you see "no benchmark in window", retry with `BT_BENCHMARK=SPY` or
  `^GSPC`.
- **PowerShell text encoding**: this workspace's `.py` files are UTF-8. Never
  use `Get-Content | Set-Content` round-trips on these files (it re-encodes via
  the OEM codepage and mangles Chinese strings). For programmatic line edits,
  drive Python directly (`conda run ... python <helper>.py`).
- **Cumulative return at t0 should be 0.0**. If it isn't, an aligned-but-NaN
  per-stock equity slipped into the sum — open `equity_*.csv` first column and
  verify there are no NaN per-stock columns at the first index.
- **Sharpe > 3 in smoke runs** is plausible (long-only momentum, no costs,
  signal sees walk-forward DES probabilities). It is **not** a bug; treat as a
  ceiling that real-world frictions (slippage, taxes, capacity) will erode.
- **Total-return mode** only credits cash dividends to the cash leg; it does
  **not** apply to the benchmark (benchmark is index price). So
  `benchmark_cum_return` is always a price index regardless of `BT_TOTAL_RETURN`.
- **`DES_update_ATT_US.py` import safety**: backtest imports `DOW30_NAME` from
  this module. The module's interactive loop is guarded behind
  `if __name__ == "__main__": main()`, so importing **does not** trigger any
  prompts. If you ever see a prompt during backtest startup, the guard was
  reverted — restore it.
