---
name: us-daily-inference
description: >-
  Run repeatable US daily inference after DES training is available: update
  features and CUSUM for user-selected date range (default recent trading day),
  refresh ATT logits and DES predictions, then generate US AI_score and AI_tree
  per preset universe (dow30 / sox30 / ndx100 / sp100). Also covers WSL backfill
  when many DES pred files fall stale, diagnostic scripts for feature/DES/AI-score
  discontinuities, and the `keep="last"` merge invariant in prediction updates.
  USE WHEN the user asks for day-to-day refresh/inference/visualization on the
  existing US DES universe, backfill of stale DES predictions, per-index AI
  score/tree outputs, or diagnosis of flat/dropped AI score after an update.
  DO NOT USE for ATT retraining or hyperparameter search (use us-stock-pipeline
  for training), or for portfolio backtests (use backtesting-US).
---

# US Daily Inference Skill

This skill is for post-training daily operations over an already-trained US DES
universe (100+ tickers). The required execution order is:

1. Update four feature families (`fundamental, tech_trend, moment, macro`)
2. Refresh ATT logits and DES predictions (using updated features)
3. Render US AI score and US AI tree

Note: CUSUM refresh is part of the same daily pipeline and must stay in sync
with feature/prediction dates.

Primary runner: `run_us_daily_pipeline.py`.

## Golden rules

1. Default runtime is Windows `finlabUS` environment; **use WSL `finlabUS` for
   batch backfill** to avoid `bad marshal data` on legacy `.keras` models.
2. Universe default comes from existing `model_pred_DES_US/DES_pred_*` files;
   pass `--universe dow30|sox30|ndx100|sp100` for a preset, or `--all-universes`
   to score all four in one shot.
3. Default date is recent US trading day; `--start/--end` override.
4. Missing ticker artifacts are skipped and recorded in summary JSON.
5. Do not retrain ATT/DES in this skill; inference only.
6. **Merge invariant: `prediction_update_US.py` must use `keep="last"`** when
   deduplicating dates so freshly-computed predictions overwrite cached values.
   Do not revert to `keep="first"` — that silently freezes DES after retrain
   or bug-fix reruns and produces flat AI score curves.
7. **Prediction-window index invariant** — in
   `_update_att_family_predictions` the `new_vals` Series that carries
   `model.predict(X_test)` back to the CSV **must** be indexed by the
   dates actually sliced, i.e.
   `data.index[slc.stop - len(X_test) : slc.stop]`, **not**
   `data.index[-len(X_test):]`. See the 2026-07-17 anti-pattern below for
   why the tail-slice form silently corrupts one day whenever features
   extend past `end_ts` (typical when the orchestrator runs before US
   close but yfinance already returned today's intraday bar).
8. **Do not forward the orchestrator's daily `start`/`end` to
   `generate_ai_score*`** — those bounds target the feature/CUSUM/prediction
   refresh window (typically 1 trading day). AIScore filters `wide` by
   `[start, end]`; a 1-day window degenerates the score computation and
   trips the empty-fallback branch that dumps the full 2006→today history
   with reset smoothing. Let AIScore use its own defaults
   (`start="2024-01-01"`, `smooth_window=10`) inside the pipeline.

## Components

| File | Role | Output |
|------|------|--------|
| `run_us_daily_pipeline.py` | one-click orchestrator; supports `--universe` / `--all-universes` | `AI_us/summary/daily_pipeline_summary_*.json` |
| `prediction_update_US.py` | refresh ATT logits + DES pred (dedup uses `keep="last"`) | `experiment/ATT_*/experiment_result_*.csv`, `model_pred_DES_US/` |
| `AIScore_US.py` | blended score + market-cap aggregation + benchmark chart; per-universe outputs | `AI_us/score/`, `AI_us/snapshots/`, `AI_us/summary/` |
| `AITree_US.py` | sector->industry treemap; universe inferred from snapshot filename | `AI_us/tree/`, `AI_us/summary/` |
| `_diag_030_break.py` | discontinuity probe for one ticker's features/DES/CUSUM + AI score per-universe | stdout only |
| `_diag_des_freshness.py` | scan every DES pred file: last date, post-cut std, stale/flat/healthy counts per universe | stdout only |
| `_backfill_all.sh` / `_backfill_status.sh` | WSL backfill launcher + progress polling helper | log under `AI_us/summary/` |

## Default formulas and policies

- Single-stock blended score:
  - `score = 0.50*DES + 0.20*cumSum_prob_6_norm01 + 0.30*cumSum_prob_12_norm01`
  - `cumSum_prob_{6,12}` values are mapped to `[0,1]` before blending when needed.
- Aggregation weight:
  - market-cap weighted (S&P style)
- Benchmark per preset universe:
  - `dow30 -> ^DJI`, `sox30 -> ^SOX`, `ndx100 -> ^NDX`, `sp100 -> ^OEX`
  - fallback default (no `--universe`): `^NDX`
- Default AI score display start:
  - `2024-01-01`
- Missing data:
  - skip by default and record `skipped` reason
- Ticker file-safe form:
  - dots become hyphens (`BRK.B -> BRK-B`) to match on-disk filenames
- No legacy US-only aliases (removed 2026-07-15):
  - only per-universe `AI_score_{universe}_latest.*`, `AI_tree_{universe}_latest.*`,
    `AI_score_snapshot_{universe}_latest.csv`, and `AI_score_summary_{universe}_latest.json`
    are written. The old un-suffixed `AI_score_US_latest.*`, `AI_tree_US_latest.*`,
    `AI_score_snapshot_latest.csv`, `AI_score_summary_latest.json`,
    and `AI_tree_summary_latest.json` files were bit-identical to whichever
    universe ran last in `--all-universes` (sp100) and are no longer produced.
  - the full 154-ticker default run still writes `*_us_*` (lowercase) files —
    those are a genuine universe, not aliases.

## Common commands

### Required 3-step workflow (manual, explicit)

1) Features (+CUSUM via orchestrator stage):
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python run_us_daily_pipeline.py --start 2026-07-03 --end 2026-07-03
```

2) DES update only (if rerunning failed subset):
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python prediction_update_US.py --tickers AAPL,MSFT,NVDA --start 2026-07-03 --end 2026-07-03
```

3) AI_score and AITree render:
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python AIScore_US.py --start 2024-01-01 --delta-zoom 10 --smooth-window 10
conda run -n finlabUS --no-capture-output python AITree_US.py
```

### Full default daily run (recent trading day)
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python run_us_daily_pipeline.py
```

### User-specified date range
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python run_us_daily_pipeline.py --start 2026-07-03 --end 2026-07-03
```

### AI score re-render only (from existing DES outputs)
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python AIScore_US.py --start 2024-01-01 --delta-zoom 10 --smooth-window 10
```

### Per-universe AI score (fast — reads existing DES + CUSUM only)
```powershell
cd d:\US_stock
python AIScore_US.py --universe dow30          # or sox30 / ndx100 / sp100
python AIScore_US.py --all-universes           # all four in one shot
```

### Per-universe AI tree (uses matching snapshot csv)
```powershell
cd d:\US_stock
foreach ($u in 'dow30','sox30','ndx100','sp100') {
  python AITree_US.py --snapshot "AI_us\snapshots\AI_score_snapshot_${u}_latest.csv"
}
```

### Small smoke run (3 tickers)
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python run_us_daily_pipeline.py --tickers AAPL,MSFT,NVDA --start 2026-07-03 --end 2026-07-03
```

### Strict mode (fail-fast)
```powershell
cd d:\US_stock
conda run -n finlabUS --no-capture-output python run_us_daily_pipeline.py --strict
```

## Outputs to expect

```text
AI_us/
  score/
    AI_score_{universe}_YYYYMMDD.html           # dow30 | sox30 | ndx100 | sp100 | us
    AI_score_{universe}_series_YYYYMMDD.csv
    AI_score_{universe}_latest.html
    AI_score_{universe}_series_latest.csv
  snapshots/
    AI_score_snapshot_{universe}_YYYYMMDD.csv
    AI_score_snapshot_{universe}_latest.csv
  tree/
    AI_tree_{universe}_YYYYMMDD.html
    AI_tree_{universe}_YYYYMMDD.csv
    AI_tree_{universe}_latest.html
    AI_tree_{universe}_latest.csv
  summary/
    AI_score_summary_{universe}_YYYYMMDD.json
    AI_score_summary_{universe}_latest.json
    AI_tree_summary_{universe}_YYYYMMDD.json
    AI_tree_summary_{universe}_latest.json
    daily_pipeline_summary_YYYYMMDD.json
    daily_pipeline_summary_latest.json
```

No legacy `AI_score_US_latest.*` / `AI_tree_US_latest.*` /
`AI_score_snapshot_latest.csv` / `AI_score_summary_latest.json` /
`AI_tree_summary_latest.json` files are produced. Consumers must select a
specific universe.


## Troubleshooting

- If many tickers are skipped in prediction stage:
  - check `selection/`, `scalar/`, `experiment/ATT_*` completeness.
- If treemap has many `Unknown` groups:
  - metadata fetch failed/rate-limited; rerun later and cache will improve.
- If benchmark fetch fails:
  - retry with network stable, or pass `--benchmark ^DJI` temporarily.
- **AI score suddenly flat + drop after some cutoff date (very common)**:
  - Root cause is almost always **stale DES pred files** — most tickers stopped
    updating on that date, `AIScore_US.py` ffill/bfill spreads the single
    boundary value to today, and the market-cap-weighted sum collapses to a
    near-constant.
  - Confirm with `python _diag_des_freshness.py` — look at `stale (last < ...)`
    counts per universe. Anything more than a handful is a smoking gun.
  - Fix with the WSL backfill recipe below (do NOT try Windows — legacy `.keras`
    hits `bad marshal data`).
- If a single ticker's DES post-cut window is all zeros (e.g. `ALGM`, `ARM`,
  `ALAB`, `APP`, `CEG`, `NBIS`, `NOW` observed in July 2026):
  - Model output shape edge case in `_extract_positive_class_prob` — investigate
    separately, not a pipeline-wide bug.

## Anti-patterns (negative examples)

### AP-1 — Prediction one day off, "AI score today looks completely different"

**Reported (2026-07-17)**: Full run of
`run_us_daily_pipeline.py --all-universes` at ~11:26 ET (before US market
close). Series CSVs grew from ~20 KB (previous days) to ~160 KB and DOW30
latest score jumped from `0.4631` → `0.5389`. Overlap dates diverged by
`0.02-0.05`, and the plot started at `2006-01-04` instead of `2024-01-01`.

**Two independent defects had to be fixed:**

1. Indexing bug in `prediction_update_US.py::_update_att_family_predictions`.
   The old line
   ```python
   new_vals = pd.Series(pos_prob, index=data.index[-len(X_test):])
   ```
   assumes `data.index[-1] == end_ts`. It doesn't when the orchestrator runs
   before US close: `_recent_trading_day()` returned `end=2026-07-16` (Thu)
   but yfinance already delivered an intraday bar for `2026-07-17` (Fri), so
   `data.index.max()=07-17`. The 07-16 window's prediction was written under
   07-17's date; 07-16 disappeared from `experiment_result_*.csv` and
   therefore from `DES_pred_*.csv`. Verify with:
   ```powershell
   Get-Content model_pred_DES_US\DES_pred_AAPL_2019-12-31.csv -Tail 5
   # Bug signature: dates skip a trading day (e.g. 2026-07-15 -> 2026-07-17)
   # while feature and CUSUM CSVs still contain 2026-07-16.
   ```
   **Fix:** index by the actually sliced dates:
   ```python
   new_vals = pd.Series(
       pos_prob, index=data.index[slc.stop - len(X_test) : slc.stop]
   )
   ```
2. Orchestrator forwarding the daily window to AI score in
   `run_us_daily_pipeline.py`:
   ```python
   # WRONG — the "start/end" here is the feature/prediction refresh window,
   # typically a single day. AIScore then filters `wide` to that window,
   # sees only 0-1 rows, triggers the empty-fallback branch, and writes
   # 5000+ rows from 2006 with reset smoothing.
   generate_ai_score_for_universe(universe=uni, start=start, end=end)
   ```
   **Fix:** call with the AI-score defaults; do not forward the daily
   window:
   ```python
   generate_ai_score_for_universe(universe=uni, smooth_window=10)
   ```

**Repair procedure** (once the bug is patched):
```powershell
# 1. Repair ATT logits + DES for the affected tail
$t = Get-Content AI_us\summary\_tickers_all.txt   # or build from DES_pred_*.csv
conda run -n finlabUS --no-capture-output python prediction_update_US.py `
  --tickers $t --start 2026-07-10 --end 2026-07-17 `
  --summary AI_us\summary\prediction_repair_YYYYMMDD.json
# 2. Re-render AI score with the historical defaults
conda run -n finlabUS --no-capture-output python AIScore_US.py `
  --all-universes --start 2024-01-01 --delta-zoom 10 --smooth-window 10
# 3. Re-render each universe's tree from its refreshed snapshot
foreach ($u in 'dow30','sox30','ndx100','sp100') {
  conda run -n finlabUS --no-capture-output python AITree_US.py `
    --snapshot "AI_us\snapshots\AI_score_snapshot_${u}_latest.csv"
}
```

**Confirmation that the repair worked:**
- DES CSV for any ticker now shows every trading day
  (`... 07-14, 07-15, 07-16, 07-17`).
- Overlap dates between the new series and the previous
  `AI_score_{universe}_series_YYYYMMDD.csv` differ by `< 0.005` on days
  outside the repaired window (2024-01-02, 2025-01-02, 2026-06-30, etc.).
  Larger diffs are only expected on the last few smoothing-window days
  where the newly-added 07-10/07-13/07-16 rows enter the rolling mean.

**Do-not repeat guardrails:**
- Golden rules 7 and 8 above encode the invariants — keep both when touching
  `prediction_update_US.py` or the AI-score call sites in
  `run_us_daily_pipeline.py`.
- After any change to those files, run this repro:
  ```powershell
  # experiment: "pipeline runs before US close" by supplying an end BEFORE
  # data.index.max()
  conda run -n finlabUS --no-capture-output python prediction_update_US.py `
    --tickers AAPL --start 2026-07-16 --end 2026-07-16
  Get-Content model_pred_DES_US\DES_pred_AAPL_2019-12-31.csv -Tail 3
  # Must contain a row for 2026-07-16, not skip it.
  ```

## Diagnostic recipes

### 1. Feature/DES/CUSUM discontinuity around a specific date
```powershell
cd d:\US_stock
$env:DIAG_TICKER = "AAPL"   # any ticker
python _diag_030_break.py
```
Edit `CUT` in the script if the cutoff is not `2026-03-30`. Reports:
- per-facet pre/post mean & std
- top-5 jump z-scores per facet
- per-ticker DES/CUSUM values around the cut
- per-universe AI score pre/post statistics

### 2. DES freshness scan across all tickers
```powershell
cd d:\US_stock
python _diag_des_freshness.py
```
Reports, per universe:
- `stale`   = last date < today  (need backfill)
- `flat`    = post-cut std < 0.03 (model stuck)
- `healthy` = post-cut std >= 0.03
Also prints cross-section DES mean/std at a handful of reference dates so you
can see whether ffill is inflating apparent stability.

## WSL backfill recipe (when many DES pred files are stale)

Why WSL: legacy `.keras` models trained under Linux py3.11 will raise
`ValueError: bad marshal data` when loaded in Windows py3.10. Windows daily
pipeline has an auto WSL fallback per-ticker, but for a batch of dozens/hundreds
of stale tickers the direct WSL path is much faster and more reliable.

Env: `/home/tungl/miniconda3/envs/finlabUS/bin/python` on WSL `Ubuntu-24.04`.

1. Prepare helper scripts (already checked in; regenerate if missing):
   - `_backfill_all.sh` — runs `prediction_update_US.py --end YYYY-MM-DD` on all
     DES-trained tickers with output redirected to a log file.
   - `_backfill_status.sh` — polls log for `[OK]/[SKIP]/[FAIL]` counts plus
     mtime-based DES pred file count (works even when Python stdout is buffered).

2. Launch (async so it can run 1-2h without holding a terminal):
   ```powershell
   wsl -d Ubuntu-24.04 -- bash /mnt/d/US_stock/_backfill_all.sh
   ```

3. Poll progress every ~15-30 min:
   ```powershell
   wsl -d Ubuntu-24.04 -- bash /mnt/d/US_stock/_backfill_status.sh
   ```
   During the run Python stdout is buffered and `[OK]` prints only flush at
   the end. Use the file-mtime counter in the status script for real-time
   progress, not the `OK=` count.

4. When done, regenerate AI score/tree for all universes:
   ```powershell
   cd d:\US_stock
   python AIScore_US.py --all-universes
   foreach ($u in 'dow30','sox30','ndx100','sp100') {
     python AITree_US.py --snapshot "AI_us\snapshots\AI_score_snapshot_${u}_latest.csv"
   }
   ```

5. Sanity check the fix worked:
   ```powershell
   python _diag_030_break.py 2>&1 | Select-String -Pattern 'AIscore|pre60 mean|post  mean'
   ```
   Post-cut std should return to roughly the same order of magnitude as pre60
   std (or higher when market volatility increased). If post std is still 3-4x
   smaller than pre std, some tickers are still stale — rerun the backfill for
   the specific subset.

## Known ticker config gaps (as of 2026-07-05)

These fail/skip in `prediction_update_US.py` because artefacts are missing.
Remediate separately by running the `us-stock-pipeline` skill for the ticker,
or exclude them from the universe.

| Ticker | Reason |
|--------|--------|
| DHR    | `selection/DHR.json` missing `tech_trend` key (retrained 2026-07-05: fixed) |
| MCHP   | `scalar/scaler_fundamental_MCHP.pkl` missing |
| ON     | `scalar/scaler_fundamental_ON.pkl` missing (retrained 2026-07-05: fixed) |
| SHOP   | `experiment/ATT_tech_trend_SHOP/experiment_*.keras` missing |
| SPG    | `experiment/ATT_tech_trend_SPG/experiment_*.keras` missing |

## Degenerate DES models (AI score exclusion list)

Some retrained DES models end up with `classes_=[0]` (single-class output),
so `predict_proba` returns 0 for every date and `_extract_positive_class_prob`
correctly emits zeros — these tickers then drag the market-cap-weighted AI
score toward zero.

Root cause is **upstream in DES training**, not inference: the walk-forward
window fed to the DES ensemble contained no positive labels (`y_20 > 0.6%`
never triggered), so the KNN/RF/DT base learners never saw class 1.

**Mitigation until retrained with balanced labels**: `AIScore_US.py` has a
`DEGENERATE_DES_TICKERS: set[str]` constant that is filtered out of every
`generate_ai_score()` invocation. The summary JSON records the exclusion under
`dropped_degenerate_des`.

Current list (2026-07-05):
```python
DEGENERATE_DES_TICKERS = {"SNDK", "CRWD", "CEG", "APP", "ALAB"}
```
All five belong to NDX 100 only; other universes are unaffected.

**Verify a ticker is degenerate** before adding/removing from the set:
```powershell
# WSL (Windows Python lacks joblib for the pickled DES models)
wsl -d Ubuntu-24.04 -- /home/tungl/miniconda3/envs/finlabUS/bin/python /mnt/d/US_stock/_diag_des_zero.py
```
Look for `classes_=[0]` and `proba.shape=(N, 1)` — that is the degenerate
signature. Healthy models show `classes_=[0 1]` and `proba.shape=(N, 2)`.

**Fix path**: retrain via the `us-stock-pipeline` skill with either a longer
history window, a lower label threshold, or a shorter label horizon so the
training set contains both classes. After retrain, remove the ticker from
`DEGENERATE_DES_TICKERS` and re-run `AIScore_US.py --all-universes`.

## DES retraining after ATT changes (**hard rule**)

`DES_model_US/DES_{ticker}_*.pkl` is trained on a specific ATT output
distribution. If you later retrain ATT (via `us-stock-pipeline`) **or** run
`prediction_update_US.py --start 2005-01-01` (which uses `keep="last"` to
overwrite historical `experiment_result_*.csv` values with newly-computed
ATT logits), the cached DES sees a shifted feature distribution at inference
time and its predictions become miscalibrated — usually visible as **flat /
conservative DES output** relative to fresh retrain.

**Diagnostic** — check whether ATT is newer than DES pkl for any ticker:
```powershell
python _diag_des_stale.py
```
Fields:
- `stale_keras=True` → ATT `.keras` file newer than DES pkl (DES trained on
  older ATT weights entirely).
- `stale_csv=True` → `experiment_result_*.csv` newer than DES pkl (ATT logits
  used at inference differ from what DES saw during training).

**Fix — full-universe DES retrain (WSL, ~3-6 hours for ~170 tickers)**:
```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/d/US_stock/_des_retrain_all.sh
wsl -d Ubuntu-24.04 -- bash /mnt/d/US_stock/_des_retrain_all_status.sh   # poll
```
Under the hood iterates every DES-trained ticker through
`DES_update_ATT_US_range.py --ticker <T> --force-retrain`, which rebuilds
KNORAE + underlying RF via `findBestRF` on the current ATT distribution and
overwrites `DES_model_US/`, `RF_model_US/`, `model_pred_DES_US/`,
`model_pred_RF_US/`.

After it finishes, regenerate AI score/tree (they read the refreshed DES
pred CSVs):
```powershell
python AIScore_US.py --all-universes
foreach ($u in 'dow30','sox30','ndx100','sp100') {
  python AITree_US.py --snapshot "AI_us\snapshots\AI_score_snapshot_${u}_latest.csv"
}
```

**Note on RandomizedSearchCV variance**: `findBestRF` does not seed the
sampler, so each retrain draws different hyperparameters. The improvement
observed after a full retrain post-ATT-change is dominated by the
distribution shift correction, not by random variance.

## Per-ticker price cache gap healing

`feature/_us_data.py::_download_prices` used to use a single **global**
`cache_max` when deciding the incremental refresh window. If one ticker was
newly added (recent history only) or another ticker had partial data extended
to a recent date, the global max advanced but any other ticker whose own
history stopped earlier could develop a permanent interior gap that daily
runs never healed. Seen in the wild: 170/182 tickers stuck missing
2026-04-01..2026-06-10, producing a horizontal ffill tail on the DES ensemble
plot.

The current `_download_prices` (2026-07-05 patch) now inspects each ticker
individually and refetches whenever either:
- own_last date is behind `target_end - 2 days`, or
- the ticker has an internal gap > 7 days between adjacent cached dates.

**One-time heal** to repair the current cache after any concurrent DES
retrain finishes:
```powershell
python _heal_price_cache.py
```
Uses the patched logic to detect + refetch each stale ticker (respecting
yfinance rate limits with `INTER_TICKER_SLEEP=1.5s`).

**Do not run heal while a batch retrain / prediction pipeline is running** —
both processes write to `feature/_raw/prices_ohlcv.csv` and may corrupt the
merged output.

Diagnostics available:
```powershell
python _diag_price_tail.py    # focus tickers + count of stale/gapped tickers
python _diag_price_gap.py     # per-ticker largest 2026 gap + common windows
```

## Boundary with other skills

- `us-daily-inference`: inference + visualization only.
- `us-stock-pipeline`: feature generation and ATT training/hyper search.
- `backtesting-US`: portfolio backtests on generated DES signals.
