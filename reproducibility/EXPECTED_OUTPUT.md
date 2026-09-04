# Expected output — smoke reproducibility

This file pins the artifacts users should see after running the smoke
demo (`make smoke-oof` or its Windows equivalent) at commit `HEAD` with
`DESQ_SEED=42`. It is the falsifiable half of the anti-fabrication argument:
if numbers you obtain drift outside the tolerance bands, either your setup
diverges from the pinned one (see `../requirements-lock.txt`) or something
under `tw50_pipeline/` was tampered with after this file was written.

Regenerate this file whenever the shipped data or pinned seed changes:

```bash
make smoke-oof                              # produces artifacts/des/backtest/summary.csv
python reproducibility/hash_shipped.py      # section 1
python reproducibility/verify_public_prices.py --stock-ids 2330   # section 3
```

---

## 1. Shipped data fingerprints (SHA-256, first 16 hex chars)

These are the inputs to the pipeline. Any change here means the shipped data
was modified — the numbers below no longer apply.

| file | lines | sha256 (prefix) |
| --- | ---: | --- |
| `tw50_top50.csv`                | 51   | `64d981198d67cc28` |
| `prices/2330.csv`               | 1753 | `24343fa42a1544ab` |
| `features/fundamental_2330.csv` | 8209 | `df88bc69a53ada91` |
| `features/trade_2330.csv`       | 6624 | `2b4c3c364fec9d06` |
| `features/tech_trend_2330.csv`  | 8209 | `c13c835477535a98` |
| `features/moment_2330.csv`      | 8209 | `0bd628e83ebb32ee` |
| `features/macro_2330.csv`       | 8209 | `552b0c2cc0f0c333` |

> Prefixes above are SHA-256 of the LF-normalized blob as stored in git
> (matches `git show :<file> | sha256sum`). On Windows with
> `core.autocrlf=true` you may see a different hash for the working-tree
> file; run the verification from a Linux/WSL clone or from
> `python reproducibility/check_manifest.py` (which resolves that
> difference via `.gitattributes`).

Recompute:

```bash
for f in tw50_top50.csv prices/2330.csv features/{fundamental,trade,tech_trend,moment,macro}_2330.csv; do
    printf '%s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$f"
done
```

---

## 2. Stage-3 smoke output (`make smoke-oof`, seed=42)

Command:

```bash
DESQ_SEED=42 make smoke-oof STOCK=2330
```

Expected `artifacts/des/backtest/summary.csv` (single row for 2330):

| column | value | tolerance |
| --- | --- | --- |
| `stock_id`         | `2330` | exact |
| `has_prices`       | `True` | exact |
| `n_test_days`      | `540`  | exact (test window 2024-01-01..2026-03-31 minus warmup) |
| `total_ret_stock`  | `2.002 ± 1e-3` | exact — buy-and-hold, deterministic |
| `total_ret_model`  | `1.685 ± 0.05` | repeated seed-42 clean-clone result; tolerance permits backend numeric drift |
| `acc_buy`          | `10 ± 3` | count of correct buys |
| `acc_sell`         | `9 ± 3`  | count of correct sells |

Any exact-column mismatch (`stock_id`, `has_prices`, `n_test_days`,
`total_ret_stock`) indicates the pipeline is reading a different data slice —
suspect either a data-window regression in `tw50_dflood.py` or a modified
`prices/2330.csv`.

---

## 3. Multi-seed sweep (`make seed-sweep`, n=3)

Command:

```bash
make seed-sweep STOCK=2330 SWEEP_SEEDS=42,123,456
```

Expected `artifacts/seed_sweep/aggregate.csv` central tendency
(±1σ across 3 seeds; wider bands with only 3 samples are normal):

| metric | mean | std |
| --- | --- | --- |
| `n_test_days`        | 540    | 0     (must be 0) |
| `total_ret_stock`    | 2.002  | 0     (must be 0) |
| `total_ret_model`    | 0.31   | 0.14  |
| `excess_ret`         | -1.69  | 0.14  |
| `acc_buy`            | 10.3   | 1.5   |
| `acc_sell`           | 9.3    | 1.5   |

Non-zero `n_test_days_std` or `total_ret_stock_std` proves the sweep script
mixed test windows — this is a bug, not a randomness question.

The wide `total_ret_model_std / |total_ret_model_mean| ≈ 0.46` is the
reason we do not use single-seed excess returns as headline results. Validate
the mean ± std over at least five seeds with:

```bash
make seed-sweep STOCK=2330 SWEEP_SEEDS=42,123,456,789,2024
```

---

## 4. Public-price cross-check (`verify_public_prices.py`)

Command:

```bash
python reproducibility/verify_public_prices.py --stock-ids 2330
```

Expected (2026-08-03 yfinance snapshot):

```text
[PLAN] verifying 1 tickers against yfinance, tol=0.01
  PASS     2330: n=1752, median_ratio=1.0000, max_rel_dev=0.00e+00

[SUMMARY] pass=1, fail=0, total=1
```

`median_ratio=1.0000` and `max_rel_dev=0.00e+00` means the shipped
`prices/2330.csv` is byte-for-byte the same series Yahoo Finance publishes
for TWSE 2330. Any `median_ratio` other than `1.0` or a clean split fraction
(0.5, 0.2, ...), or `max_rel_dev` > 1%, means someone edited the shipped
prices — that would falsify the study directly.

If Yahoo has retroactively applied a split that we did not, expect
`median_ratio ≠ 1.0` **but** `max_rel_dev` still tiny — that scenario is
benign and documented in the script comments.

---

## 5. Environment snapshot used to write this file

- OS: Windows 10 host, WSL2 Ubuntu-24.04 for training
- Python: `3.11.14` (conda env `finlab`)
- Key pins: TensorFlow `2.21.0`, Keras `3.14.0`, DESlib `0.3.7`,
  scikit-learn `1.7.1`, keras-tuner `1.4.8`
- Full lock: [`../requirements-lock.txt`](../requirements-lock.txt)
- Commit: run `git rev-parse HEAD` to record the exact tree state
