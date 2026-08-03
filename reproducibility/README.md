# Reviewer reproducibility kit

This folder gives an IEEE Access reviewer (or any third party) a
**10-minute, no-proprietary-data, exit-code-based** falsification path for the
DESQ paper's central claims.

Nothing in here downloads a Cmoney API key or asks for the FRED tokens the
authors used. Every input a reviewer needs is either (a) already shipped in
`prices/`, `features/`, `tw50_top50.csv`, or (b) publicly fetchable from
Yahoo Finance in one command.

## What each script proves

| script                              | proves                                                                                                     |
| ---                                 | ---                                                                                                        |
| `verify_public_prices.py`           | The shipped `prices/*.csv` are byte-identical (up to split adjustments) to Yahoo Finance — no fabrication. |
| `hash_shipped.py`                   | The shipped `features/*.csv` and `tw50_top50.csv` match the SHA-256 prefixes pinned in `EXPECTED_OUTPUT.md`. |
| `EXPECTED_OUTPUT.md`                | The `make smoke-oof` / `make seed-sweep` outputs at commit HEAD fall inside pinned tolerance bands.        |
| `../us/baselines/verify_baselines.py` | The shipped US baseline CSVs equal a fresh baseline rerun (see B5, requires US data).                    |

## Quickstart (Linux / WSL, ~10 min CPU)

```bash
# 0. install (once)
python -m pip install -r requirements-lock.txt

# 1. cross-check shipped prices against Yahoo Finance
python reproducibility/verify_public_prices.py --stock-ids 2330
# expect: PASS 2330: n=1752, median_ratio=1.0000, max_rel_dev=0.00e+00

# 2. verify shipped feature/data fingerprints
python reproducibility/hash_shipped.py
# expect: 7 lines, each SHA-256 prefix matching EXPECTED_OUTPUT.md §1

# 3. end-to-end smoke (Stage 1 -> 2 -> 3, seed=42)
DESQ_SEED=42 make smoke-oof STOCK=2330
# expect: cum_model in [0.17, 0.77], stock buy-hold = 2.002 (deterministic)

# 4. seed variance (3 seeds; ~15 min CPU because rerun of Stage 3 only)
make seed-sweep STOCK=2330 SWEEP_SEEDS=42,123,456
# expect: artifacts/seed_sweep/aggregate.csv matches EXPECTED_OUTPUT.md §3
```

## Quickstart (Windows PowerShell)

```powershell
python -m pip install -r requirements-lock.txt

python reproducibility\verify_public_prices.py --stock-ids 2330
python reproducibility\hash_shipped.py

$env:DESQ_SEED = '42'
.\run.ps1 smoke-oof -Stock 2330
.\run.ps1 seed-sweep -Stock 2330 -SweepSeeds '42,123,456'
```

## What a FAIL means

- `verify_public_prices.py` FAIL — the shipped `prices/*.csv` was edited
  after publication. Falsifies the OHLCV inputs directly.
- `hash_shipped.py` prefix mismatch — a shipped feature file was changed
  post-hoc. Recompute the full pipeline and update `EXPECTED_OUTPUT.md`.
- Stage 3 `total_ret_stock` ≠ 2.002 — the test window shifted; a data
  pipeline change slipped in. `n_test_days ≠ 540` has the same meaning.
- Stage 3 `total_ret_model` outside `[0.17, 0.77]` for seed=42 — either
  the seed threading in Stage 1/2 broke, or a stochastic layer added
  new randomness. Reproduce with `make seed-sweep` to confirm.

## What this kit does NOT prove

- Correctness of the Cmoney-derived fundamental features. Those are
  proprietary. The fingerprint check only proves the shipped CSVs were
  not tampered with after commit HEAD; it does not audit the values
  themselves. A separate ablation in the paper (aspect drop-in) is the
  argument that fundamental features contribute non-trivially and thus
  are not padding.
- Generality to non-TW50 universes. The 50-stock focus is a scope
  decision. B5 (`us/baselines/`) provides the US-market cross-check.
