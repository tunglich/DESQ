# Reproducibility kit

This folder gives any third party a
**10-minute, no-proprietary-data, exit-code-based** falsification path for the
DESQ pipeline's central results.

Nothing in here downloads a Cmoney API key or asks for the FRED tokens the
original experiments used. Every required input is either (a) already shipped in
`prices/`, `features/`, `tw50_top50.csv`, or (b) publicly fetchable from
Yahoo Finance in one command.

## What each script proves

| script                              | proves                                                                                                     |
| ---                                 | ---                                                                                                        |
| `check_manifest.py`                 | Every shipped `features/*.csv`, `prices/*.csv`, `tw50_top50.csv` matches the SHA-256 pinned in `MANIFEST.sha256`. Also enforced in CI (`ci.yml -> fast-checks`) — any post-hoc edit turns the badge red. |
| `verify_public_prices.py`           | The shipped `prices/*.csv` are byte-identical (up to split adjustments) to Yahoo Finance — no fabrication. |
| `hash_shipped.py`                   | Prints the SHA-256 fingerprints of a small (7-file) fast-check subset in a human-readable table.           |
| `EXPECTED_OUTPUT.md`                | The `make smoke-oof` / `make seed-sweep` outputs at commit HEAD fall inside pinned tolerance bands.        |
| `../us/baselines/verify_baselines.py` | The shipped US baseline CSVs equal a fresh baseline rerun (see B5, requires US data).                    |

## Quickstart (Linux / WSL, ~10 min CPU)

```bash
# 0. install (once)
python -m pip install -r requirements-lock.txt

# 1. verify every shipped features/*, prices/*, tw50_top50.csv (252 files)
python reproducibility/check_manifest.py
# expect: MANIFEST OK: 252 files verified against reproducibility/MANIFEST.sha256

# 2. cross-check shipped prices against Yahoo Finance
python reproducibility/verify_public_prices.py --stock-ids 2330
# expect: PASS 2330: n=1752, median_ratio=1.0000, max_rel_dev=0.00e+00

# 2. verify shipped feature/data fingerprints
python reproducibility/hash_shipped.py
# expect: 7 lines, each SHA-256 prefix matching EXPECTED_OUTPUT.md §1

# 3. end-to-end smoke (Stage 1 -> 2 -> 3, seed=42)
DESQ_SEED=42 make smoke-oof STOCK=2330
# expect: cum_model in [1.635, 1.735], stock buy-hold = 2.002 (deterministic)

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

- `check_manifest.py` FAIL — a shipped `features/*.csv`, `prices/*.csv`, or
  `tw50_top50.csv` was edited after the manifest was recorded. The CI badge on the main
  README turns red the moment this happens.
- `verify_public_prices.py` FAIL — the shipped `prices/*.csv` was edited
  after the manifest was recorded. Falsifies the OHLCV inputs directly.
- Stage 3 `total_ret_stock` ≠ 2.002 — the test window shifted; a data
  pipeline change slipped in. `n_test_days ≠ 540` has the same meaning.
- Stage 3 `total_ret_model` outside `[1.635, 1.735]` for seed=42 — either
  the seed threading in Stage 1/2 broke, or a stochastic layer added
  new randomness. Reproduce with `make seed-sweep` to confirm.

## What this kit does NOT prove

- Correctness of the Cmoney-derived fundamental features. Those are
  proprietary. The fingerprint check only proves the shipped CSVs were
  not tampered with after commit HEAD; it does not audit the values
  themselves. A separate aspect drop-in ablation is the
  argument that fundamental features contribute non-trivially and thus
  are not padding.
- Generality to non-TW50 universes. The 50-stock focus is a scope
  decision. B5 (`us/baselines/`) provides the US-market cross-check.
