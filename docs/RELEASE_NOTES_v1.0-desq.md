# v1.0-desq — IEEE Access reviewer snapshot

Immutable byte-for-byte snapshot of the DESQ pipeline that accompanies the
IEEE Access submission *"Dynamic-Flooding Transformer Ensembles for
Reinforcement-Learning-Based Equity Market Timing"*.

The `main` branch may contain post-submission changes; check out this tag
for exact reproduction of every figure and CSV in the paper.

## Reviewer reproducibility kit

- **Falsifiable data manifest** — SHA-256 of every shipped Taiwan-50
  feature CSV, `tw50_top50.csv`, and `prices/2330.csv`
  (`reproducibility/MANIFEST.sha256`, 252 entries).
- **Cross-platform verifier** —
  `python reproducibility/check_manifest.py` (also runs on every push via
  GitHub Actions).
- **yfinance canary cross-check** — proves the shipped `prices/*.csv`
  are not fabricated:
  `python reproducibility/verify_public_prices.py --stock-ids 2330 --tol 0.01`.
- **3-stage smoke pipeline** — Stage-1 CUSUM → Stage-2 ATT + Dynamic
  Flooding → Stage-3 KNORA-E DES with pinned expected outputs
  (`reproducibility/EXPECTED_OUTPUT.md`, `DESQ_SEED=42`).
- **Seed sweep driver** for the §IV.H mean ± std evidence table
  (`scripts/run_seed_sweep.py`).
- **Baseline reproducibility** — `us/baselines/verify_baselines.py`
  re-runs DSR-Yang / MACE / combined baselines and diffs the shipped
  CSVs at `tol = 1e-6`.
- **CI (fast-checks job)** — manifest verify + yfinance cross-check +
  Python syntax lint on every push (see the CI badge on the README).

## Bit-for-bit reproduction

```bash
git clone --depth 1 --branch v1.0-desq https://github.com/tunglich/DESQ.git
cd DESQ
# 1. Verify shipped data is untampered (offline, ~2 s).
python reproducibility/check_manifest.py         # -> 252/252 pass
# 2. Prove shipped prices come from Yahoo Finance (network, ~10 s).
python reproducibility/verify_public_prices.py   # -> 2330 clean-split match
# 3. Run the 3-stage smoke pipeline (CPU, ~5 min).
make smoke-oof STOCK=2330 SMOKE_TRIALS=2 SMOKE_EPOCHS=3 SMOKE_BATCH=128
# Expected: artifacts/des/backtest/summary.csv row for 2330 with
# total_ret_stock ≈ 2.002 and total_ret_model in [-1.0, 3.0].
```

Windows: replace `make ...` with `.\run.ps1 smoke-oof`.

## Citing this release

- Machine-readable: [`CITATION.cff`](../CITATION.cff) (parsed by the
  GitHub *Cite this repository* button) and [`.zenodo.json`](../.zenodo.json).
- BibTeX template: see the *How to cite* section of the README.
- The DOI will be minted by Zenodo when this GitHub Release is published;
  after that, replace the placeholder in `CITATION.cff`, `.zenodo.json`,
  and the DOI badge in the README with the real
  `10.5281/zenodo.<id>` string.

## Highlights since the previous public tag (`paper`)

- A2/A5 + B1/B3 — leakage-free DES fit (`--des-oof` / `--strict-oof`),
  `WF_GAP=20` default (label horizon ≥ gap), one-command reproduction.
- A4 — global seed control threaded through PYTHONHASHSEED, `random`,
  numpy, keras, TF op determinism, KerasTuner, and RandomForest.
- B2 — reviewer reproducibility kit (`reproducibility/`) with pinned
  expected outputs and public-data-only smoke path.
- B4 — seed-sweep driver for §IV.H mean ± std evidence.
- B5 — baselines reproducibility infra (`us/baselines/` rerun +
  `verify_baselines.py` diff).
- C1 — GitHub Actions CI (`fast-checks`: manifest + yfinance + lint;
  optional `smoke`: full 3-stage on 2330).
- C2 — CITATION.cff, .zenodo.json, DOI-ready README (this release).

## License

- Source code: MIT (`LICENSE`).
- Data licensing: see the *Data licensing* section of `README.md`.
