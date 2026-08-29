# =============================================================================
# TW-50 DESQ pipeline — Makefile
# =============================================================================
# Wraps the four pipeline stages plus figure/table regeneration so reviewers
# can reproduce the paper's results with a single `make` command per artifact.
#
# Usage
# -----
#   make smoke           # end-to-end plumbing check on 2330 (~5 min, CPU OK)
#   make smoke-oof       # same, but with --des-oof leakage-free DES fit
#   make full-2330       # production settings for TSMC (~20 min on GPU)
#   make full-flagships  # TSMC + MediaTek
#   make full-top50      # complete TW-50 batch
#   make figures         # regenerate paper Fig 17 from shipped CSVs
#   make figures-us      # regenerate paper Fig 19 (US extension)
#   make tables          # regenerate summary tables
#   make preflight       # environment sanity checks
#   make lint            # ast.parse on all pipeline scripts
#   make clean-smoke     # remove artifacts from smoke test
#   make clean-all       # !! wipes all artifacts/, prices/, cache
#   make help            # print this list
#
# Notes
# -----
#   * All targets assume the `finlab` conda env is already active.
#   * Set STOCK, TRIALS, EPOCHS, BATCH on the command line to override:
#         make full-2330 EPOCHS=60
#         make smoke STOCK=2454
#   * On Windows you can run these targets via `nmake` or WSL; see the
#     `run.ps1` script in the repo root for a native-PowerShell alternative.
# =============================================================================

PY            ?= python
STOCK         ?= 2330
FLAG_STOCKS   ?= 2330,2454
TRIALS        ?= 12
EPOCHS        ?= 80
DFLOOD_EPOCHS ?= 120
DQN_HOURS     ?= 1.5
BATCH         ?= 64
SMOKE_TRIALS  ?= 2
SMOKE_EPOCHS  ?= 3
SMOKE_BATCH   ?= 128
SWEEP_SEEDS   ?= 42,123,456,789,2024
SWEEP_STAGES  ?= 3

.PHONY: help smoke smoke-oof full-2330 full-flagships full-top50 \
		stage1 stage2 stage2-oof stage3 stage3-strict stage4-data stage4-train stage4-backtest \
	prices figures figures-us tables preflight lint monitor-smoke monitor-stage2 \
        seed-sweep \
        rerun-baselines verify-baselines snapshot-baselines \
        verify-prices hash-shipped manifest-check repro \
        clean-smoke clean-artifacts clean-all

help:
	@echo "TW-50 DESQ Makefile — see file header for all targets."
	@echo ""
	@echo "  Quick targets:"
	@echo "    make smoke        # 5-min plumbing check on stock $(STOCK)"
	@echo "    make smoke-oof    # smoke + --des-oof (leakage-free DES fit)"
	@echo "    make full-2330    # production settings for TSMC"
	@echo "    make seed-sweep   # multi-seed Stage 3 sweep -> mean +/- std CSV"
	@echo "    make stage4-data  # build Double-DQN input from Stage 3 output"
	@echo "    make stage4-train # train all five DQN walk-forward folds"
	@echo "    make stage4-backtest # evaluate a promoted Stage 4 checkpoint"
	@echo "    make monitor-smoke # synthetic Appendix-F Level 0-3 smoke"
	@echo "    make monitor-stage2 # immutable Stage 2 snapshot for STOCK"
	@echo "    make repro        # reviewer path: hashes + yfinance check + smoke"
	@echo "    make figures      # regenerate paper Fig 17 from shipped CSVs"
	@echo "    make preflight    # environment sanity checks"
	@echo ""
	@echo "  Override with STOCK=, TRIALS=, EPOCHS=, BATCH= on the command line."

# ---------------------------------------------------------------------------
# Environment sanity
# ---------------------------------------------------------------------------
preflight:
	@$(PY) -c "import tensorflow as tf, keras_tuner, deslib, sklearn, joblib; \
	print('tf', tf.__version__, 'sklearn', sklearn.__version__, 'deslib', deslib.__version__)"
	@$(PY) -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
	@$(PY) -c "from pathlib import Path; n = len(list(Path('features').glob('*.csv'))); print(f'features: {n} CSVs')"
	@$(PY) -c "from pathlib import Path; miss = [s for s in ['$(STOCK)'] if not (Path('prices')/f'{s}.csv').exists()]; print('missing prices:', miss)"

lint:
	@$(PY) -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['tw50_flood.py','tw50_dflood.py','tw50_des.py','fetch_prices.py']]; print('LINT OK')"

# ---------------------------------------------------------------------------
# Individual stages (fine-grained control)
# ---------------------------------------------------------------------------
prices:
	$(PY) fetch_prices.py --stock-ids $(STOCK)

stage1:
	$(PY) tw50_flood.py  --stock-ids $(STOCK) --aspect all --trials $(TRIALS) --epochs $(EPOCHS) --batch-size $(BATCH)

stage2:
	$(PY) tw50_dflood.py --stock-ids $(STOCK) --aspect all --epochs $(DFLOOD_EPOCHS) --batch-size $(BATCH)

# Leakage-free variant of stage 2: emits OOF DES-train predictions.
stage2-oof:
	$(PY) tw50_dflood.py --stock-ids $(STOCK) --aspect all --epochs $(DFLOOD_EPOCHS) --batch-size $(BATCH) --des-oof

stage3:
	$(PY) tw50_des.py    --stock-ids $(STOCK) --no-show

# Aborts if any aspect's DES-train slice was produced in-sample.
stage3-strict:
	$(PY) tw50_des.py    --stock-ids $(STOCK) --no-show --strict-oof

stage4-data:
	cd dqn && $(PY) build_dqn_data.py --stock-ids $(STOCK) --overwrite

stage4-train:
	cd dqn && $(PY) src/train_dqn.py --symbol $(STOCK) --fold all --hours $(DQN_HOURS)

stage4-backtest:
	cd dqn && $(PY) src/backtest.py --symbol $(STOCK) --out backtest_summary.csv

monitor-smoke:
	$(PY) -m monitoring smoke

monitor-stage2:
	$(PY) -m monitoring collect-stage2 --stock-id $(STOCK)

# ---------------------------------------------------------------------------
# Seed sweep (multi-seed backtest -> mean +/- std for paper Table)
# ---------------------------------------------------------------------------
seed-sweep:
	$(PY) scripts/run_seed_sweep.py --stock-ids $(STOCK) --seeds $(SWEEP_SEEDS) --stages $(SWEEP_STAGES)

# ---------------------------------------------------------------------------
# Baseline reproducibility (US market: DSR-Yang, MACE + combined figure)
# ---------------------------------------------------------------------------
# Snapshot shipped CSVs, rerun baselines, then diff shipped vs rerun.
rerun-baselines:
	bash us/baselines/run_all_baselines.sh

# Diff-only pass: assumes _shipped_snapshot/ exists and rerun outputs are in place.
verify-baselines:
	$(PY) us/baselines/verify_baselines.py

# Just refresh the shipped snapshot (no rerun, no verify).
snapshot-baselines:
	bash us/baselines/run_all_baselines.sh --verify-only --force-snapshot || true

# ---------------------------------------------------------------------------
# Reviewer reproducibility kit (public-data checks, no proprietary access)
# ---------------------------------------------------------------------------
verify-prices:
	$(PY) reproducibility/verify_public_prices.py --stock-ids $(STOCK)

hash-shipped:
	$(PY) reproducibility/hash_shipped.py

# Enforce shipped-data integrity (CI + local pre-push check).
manifest-check:
	$(PY) reproducibility/check_manifest.py

# One-shot reviewer path: fingerprints -> public-price cross-check -> smoke.
repro: hash-shipped verify-prices smoke-oof

# ---------------------------------------------------------------------------
# End-to-end recipes
# ---------------------------------------------------------------------------
# 5-minute CPU-friendly plumbing test; NOT a headline reproduction.
smoke:
	$(PY) fetch_prices.py --stock-ids $(STOCK)
	$(PY) tw50_flood.py   --stock-ids $(STOCK) --aspect all --trials $(SMOKE_TRIALS) --epochs $(SMOKE_EPOCHS) --batch-size $(SMOKE_BATCH)
	$(PY) tw50_dflood.py  --stock-ids $(STOCK) --aspect all --epochs 5 --batch-size $(SMOKE_BATCH)
	$(PY) tw50_des.py     --stock-ids $(STOCK) --no-show
	@echo "Smoke test complete. See artifacts/des/backtest/summary.csv"

# Same as smoke but with --des-oof so reviewers can verify the OOF path.
smoke-oof:
	$(PY) fetch_prices.py --stock-ids $(STOCK)
	$(PY) tw50_flood.py   --stock-ids $(STOCK) --aspect all --trials $(SMOKE_TRIALS) --epochs $(SMOKE_EPOCHS) --batch-size $(SMOKE_BATCH)
	$(PY) tw50_dflood.py  --stock-ids $(STOCK) --aspect all --epochs 5 --batch-size $(SMOKE_BATCH) --des-oof
	$(PY) tw50_des.py     --stock-ids $(STOCK) --no-show --strict-oof
	@echo "OOF smoke test complete."

full-2330:
	$(PY) fetch_prices.py --stock-ids 2330
	$(PY) tw50_flood.py   --stock-ids 2330 --aspect all --trials $(TRIALS) --epochs $(EPOCHS)
	$(PY) tw50_dflood.py  --stock-ids 2330 --aspect all --epochs $(DFLOOD_EPOCHS) --des-oof
	$(PY) tw50_des.py     --stock-ids 2330 --no-show --strict-oof

full-flagships:
	$(PY) fetch_prices.py --stock-ids $(FLAG_STOCKS)
	$(PY) tw50_flood.py   --stock-ids $(FLAG_STOCKS) --aspect all --trials $(TRIALS) --epochs $(EPOCHS)
	$(PY) tw50_dflood.py  --stock-ids $(FLAG_STOCKS) --aspect all --epochs $(DFLOOD_EPOCHS) --des-oof
	$(PY) tw50_des.py     --stock-ids $(FLAG_STOCKS) --no-show --strict-oof

full-top50:
	$(PY) fetch_prices.py --top50 --sleep 0.4
	$(PY) tw50_flood.py   --top50 --aspect all --trials $(TRIALS) --epochs $(EPOCHS)
	$(PY) tw50_dflood.py  --top50 --aspect all --epochs $(DFLOOD_EPOCHS) --des-oof
	$(PY) tw50_des.py     --top50 --no-show --strict-oof

# ---------------------------------------------------------------------------
# Paper figures / tables regenerated from shipped CSVs (no training needed).
# ---------------------------------------------------------------------------
figures:
	$(PY) evaluation/render_figure_backtest.py

figures-us:
	$(PY) us/baselines/combined/combined_comparison.py

tables:
	$(PY) evaluation/paper/generate_tables.py

tables-check:
	$(PY) -m unittest discover -s evaluation/paper/tests -v

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean-smoke:
	@echo "Removing smoke-test artifacts for STOCK=$(STOCK) ..."
	rm -f artifacts/des/backtest/$(STOCK)_*.png artifacts/des/backtest/$(STOCK)_*.csv
	rm -f artifacts/des/pred/*_$(STOCK).csv
	rm -f artifacts/des/models/*_$(STOCK).pkl
	rm -f artifacts/dflood/pred/$(STOCK)_*.csv
	rm -f artifacts/dflood/models/$(STOCK)_*.keras artifacts/dflood/models/$(STOCK)_*_dflood_history.json

clean-artifacts:
	rm -rf artifacts/

# Nuclear option: wipe everything the pipeline generated.
clean-all: clean-artifacts
	rm -rf prices/
	@echo "All generated artifacts removed. `features/` is retained."
