#!/usr/bin/env bash
# Re-run ATT+Flood hyperparameter search for 2330 with the F1 (val_fbeta_score) objective,
# reduced budget (half trials: stage1=6, stage2=12), output to a SEPARATE
# hyperbayes folder so the existing D:/hyperbayes_test is not overwritten.
#
# Usage:  bash wsl_flood_hpo.sh
# Logs to logs/flood_hpo_<timestamp>.log
set -e
cd /mnt/d/US_stock

ENV_NAME="${ENV_NAME:-finlabUS}"
SP="/home/tungl/miniconda3/envs/${ENV_NAME}/lib/python3.11/site-packages"
PYBIN="/home/tungl/miniconda3/envs/${ENV_NAME}/bin/python"

# --- GPU library paths (TF 2.21 wheel RPATH is not effective on WSL) ---
NV_LIBS=""
for d in "$SP"/nvidia/*/lib; do
  [ -d "$d" ] && NV_LIBS="${NV_LIBS}:${d}"
done
export LD_LIBRARY_PATH="${NV_LIBS#:}:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SP}/nvidia/cuda_nvcc"
export CUDA_VISIBLE_DEVICES='0'

# --- search scope & budget (PR-AUC objective, half trials) ---
export VALIDATION_MODE='rolling'                 # walk-forward rolling
export FEATURE_PREPROCESS='1'                     # no interactive prompt
export STOCK_IDS='2330'
export MODEL_TYPES='fundamental,trade,moment,sentiment,tech_trend,macro'
export HYPERBAYES_ATT_DIR='D:/hyperbayes_test_f1'    # separate output, keeps hyperbayes_test intact
export STAGE1_MAX_TRIALS='6'                       # half of 12
export STAGE2_MAX_TRIALS='12'                      # half of 24
# epochs left at script defaults (80/120); setting trial env disables auto-reduce
export AUTO_REDUCE_SEARCH_FOR_WF='0'
unset FAST_DEBUG

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/flood_hpo_${TS}.log"
echo "[start] $(date)  log=${LOG}"
echo "  ENV=${ENV_NAME}  STOCK_IDS=${STOCK_IDS}  MODEL_TYPES=${MODEL_TYPES}"
echo "  OBJECTIVE=val_fbeta_score  STAGE1=${STAGE1_MAX_TRIALS}  STAGE2=${STAGE2_MAX_TRIALS}"
echo "  OUTPUT=${HYPERBAYES_ATT_DIR}"

"${PYBIN}" "ATT+Flood_floodexp.py" > "${LOG}" 2>&1

echo "[done] $(date)  exit=$?  log=${LOG}"
