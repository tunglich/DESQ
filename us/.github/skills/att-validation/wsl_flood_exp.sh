#!/usr/bin/env bash
# Flooding ablation experiment on 2330 (none / static / dynamic) using
# ATT+Dflooding_floodexp.py, then aggregate comparison plots.
#
# Usage:  bash wsl_flood_exp.sh
# Logs to logs/flood_exp_<timestamp>.log
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

# --- experiment scope & budget ---
export VALIDATION_MODE='rolling'           # walk-forward rolling (matches search)
export FEATURE_PREPROCESS='1'              # no interactive prompt
export STOCK_IDS='2330'
export HYPER_ROOT='D:/hyperbayes_test_f1'         # F1 (val_fbeta_score) re-searched hyperparameters for 2330
# MODEL_TYPES: comma-separated subset of the 6 aspects; default = all six. Override at runtime.
MODEL_TYPES="${MODEL_TYPES:-fundamental,trade,moment,sentiment,tech_trend,macro}"
export MODEL_TYPES
export NUM_REPEATS='12'
export MAX_EPOCHS='300'
export DISABLE_EARLY_STOPPING='1'
unset FAST_DEBUG

# FLOOD_MODES: comma-separated subset of {none,static,dynamic}; default = all three.
FLOOD_MODES="${FLOOD_MODES:-none,static,dynamic}"
MODES_LIST="${FLOOD_MODES//,/ }"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/flood_exp_${TS}.log"
echo "[start] $(date)  log=${LOG}"
echo "  ENV=${ENV_NAME}  STOCK_IDS=${STOCK_IDS}  MODEL_TYPES=${MODEL_TYPES}"
echo "  NUM_REPEATS=${NUM_REPEATS}  MAX_EPOCHS=${MAX_EPOCHS}  VALIDATION_MODE=${VALIDATION_MODE}"
echo "  FLOOD_MODES=${FLOOD_MODES}"

{
  for m in ${MODES_LIST}; do
    echo "==================== FLOOD_MODE=${m} :: $(date) ===================="
    FLOOD_MODE="${m}" "${PYBIN}" "ATT+Dflooding_floodexp.py"
  done
  echo "==================== aggregate comparison plots :: $(date) ===================="
  "${PYBIN}" plot_flood_compare.py
} > "${LOG}" 2>&1

echo "[done] $(date)  exit=$?  log=${LOG}"
