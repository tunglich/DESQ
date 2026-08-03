#!/bin/bash
# =============================================================================
# End-to-end baseline reproducibility runner.
#
# 1. Snapshots current (shipped) baseline outputs to us/baselines/_shipped_snapshot/
#    (skipped if the snapshot already exists; use --force-snapshot to override).
# 2. Reruns the three baseline scripts (DSR-Yang, MACE) over each universe
#    (dow30, sp100, ndx100), overwriting the shipped output directories.
# 3. Reruns the combined comparison figure/CSV generator.
# 4. Runs verify_baselines.py to prove the fresh outputs match the shipped
#    snapshot to within --tol (default 1e-6).
#
# Requirements
# ------------
# * US price / universe data available at the paths hard-coded in the baseline
#   scripts (typically d:\US_stock on Windows). This runner is IDEMPOTENT with
#   respect to shipped CSVs: a failed rerun leaves _shipped_snapshot/ intact so
#   you can restore via:
#       rm -rf us/baselines/dsr_yang/backtest_* us/baselines/mi_abbade/backtest_*
#       cp -r us/baselines/_shipped_snapshot/* us/baselines/
#
# Usage
# -----
#   bash us/baselines/run_all_baselines.sh
#   bash us/baselines/run_all_baselines.sh --force-snapshot   # reset snapshot
#   bash us/baselines/run_all_baselines.sh --verify-only      # skip rerun
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/us/baselines"
SNAP_DIR="$BASELINES_DIR/_shipped_snapshot"
PY="${PY:-python}"

FORCE_SNAP=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force-snapshot) FORCE_SNAP=1 ;;
    --verify-only)    VERIFY_ONLY=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# 1. Snapshot
# -----------------------------------------------------------------------------
if [ "$FORCE_SNAP" = 1 ] && [ -d "$SNAP_DIR" ]; then
  echo "[SNAP] --force-snapshot given; removing existing $SNAP_DIR"
  rm -rf "$SNAP_DIR"
fi

if [ ! -d "$SNAP_DIR" ]; then
  echo "[SNAP] creating $SNAP_DIR from current baseline outputs ..."
  mkdir -p "$SNAP_DIR"
  # Copy every CSV under us/baselines/, preserving structure.
  ( cd "$BASELINES_DIR" && \
    find . -type f -name '*.csv' -not -path './_shipped_snapshot/*' \
      -exec cp --parents {} "$SNAP_DIR" \; ) || true
  n=$(find "$SNAP_DIR" -type f -name '*.csv' | wc -l)
  echo "[SNAP] $n CSVs snapshotted -> $SNAP_DIR"
else
  echo "[SNAP] $SNAP_DIR already exists; keeping as-is (use --force-snapshot to reset)"
fi

# -----------------------------------------------------------------------------
# 2 + 3. Rerun baselines and combined comparison
# -----------------------------------------------------------------------------
if [ "$VERIFY_ONLY" = 0 ]; then
  for U in dow30 sp100 ndx100; do
    echo ""
    echo "=== DSR-Yang / $U ==="
    $PY "$BASELINES_DIR/dsr_yang/dsr_backtest.py" "$U"
    echo ""
    echo "=== MACE / $U ==="
    $PY "$BASELINES_DIR/mi_abbade/mace_backtest.py" "$U"
  done

  echo ""
  echo "=== combined_comparison ==="
  $PY "$BASELINES_DIR/combined/combined_comparison.py"
else
  echo "[SKIP] --verify-only given; not rerunning baselines."
fi

# -----------------------------------------------------------------------------
# 4. Verify
# -----------------------------------------------------------------------------
echo ""
echo "=== verify_baselines ==="
$PY "$BASELINES_DIR/verify_baselines.py"
