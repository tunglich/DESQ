"""Baseline reproducibility verifier.

Compares two directory trees of CSVs (typically a `_shipped_snapshot/` produced
by `run_all_baselines.sh` versus the freshly re-run outputs still in the
working tree) and reports the maximum absolute numeric difference per file.

Exits 0 if every matched CSV agrees within --tol, 1 otherwise. Non-numeric
columns are compared by strict equality.

Typical usage
-------------
    # After run_all_baselines.sh has snapshot + re-run:
    python us/baselines/verify_baselines.py

    # Explicit dirs (for CI or ad-hoc checks):
    python us/baselines/verify_baselines.py \
        --shipped-dir us/baselines/_shipped_snapshot \
        --rerun-dir   us/baselines

Notes
-----
- Files present in shipped but missing in rerun are FAIL. Files in rerun but
  not shipped are ignored (may be intermediate artifacts).
- The comparison focuses on peer-method `metrics.csv`, `predictions.csv`,
  `selections.csv`, `equity_*.csv`, and current Table 6 `*_comparison.csv`
  inputs. Override with `--pattern`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / 'us' / 'baselines'
DEFAULT_SNAPSHOT_DIR = BASELINES_DIR / '_shipped_snapshot'

DEFAULT_PATTERNS = (
    'metrics.csv',
    'predictions.csv',
    'selections.csv',
    'equity_*.csv',
    '*_comparison.csv',
)


def _find_csvs(root: Path, patterns: Sequence[str]) -> list[Path]:
    found: set[Path] = set()
    for pat in patterns:
        found.update(root.rglob(pat))
    return sorted(found)


def _max_numeric_diff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, str]:
    """Return (max_abs_diff, worst_col). NaN treated as equal to NaN."""
    if a.shape != b.shape:
        return float('inf'), f'shape mismatch: {a.shape} vs {b.shape}'
    if list(a.columns) != list(b.columns):
        return float('inf'), f'column mismatch: {list(a.columns)} vs {list(b.columns)}'

    worst = 0.0
    worst_col = ''
    for col in a.columns:
        ca, cb = a[col], b[col]
        if pd.api.types.is_numeric_dtype(ca) and pd.api.types.is_numeric_dtype(cb):
            diff = (ca.astype(float) - cb.astype(float)).abs()
            mask = ~(ca.isna() & cb.isna())
            m = float(diff[mask].max()) if mask.any() else 0.0
        else:
            neq = (ca.astype(str) != cb.astype(str)) & ~(ca.isna() & cb.isna())
            m = float('inf') if neq.any() else 0.0
        if m > worst:
            worst = m
            worst_col = col
    return worst, worst_col


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--shipped-dir', default=str(DEFAULT_SNAPSHOT_DIR),
                   help=f'snapshot of shipped CSVs (default: {DEFAULT_SNAPSHOT_DIR})')
    p.add_argument('--rerun-dir', default=str(BASELINES_DIR),
                   help=f'freshly-run baseline outputs (default: {BASELINES_DIR})')
    p.add_argument('--tol', type=float, default=1e-6,
                   help='max abs numeric diff considered PASS (default: 1e-6)')
    p.add_argument('--pattern', action='append',
                   help=f'glob(s) to compare (repeatable). Default: {DEFAULT_PATTERNS}')
    args = p.parse_args(argv)

    shipped_dir = Path(args.shipped_dir).resolve()
    rerun_dir = Path(args.rerun_dir).resolve()
    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS

    if not shipped_dir.is_dir():
        print(f'[ERR] shipped-dir not found: {shipped_dir}')
        print('      Run us/baselines/run_all_baselines.sh first to create it.')
        return 2

    shipped_csvs = _find_csvs(shipped_dir, patterns)
    if not shipped_csvs:
        print(f'[ERR] no CSVs matched under {shipped_dir}')
        return 2

    print(f'[PLAN] comparing {len(shipped_csvs)} CSVs, tol={args.tol}')
    print(f'       shipped: {shipped_dir}')
    print(f'       rerun:   {rerun_dir}')

    n_pass = n_fail = n_missing = 0
    fails: list[tuple[str, float, str]] = []
    for sp in shipped_csvs:
        rel = sp.relative_to(shipped_dir)
        rp = rerun_dir / rel
        if not rp.exists():
            print(f'  MISSING  {rel}')
            n_missing += 1
            continue
        try:
            a = pd.read_csv(sp)
            b = pd.read_csv(rp)
            diff, col = _max_numeric_diff(a, b)
        except Exception as err:
            print(f'  ERROR    {rel}: {err}')
            n_fail += 1
            fails.append((str(rel), float('nan'), str(err)))
            continue
        if diff <= args.tol:
            print(f'  PASS     {rel}  (max_diff={diff:.2e})')
            n_pass += 1
        else:
            print(f'  FAIL     {rel}  (max_diff={diff:.6g}, worst_col={col!r})')
            n_fail += 1
            fails.append((str(rel), diff, col))

    print(f'\n[SUMMARY] pass={n_pass}, fail={n_fail}, missing={n_missing}, '
          f'total={len(shipped_csvs)}')
    if fails:
        print('\n[FAILS]')
        for rel, diff, col in fails:
            print(f'  {rel}  diff={diff}  col={col!r}')
    return 0 if (n_fail == 0 and n_missing == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
