"""Seed-sweep driver for the TW-50 DESQ pipeline.

Runs the pipeline stages under a list of RNG seeds and aggregates per-stock
backtest metrics with mean/std across seeds. Produces the evidence CSV that
IEEE Access §IV.H reviewers expect to see for "mean +/- std across seeds".

Typical usage
-------------
    # Stage 3 only (fast; ~1-2 min per seed; reuses cached Stage 1/2 artifacts)
    python scripts/run_seed_sweep.py --stock-ids 2330,2454 \
        --seeds 42,123,456,789,2024 --stages 3

    # Full retrain per seed (slow; reruns Stages 1+2+3)
    python scripts/run_seed_sweep.py --stock-ids 2330 \
        --seeds 42,123,456 --stages 123 --trials 12 --epochs 60 --dflood-epochs 120

Output
------
    artifacts/seed_sweep/per_run.csv     # one row per (stock_id, seed)
    artifacts/seed_sweep/aggregate.csv   # one row per stock_id: mean +/- std
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DES_SUMMARY = REPO_ROOT / 'artifacts' / 'des' / 'backtest' / 'summary.csv'
DES_MODEL_DIR = REPO_ROOT / 'artifacts' / 'des' / 'models'
OUT_DIR = REPO_ROOT / 'artifacts' / 'seed_sweep'

METRIC_COLS = ('n_test_days', 'acc_buy', 'acc_sell',
               'total_ret_model', 'total_ret_stock', 'excess_ret')


def _run(argv: list[str]) -> None:
    print(f'  $ {" ".join(argv)}', flush=True)
    subprocess.run(argv, check=True, cwd=str(REPO_ROOT))


def _clear_des_cache(stock_ids: list[str]) -> None:
    for sid in stock_ids:
        for p in (DES_MODEL_DIR / f'DES_{sid}.pkl', DES_MODEL_DIR / f'RF_{sid}.pkl'):
            if p.exists():
                p.unlink()


def _run_one_seed(stock_ids: list[str], seed: int, args: argparse.Namespace) -> pd.DataFrame:
    py = sys.executable
    stock_arg = ','.join(stock_ids)

    if '1' in args.stages:
        _run([py, 'tw50_flood.py', '--stock-ids', stock_arg,
              '--aspect', 'all', '--trials', str(args.trials),
              '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
              '--seed', str(seed)])

    if '2' in args.stages:
        cmd = [py, 'tw50_dflood.py', '--stock-ids', stock_arg,
               '--aspect', 'all', '--epochs', str(args.dflood_epochs),
               '--batch-size', str(args.batch_size), '--seed', str(seed)]
        if args.des_oof:
            cmd.append('--des-oof')
        _run(cmd)

    # Stage 3 is always executed; strict-oof forces RF/KNORAE refit per seed.
    _clear_des_cache(stock_ids)
    cmd = [py, 'tw50_des.py', '--stock-ids', stock_arg, '--no-show', '--seed', str(seed)]
    if args.strict_oof:
        cmd.append('--strict-oof')
    _run(cmd)

    if not DES_SUMMARY.exists():
        raise RuntimeError(f'expected {DES_SUMMARY} after Stage 3; not found.')
    df = pd.read_csv(DES_SUMMARY)
    df.insert(1, 'seed', seed)
    return df


def _aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    grouped = per_run.groupby('stock_id')
    out_rows = []
    for sid, g in grouped:
        row = {'stock_id': sid, 'n_seeds': len(g)}
        for m in METRIC_COLS:
            if m in g.columns:
                row[f'{m}_mean'] = float(g[m].mean())
                row[f'{m}_std'] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stock-ids', required=True, help='comma-separated list, e.g. 2330,2454')
    p.add_argument('--seeds', default='42,123,456,789,2024',
                   help='comma-separated integer seeds (default: 42,123,456,789,2024)')
    p.add_argument('--stages', default='3',
                   help="which stages to rerun per seed. '3' = Stage 3 only (fast); "
                        "'23' = Stages 2+3; '123' = full retrain. Default: '3'.")
    p.add_argument('--trials', type=int, default=12)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--dflood-epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--des-oof', action='store_true', default=True,
                   help='(default on) emit OOF DES-train preds when rerunning Stage 2.')
    p.add_argument('--no-des-oof', dest='des_oof', action='store_false',
                   help='disable --des-oof when rerunning Stage 2.')
    p.add_argument('--strict-oof', action='store_true', default=True,
                   help='(default on) pass --strict-oof to Stage 3 (leakage guard).')
    p.add_argument('--no-strict-oof', dest='strict_oof', action='store_false',
                   help='disable --strict-oof on Stage 3.')
    p.add_argument('--out-dir', default=str(OUT_DIR),
                   help=f'output directory (default: {OUT_DIR})')
    args = p.parse_args(argv)

    stock_ids = [s.strip() for s in args.stock_ids.split(',') if s.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[PLAN] stock_ids={stock_ids}, seeds={seeds}, stages={args.stages}, '
          f'des_oof={args.des_oof}, strict_oof={args.strict_oof}, out={out_dir}')

    all_rows = []
    for seed in seeds:
        print(f'\n===== seed={seed} =====')
        try:
            df = _run_one_seed(stock_ids, seed, args)
            all_rows.append(df)
        except subprocess.CalledProcessError as exc:
            print(f'[FAIL] seed={seed}: {exc}')
            continue

    if not all_rows:
        print('[ERR] no successful runs; aborting.')
        return 1

    per_run = pd.concat(all_rows, ignore_index=True)
    per_run_path = out_dir / 'per_run.csv'
    per_run.to_csv(per_run_path, index=False)
    print(f'\n[PER-RUN] {len(per_run)} rows -> {per_run_path}')

    agg = _aggregate(per_run)
    agg_path = out_dir / 'aggregate.csv'
    agg.to_csv(agg_path, index=False)
    print(f'[AGG] {len(agg)} rows -> {agg_path}')
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(agg.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
