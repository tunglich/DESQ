"""5-fold contiguous walk-forward CV for TW-DQN training.

Splits ``2005-01-03 ~ TEST_START`` history into 5 equal-length contiguous
chunks. For fold ``k``, ``val = chunk_k`` and ``train = concat(chunks j != k)``.
Test window (``date >= TEST_START``) is held out entirely.

Per-symbol adaptive start: the first row where OHLC are all > 0 is used as the
effective start (some symbols IPO after 2005). All non-finite / zero-open rows
in the retained span are also dropped so ``prices_to_relative`` doesn't divide
by zero.

CLI:
    python src/walk_forward.py --symbol 2330 --window 75 --dry-run
    python src/walk_forward.py --symbol 2330 --window 75 --fold 0 --out saves/2330_all_75/fold_0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
TEST_START = pd.Timestamp("2024-01-01")
N_FOLDS = 5


class FoldPaths(NamedTuple):
    fold_id: int
    train_csv: Path
    val_csv: Path
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]


def load_prefiltered(csv_path: Path, test_start: pd.Timestamp = TEST_START) -> pd.DataFrame:
    """Load a public-format CSV and trim to the pre-test walk-forward span."""
    df = pd.read_csv(csv_path)
    if "<DATE>" not in df.columns:
        raise ValueError(f"{csv_path}: expected <DATE> column, got {list(df.columns)[:6]}")
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    for col in ("<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = (df["<OPEN>"] > 0) & (df["<HIGH>"] > 0) & (df["<LOW>"] > 0) & (df["<CLOSE>"] > 0)
    ok &= np.isfinite(df["<OPEN>"]) & np.isfinite(df["<CLOSE>"])
    if not ok.any():
        raise RuntimeError(f"{csv_path}: no rows with all OHLC > 0")
    first_ok = int(ok.idxmax())
    df = df.loc[first_ok:].copy()
    df = df.loc[ok.loc[first_ok:]].copy()

    df = df[df["<DATE>"] < test_start].copy()
    if df.empty:
        raise RuntimeError(f"{csv_path}: no rows before test_start={test_start.date()}")
    return df.sort_values("<DATE>").reset_index(drop=True)


def split_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return [(train_df, val_df) for each fold]."""
    n = len(df)
    if n < n_folds * 20:
        raise RuntimeError(f"Only {n} rows, too few for {n_folds}-fold CV")
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds = []
    for k in range(n_folds):
        lo, hi = int(edges[k]), int(edges[k + 1])
        val = df.iloc[lo:hi].copy()
        train = pd.concat([df.iloc[:lo], df.iloc[hi:]], axis=0).copy()
        folds.append((train, val))
    return folds


def write_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, fold_id: int) -> FoldPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / f"fold_{fold_id}_train.csv"
    val_csv = out_dir / f"fold_{fold_id}_val.csv"
    # Format dates as YYYY-MM-DD to stay compatible with lib.data.read_csv (skips DATE, keeps DES/OHLC).
    for df, dst in ((train_df, train_csv), (val_df, val_csv)):
        out = df.copy()
        out["<DATE>"] = out["<DATE>"].dt.strftime("%Y-%m-%d")
        out.to_csv(dst, index=False, lineterminator="\n")
    return FoldPaths(
        fold_id=fold_id,
        train_csv=train_csv,
        val_csv=val_csv,
        train_dates=(train_df["<DATE>"].min(), train_df["<DATE>"].max()),
        val_dates=(val_df["<DATE>"].min(), val_df["<DATE>"].max()),
    )


def build_all_folds(csv_path: Path, out_dir: Path, n_folds: int = N_FOLDS) -> list[FoldPaths]:
    df = load_prefiltered(csv_path)
    folds = split_folds(df, n_folds=n_folds)
    return [write_fold(tr, va, out_dir, k) for k, (tr, va) in enumerate(folds)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", required=True, help="e.g. 2330")
    ap.add_argument("--window", required=True, type=int, choices=(55, 60, 65, 75))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory to write fold CSVs (default: saves/{sym}_all_{window}/)")
    ap.add_argument("--fold", type=int, default=None,
                    help="Only export a specific fold (0..N-1); default all")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--dry-run", action="store_true", help="Print fold ranges without writing files")
    args = ap.parse_args()

    csv_path = args.data_dir / f"{args.symbol}_all_{args.window}.csv"
    if not csv_path.is_file():
        raise SystemExit(f"missing: {csv_path}")
    out_dir = args.out or (REPO_ROOT / "saves" / f"{args.symbol}_all_{args.window}")

    df = load_prefiltered(csv_path)
    print(f"{csv_path.name}: {len(df)} pre-test rows "
          f"({df['<DATE>'].min().date()} ~ {df['<DATE>'].max().date()})")
    folds = split_folds(df, n_folds=args.n_folds)
    for k, (tr, va) in enumerate(folds):
        vstart, vend = va["<DATE>"].min().date(), va["<DATE>"].max().date()
        print(f"  fold {k}: val {vstart} ~ {vend} ({len(va)} rows), train {len(tr)} rows")

    if args.dry_run:
        return 0

    if args.fold is None:
        wrote = build_all_folds(csv_path, out_dir, n_folds=args.n_folds)
    else:
        tr, va = folds[args.fold]
        wrote = [write_fold(tr, va, out_dir, args.fold)]

    print(f"\nWrote {len(wrote)} fold(s) to {out_dir}")
    for fp in wrote:
        print(f"  fold {fp.fold_id}: train {fp.train_csv.name}, val {fp.val_csv.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
