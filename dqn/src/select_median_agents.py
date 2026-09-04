"""Select the median-validation DDQN agent independently for each stock.

The input manifest must contain exactly nine unique seeds per stock and the
columns ``stock_id``, ``seed``, ``validation_return``, and ``checkpoint_path``.
Rows are ordered by validation return and then seed; the fifth row is selected.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("stock_id", "seed", "validation_return", "checkpoint_path")
REFERENCE_SEED_COUNT = 9


def select_median_agents(candidates: pd.DataFrame,
                         manifest_dir: Path | None = None) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in candidates.columns]
    if missing:
        raise ValueError(f"candidate manifest missing columns: {missing}")

    frame = candidates.loc[:, REQUIRED_COLUMNS].copy()
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["validation_return"] = pd.to_numeric(
        frame["validation_return"], errors="raise")
    if not np.isfinite(frame["validation_return"]).all():
        raise ValueError("validation_return must be finite")

    selected = []
    for stock_id, group in frame.groupby("stock_id", sort=True):
        if len(group) != REFERENCE_SEED_COUNT or group["seed"].nunique() != REFERENCE_SEED_COUNT:
            raise ValueError(
                f"{stock_id}: expected exactly {REFERENCE_SEED_COUNT} unique seeds, "
                f"got {len(group)} rows/{group['seed'].nunique()} unique")
        ordered = group.sort_values(["validation_return", "seed"], kind="stable")
        row = ordered.iloc[REFERENCE_SEED_COUNT // 2].copy()
        checkpoint = Path(str(row["checkpoint_path"]))
        if manifest_dir is not None and not checkpoint.is_absolute():
            checkpoint = manifest_dir / checkpoint
        if manifest_dir is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        row["checkpoint_path"] = checkpoint.as_posix()
        row["rank_of_nine"] = REFERENCE_SEED_COUNT // 2 + 1
        row["candidate_count"] = REFERENCE_SEED_COUNT
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-check", action="store_true",
                        help="Validate selection metadata without requiring checkpoint files")
    args = parser.parse_args(argv)

    candidates = pd.read_csv(args.manifest, dtype={"stock_id": str})
    manifest_dir = None if args.skip_checkpoint_check else args.manifest.resolve().parent
    selected = select_median_agents(candidates, manifest_dir=manifest_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, lineterminator="\n")
    print(f"Selected {len(selected)} stock-level median agents -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
