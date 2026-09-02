from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.select_median_agents import select_median_agents  # noqa: E402


def candidates(stock_id: str = "2330") -> pd.DataFrame:
    return pd.DataFrame({
        "stock_id": [stock_id] * 9,
        "seed": [9, 1, 8, 2, 7, 3, 6, 4, 5],
        "validation_return": [90, 10, 80, 20, 70, 30, 60, 40, 50],
        "checkpoint_path": [f"seed_{seed}.data" for seed in range(9)],
    })


class MedianAgentSelectionTests(unittest.TestCase):
    def test_selects_fifth_validation_return_per_stock(self) -> None:
        frame = pd.concat([candidates("2454"), candidates("2330")], ignore_index=True)
        selected = select_median_agents(frame)
        self.assertEqual(selected["stock_id"].tolist(), ["2330", "2454"])
        self.assertEqual(selected["validation_return"].tolist(), [50, 50])
        self.assertEqual(selected["rank_of_nine"].tolist(), [5, 5])

    def test_seed_breaks_validation_return_ties_deterministically(self) -> None:
        frame = candidates()
        frame["validation_return"] = 1.0
        selected = select_median_agents(frame)
        self.assertEqual(int(selected.iloc[0]["seed"]), 5)

    def test_rejects_incomplete_or_duplicate_seed_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 9 unique seeds"):
            select_median_agents(candidates().iloc[:-1])
        frame = candidates()
        frame.loc[8, "seed"] = frame.loc[7, "seed"]
        with self.assertRaisesRegex(ValueError, "exactly 9 unique seeds"):
            select_median_agents(frame)


if __name__ == "__main__":
    unittest.main()
