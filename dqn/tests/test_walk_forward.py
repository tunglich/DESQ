from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DQN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DQN_ROOT))

from src.walk_forward import REFERENCE_WINDOWS, split_folds  # noqa: E402


class ReferenceWalkForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.bdate_range("2005-01-03", "2023-12-29")
        cls.frame = pd.DataFrame({
            "<DATE>": dates,
            "<OPEN>": np.ones(len(dates)),
            "<HIGH>": np.ones(len(dates)),
            "<LOW>": np.ones(len(dates)),
            "<CLOSE>": np.ones(len(dates)),
        })

    def test_uses_the_five_reference_windows(self):
        folds = split_folds(self.frame)
        self.assertEqual(len(folds), 5)
        for (train, validation), (start, end) in zip(folds, REFERENCE_WINDOWS):
            self.assertEqual(train["<DATE>"].min().year, start.year)
            self.assertEqual(validation["<DATE>"].max().year, end.year)

    def test_is_rolling_and_observes_label_plus_purge_gap(self):
        folds = split_folds(self.frame)
        for train, validation in folds:
            self.assertLess(train["<DATE>"].max(), validation["<DATE>"].min())
            window_dates = self.frame.loc[
                self.frame["<DATE>"].between(train["<DATE>"].min(), validation["<DATE>"].max()),
                "<DATE>",
            ].reset_index(drop=True)
            train_pos = int(window_dates[window_dates == train["<DATE>"].max()].index[0])
            validation_pos = int(window_dates[window_dates == validation["<DATE>"].min()].index[0])
            self.assertGreaterEqual(validation_pos - train_pos, 50)


if __name__ == "__main__":
    unittest.main()
