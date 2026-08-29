from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from monitoring.adapters import adapt_stage2_prediction
from monitoring.schemas import MonitoringSnapshot


class AdapterTest(unittest.TestCase):
    def test_stage2_uses_only_mature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "2330_macro.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Date", "y_true_20", "prob_down", "prob_up", "source"])
                start = date(2026, 1, 1)
                for index in range(80):
                    probability = 0.8 if index % 2 else 0.2
                    writer.writerow([(start + timedelta(days=index)).isoformat(), index % 2,
                                     1.0 - probability, probability, "oof"])
            result = adapt_stage2_prediction(path, "2330", "macro", date(2026, 12, 31))
            self.assertEqual(result.status, "valid")
            self.assertTrue(all(metric.sample_count == 60 for metric in result.metrics))
            self.assertEqual(result.observation_start, "2026-01-01")
            self.assertEqual(result.observation_end, "2026-03-01")
            self.assertEqual(result.mature_label_cutoff, "2026-03-01")

    def test_snapshot_write_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = MonitoringSnapshot("2026-08-29", "2026-05-01", "2026-08-01",
                                          "2026-08-01", "contract", "policy", "evaluator")
            first = snapshot.write(Path(temporary))
            second = snapshot.write(Path(temporary))
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()