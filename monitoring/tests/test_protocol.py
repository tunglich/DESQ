from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from monitoring.config import load_contract, load_policy
from monitoring.protocol import evaluate_batches, load_diagnostic_batch


def window(stock_id: str) -> dict:
    return {
        "stock_id": stock_id,
        "sample_count": 40,
        "precision_gap": 0.09,
        "return_gap": 0.01,
        "sharpe": 0.42,
        "information_ratio": 0.18,
        "disagreement": 0.3,
        "training_disagreement_q90": 0.2,
        "flooding_upper_fraction": 0.1,
        "max_psi": 0.3,
        "affected_groups": ["trade"],
    }


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.contract_hash = load_contract()
        cls.policy, cls.policy_hash = load_policy()

    def write_batch(self, root: Path, name: str, start: str, end: str, anchor_index: int,
                    contract_hash: str | None = None, **extra: object) -> Path:
        path = root / name
        payload = {
            "schema_version": "1.0",
            "monitoring_anchor_index": anchor_index,
            "observation_start": start,
            "observation_end": end,
            "paper_contract_hash": contract_hash or self.contract_hash,
            "policy_hash": self.policy_hash,
            "windows": [window("2330")],
        }
        payload.update(extra)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_adjacent_batches_produce_immutable_dry_run_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = load_diagnostic_batch(
                self.write_batch(root, "previous.json", "2026-01-01", "2026-03-31", 100)
            )
            current = load_diagnostic_batch(
                self.write_batch(root, "current.json", "2026-04-01", "2026-06-30", 160)
            )
            evaluation = evaluate_batches(current, previous, self.contract, self.contract_hash,
                                          self.policy_hash, self.policy)
            self.assertEqual(evaluation.decision.level, 1)
            self.assertTrue(evaluation.candidate_plan.dry_run)
            self.assertFalse(evaluation.candidate_plan.executable)
            first = evaluation.write(root / "out")
            self.assertEqual(first, evaluation.write(root / "out"))
            memory_path = first.with_name("research_memory.json")
            memory = json.loads(memory_path.read_text(encoding="ascii"))
            self.assertEqual(memory["source_equation"], 43)
            self.assertEqual(memory["promotion_status"], "not_evaluated")
            self.assertIsNone(memory["promoted"])
            self.assertEqual(memory["candidate_parameters"], ["des_threshold"])
            self.assertIsNone(memory["metric_deltas"][0]["drawdown"])

    def test_hash_and_window_order_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = load_diagnostic_batch(
                self.write_batch(root, "bad.json", "2026-01-01", "2026-03-31", 100, "bad")
            )
            current = load_diagnostic_batch(
                self.write_batch(root, "current.json", "2026-01-01", "2026-03-31", 100)
            )
            with self.assertRaisesRegex(ValueError, "contract hash"):
                evaluate_batches(current, bad, self.contract, self.contract_hash,
                                 self.policy_hash, self.policy)
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                evaluate_batches(current, current, self.contract, self.contract_hash,
                                 self.policy_hash, self.policy)

    def test_windows_utf8_bom_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_batch(root, "windows.json", "2026-01-01", "2026-03-31", 100)
            path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8-sig")
            self.assertEqual(load_diagnostic_batch(path).windows[0].stock_id, "2330")

    def test_malformed_values_and_non_adjacent_indices_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed = self.write_batch(root, "malformed.json", "2026-01-01", "2026-03-31",
                                         100, schema_version="999")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_diagnostic_batch(malformed)
            previous = load_diagnostic_batch(
                self.write_batch(root, "previous.json", "2026-01-01", "2026-03-31", 100)
            )
            current = load_diagnostic_batch(
                self.write_batch(root, "current.json", "2026-04-01", "2026-06-30", 161)
            )
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                evaluate_batches(current, previous, self.contract, self.contract_hash,
                                 self.policy_hash, self.policy)

    def test_non_finite_wrong_type_and_unknown_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_batch(root, "batch.json", "2026-01-01", "2026-03-31", 100)
            original = json.loads(path.read_text(encoding="utf-8"))

            path.write_text(path.read_text(encoding="utf-8").replace(
                '"sample_count": 40', '"sample_count": NaN'
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_diagnostic_batch(path)

            wrong_groups = {**original, "windows": [{**original["windows"][0],
                                                       "affected_groups": "trade"}]}
            path.write_text(json.dumps(wrong_groups), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "array of strings"):
                load_diagnostic_batch(path)

            path.write_text(json.dumps({**original, "unexpected": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra=.*unexpected"):
                load_diagnostic_batch(path)

            invalid_domains = {**original, "windows": [{**original["windows"][0],
                                                          "disagreement": -1.0}]}
            path.write_text(json.dumps(invalid_domains), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagreement must be in"):
                load_diagnostic_batch(path)

            wrong_count = {**original, "windows": [{**original["windows"][0],
                                                      "sample_count": 41}]}
            path.write_text(json.dumps(wrong_count), encoding="utf-8")
            current = load_diagnostic_batch(path)
            previous_path = self.write_batch(root, "previous.json", "2025-10-01",
                                             "2025-12-31", 40)
            previous = load_diagnostic_batch(previous_path)
            with self.assertRaisesRegex(ValueError, "exactly 40"):
                evaluate_batches(current, previous, self.contract, self.contract_hash,
                                 self.policy_hash, self.policy)


if __name__ == "__main__":
    unittest.main()