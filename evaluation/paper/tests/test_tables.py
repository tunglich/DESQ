from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))

import generate_tables  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
	with path.open("r", encoding="utf-8-sig", newline="") as handle:
		return list(csv.DictReader(handle))


class EvaluationTablesTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		generate_tables.main()

	def test_manifest_covers_tables_3_through_10(self):
		manifest = json.loads((EVALUATION_DIR / "tables_manifest.json").read_text(encoding="utf-8"))
		self.assertEqual(set(manifest["tables"]), {str(number) for number in range(3, 11)})
		for number in range(3, 11):
			prefix = f"table{number}_"
			for suffix in ("csv", "md", "tex"):
				self.assertTrue(any((EVALUATION_DIR / "tables").glob(f"{prefix}*.{suffix}")))

	def test_source_hash_is_newline_independent(self):
		with tempfile.TemporaryDirectory() as directory:
			lf_path = Path(directory) / "lf.csv"
			crlf_path = Path(directory) / "crlf.csv"
			lf_path.write_bytes(b"a,b\n1,2\n")
			crlf_path.write_bytes(b"a,b\r\n1,2\r\n")
			self.assertEqual(generate_tables.sha256(lf_path), generate_tables.sha256(crlf_path))

	def test_table6_core_metrics_reproduce_from_shipped_nav(self):
		rows = read_csv(EVALUATION_DIR / "validation/table6_shipped_nav_check.csv")
		core = {"total_return_pct", "excess_return_pct", "annual_return_pct",
				"annual_volatility_pct", "max_drawdown_pct", "calmar"}
		self.assertTrue(all(row["pass_0_02"] == "True" for row in rows if row["metric"] in core))
		self.assertTrue(any(row["metric"] == "sortino_tbill" and row["pass_0_02"] == "False"
							for row in rows))

	def test_table8_reference_contract(self):
		rows = read_csv(EVALUATION_DIR / "validation/table8_reference_contract_check.csv")
		self.assertEqual([int(row["reference_days"]) for row in rows], [520, 475, 15, 30])
		self.assertTrue(all(row["excess_arithmetic_ok"] == "True" for row in rows))
		self.assertTrue(all(row["partition_days_ok"] == "True" for row in rows))

	def test_table9_has_78_common_features_and_audited_drift(self):
		taxonomy = read_csv(EVALUATION_DIR / "tables/table9_feature_taxonomy.csv")
		drift = read_csv(EVALUATION_DIR / "tables/table9_schema_drift.csv")
		self.assertEqual(sum(int(row["feature_count"]) for row in taxonomy), 78)
		self.assertEqual(len(drift), 6)
		self.assertEqual({row["extra_features"] for row in drift}, {"CMDTY"})

	def test_table10_arithmetic_and_known_ticker_mismatch(self):
		rows = read_csv(EVALUATION_DIR / "validation/table10_arithmetic_universe_check.csv")
		self.assertEqual(len(rows), 50)
		self.assertTrue(all(row["transaction_cost_ok"] == "True" and
							row["return_arithmetic_ok"] == "True" for row in rows))
		mismatches = [row["ticker"] for row in rows if row["in_repo_universe"] != "True"]
		self.assertEqual(mismatches, ["2324.TT"])


if __name__ == "__main__":
	unittest.main()
