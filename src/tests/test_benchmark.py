from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.model.benchmark import format_benchmark_table, load_comparison_rows


class BenchmarkTest(unittest.TestCase):
    def test_format_benchmark_table_includes_own_and_external_model(self) -> None:
        own_result = {
            "model": "SFW-SwinCBM",
            "split": "test",
            "sample_count": 100,
            "metrics": {"accuracy": 0.8, "balanced_accuracy": 0.75, "precision_ai": 0.7, "recall_ai": 0.9, "f1_ai": 0.7879},
        }
        table = format_benchmark_table(own_result, [{"model": "External", "sample_count": 100, "accuracy": 0.6}])

        self.assertIn("SFW-SwinCBM", table)
        self.assertIn("External", table)
        self.assertIn("0.8000", table)

    def test_load_comparison_rows_flattens_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "compare.json"
            path.write_text(
                json.dumps({"models": [{"model": "Paper model", "metrics": {"accuracy": 0.7}}]}),
                encoding="utf-8",
            )
            rows = load_comparison_rows(path)

        self.assertEqual(rows[0]["model"], "Paper model")
        self.assertEqual(rows[0]["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
