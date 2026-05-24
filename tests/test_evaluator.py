import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluator import (
    build_grouped_summary,
    write_rows_csv,
    write_summary_csv,
    _counts,
    _infer_example_type_from_path,
    _metrics,
    _sample_prompt_metadata,
)


class EvaluatorMetricTests(unittest.TestCase):
    def test_fewshot_example_type_helpers_do_not_expose_gold_sql(self):
        self.assertEqual(_infer_example_type_from_path("data/EHRSQL/测试集/mimic_iii_test_not_empty.json", "auto"), "time")
        self.assertEqual(_infer_example_type_from_path("data/EHRSQL/测试集/mimic_iii_test_empty.json", "auto"), "basic")
        self.assertEqual(_infer_example_type_from_path("custom.json", "basic"), "basic")

        metadata = _sample_prompt_metadata(
            {
                "id": "abc",
                "tag": "tag",
                "t_tag": ["exact-first"],
                "query": "select secret",
                "is_impossible": False,
            },
            "time",
        )

        self.assertEqual(metadata["example_type"], "time")
        self.assertEqual(metadata["t_tag"], ["exact-first"])
        self.assertNotIn("query", metadata)
        self.assertNotIn("is_impossible", metadata)

    def test_counts_and_metrics_include_report_fields(self):
        rows = [
            {
                "expected_unanswerable": False,
                "predicted_unanswerable": False,
                "ok": True,
                "match": True,
                "repair_count": 1,
                "json_or_model_error": False,
            },
            {
                "expected_unanswerable": False,
                "predicted_unanswerable": True,
                "ok": None,
                "match": False,
                "repair_count": 0,
                "json_or_model_error": True,
            },
            {
                "expected_unanswerable": True,
                "predicted_unanswerable": True,
                "ok": None,
                "match": True,
                "repair_count": 0,
                "json_or_model_error": False,
            },
            {
                "expected_unanswerable": True,
                "predicted_unanswerable": False,
                "ok": True,
                "match": False,
                "repair_count": 0,
                "json_or_model_error": False,
            },
        ]

        counts = _counts(rows)
        metrics = _metrics(counts)

        self.assertEqual(counts["expected_answerable"], 2)
        self.assertEqual(counts["true_unanswerable"], 1)
        self.assertEqual(counts["false_unanswerable"], 1)
        self.assertEqual(counts["missed_unanswerable"], 1)
        self.assertEqual(counts["answerable_execution_success"], 1)
        self.assertEqual(counts["json_or_model_errors"], 1)
        self.assertEqual(metrics["unanswerable_precision"], 0.5)
        self.assertEqual(metrics["unanswerable_recall"], 0.5)
        self.assertEqual(metrics["unanswerable_f1"], 0.5)
        self.assertEqual(metrics["average_repair_count"], 0.25)
        self.assertEqual(metrics["json_or_model_error_rate"], 0.25)

    def test_grouped_summary_and_csv_outputs(self):
        mimic = {
            "counts": {
                "total": 2,
                "expected_answerable": 1,
                "expected_unanswerable": 1,
                "predicted_unanswerable": 1,
                "true_unanswerable": 1,
                "false_unanswerable": 0,
                "missed_unanswerable": 0,
                "execution_success": 1,
                "execution_match": 2,
                "answerable_execution_success": 1,
                "answerable_execution_match": 1,
                "gold_unavailable": 0,
                "total_repairs": 1,
                "json_or_model_errors": 0,
            }
        }
        eicu = {
            "counts": {
                "total": 2,
                "expected_answerable": 2,
                "expected_unanswerable": 0,
                "predicted_unanswerable": 1,
                "true_unanswerable": 0,
                "false_unanswerable": 1,
                "missed_unanswerable": 0,
                "execution_success": 1,
                "execution_match": 1,
                "answerable_execution_success": 1,
                "answerable_execution_match": 1,
                "gold_unavailable": 0,
                "total_repairs": 0,
                "json_or_model_errors": 1,
            }
        }

        summary = build_grouped_summary({"mimic_iii": mimic, "eicu": eicu})

        self.assertEqual(summary["overall"]["counts"]["total"], 4)
        self.assertEqual(summary["overall"]["counts"]["expected_answerable"], 3)
        self.assertEqual(summary["overall"]["counts"]["false_unanswerable"], 1)
        self.assertEqual(summary["overall"]["metrics"]["answerable_execution_match_rate"], 2 / 3)
        self.assertEqual([row["db_id"] for row in summary["summary_rows"]], ["eicu", "mimic_iii", "overall"])

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_csv = Path(tmpdir) / "summary.csv"
            rows_csv = Path(tmpdir) / "rows.csv"
            write_summary_csv(summary, summary_csv)
            write_rows_csv(
                {
                    "rows": [
                        {
                            "index": 0,
                            "id": "a",
                            "question": "q",
                            "expected_unanswerable": False,
                            "predicted_unanswerable": False,
                            "ok": True,
                            "match": True,
                            "repair_count": 0,
                            "json_or_model_error": False,
                            "generated_sql": "select 1",
                            "log_path": "run.json",
                            "gold_error": {"type": "none"},
                        }
                    ]
                },
                rows_csv,
            )

            with summary_csv.open("r", encoding="utf-8", newline="") as fh:
                summary_rows = list(csv.DictReader(fh))
            self.assertEqual(summary_rows[-1]["db_id"], "overall")
            self.assertEqual(summary_rows[-1]["total"], "4")

            with rows_csv.open("r", encoding="utf-8", newline="") as fh:
                sample_rows = list(csv.DictReader(fh))
            self.assertEqual(sample_rows[0]["generated_sql"], "select 1")
            self.assertEqual(json.loads(sample_rows[0]["gold_error"]), {"type": "none"})


if __name__ == "__main__":
    unittest.main()
