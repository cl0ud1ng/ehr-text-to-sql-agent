import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.sql_executor import execute_sql


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "ehr.sqlite"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE patient (id INTEGER PRIMARY KEY, gender TEXT);
                INSERT INTO patient VALUES (1, 'Female'), (2, 'Male'), (3, 'Female');
                """
            )
        finally:
            conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_execute_select_with_max_rows_truncation(self):
        result = execute_sql(
            "SELECT id, gender FROM patient ORDER BY id",
            "test",
            db_path=str(self.db_path),
            max_rows=2,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["columns"], ["id", "gender"])
        self.assertEqual(result["row_count"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rows"][0], [1, "Female"])

    def test_rejects_write_sql_before_execution(self):
        result = execute_sql("INSERT INTO patient VALUES (4, 'Other')", "test", db_path=str(self.db_path))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "validation_error")
        self.assertIn("SELECT", result["error"]["message"])
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM patient").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 3)

    def test_returns_error_dict_for_sql_errors(self):
        result = execute_sql("SELECT missing_column FROM patient", "test", db_path=str(self.db_path))

        self.assertFalse(result["ok"])
        self.assertEqual(result["columns"], [])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["error"]["type"], "OperationalError")
        self.assertIn("missing_column", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
