import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.schema_index import load_schema
from src.sql_guard import validate_sql


class SqlGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "ehr.sqlite"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE patient (patientunitstayid INTEGER PRIMARY KEY, gender TEXT);
                CREATE TABLE lab (patientunitstayid INTEGER, labname TEXT);
                INSERT INTO patient VALUES (1, 'Female');
                INSERT INTO lab VALUES (1, 'glucose');
                """
            )
        finally:
            conn.close()
        self.schema = load_schema("test", db_path=str(self.db_path), refresh=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_allows_select_and_with_select(self):
        select_result = validate_sql("SELECT gender FROM patient", schema=self.schema)
        with_result = validate_sql(
            "WITH p AS (SELECT patientunitstayid FROM patient) SELECT * FROM p",
            schema=self.schema,
        )

        self.assertTrue(select_result["ok"])
        self.assertEqual(select_result["tables"], ["patient"])
        self.assertTrue(with_result["ok"])
        self.assertEqual(with_result["tables"], ["patient"])

    def test_blocks_dangerous_and_multiple_statements(self):
        for sql in [
            "DELETE FROM patient",
            "DROP TABLE patient",
            "PRAGMA table_info(patient)",
            "REPLACE INTO patient VALUES (2, 'Male')",
            "SELECT * FROM patient; SELECT * FROM lab",
        ]:
            with self.subTest(sql=sql):
                result = validate_sql(sql, schema=self.schema)
                self.assertFalse(result["ok"])
                self.assertIsNotNone(result["error"])

    def test_unknown_table_is_rejected(self):
        result = validate_sql("SELECT * FROM missing_table", schema=self.schema)

        self.assertFalse(result["ok"])
        self.assertIn("unknown table", result["error"])


if __name__ == "__main__":
    unittest.main()
