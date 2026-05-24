import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.agent.planner import run_agent


class RepairMechanismTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "ehr.sqlite"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE admissions (
                    subject_id INTEGER,
                    admittime TEXT,
                    dischtime TEXT
                );
                CREATE TABLE patients (
                    subject_id INTEGER PRIMARY KEY,
                    gender TEXT
                );
                INSERT INTO admissions VALUES (75581, '2020-01-01', '2020-01-03');
                INSERT INTO admissions VALUES (75581, '2020-02-01', '2020-02-04');
                INSERT INTO patients VALUES (1, 'Female'), (2, 'Male'), (3, 'Female');
                """
            )
        finally:
            conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_repairs_unknown_table_from_validation_error(self):
        initial_sql = "select count(*) as admission_count from admission"

        result = self._run_with_initial_sql("How many hospital admissions are recorded?", initial_sql)

        self.assertTrue(result["execution"]["ok"])
        self.assertEqual(result["execution"]["rows"], [[2]])
        self.assertEqual(len(result["repairs"]), 1)
        self.assertIn("unknown table", result["repairs"][0]["error"])
        self.assertEqual(result["repairs"][0]["failed_sql"], initial_sql)
        self.assertIn("from admissions", result["repairs"][0]["repair"]["sql"])

    def test_repairs_missing_column_from_sqlite_error(self):
        initial_sql = (
            "select admissions.discharge_time from admissions "
            "where admissions.subject_id = 75581 "
            "order by admissions.admittime asc limit 1"
        )

        result = self._run_with_initial_sql(
            "What is the discharge time for the first admission of patient 75581?",
            initial_sql,
        )

        self.assertTrue(result["execution"]["ok"])
        self.assertEqual(result["execution"]["rows"], [["2020-01-03"]])
        self.assertEqual(len(result["repairs"]), 1)
        self.assertIn("no such column", result["repairs"][0]["error"])
        self.assertIn("admissions.dischtime", result["repairs"][0]["repair"]["sql"])

    def test_repairs_unquoted_filter_literal_from_sqlite_error(self):
        initial_sql = "select count(*) as female_count from patients where patients.gender = Female"

        result = self._run_with_initial_sql("How many female patients are there?", initial_sql)

        self.assertTrue(result["execution"]["ok"])
        self.assertEqual(result["execution"]["rows"], [[2]])
        self.assertEqual(len(result["repairs"]), 1)
        self.assertIn("no such column", result["repairs"][0]["error"])
        self.assertIn("patients.gender = 'Female'", result["repairs"][0]["repair"]["sql"])

    def _run_with_initial_sql(self, question: str, sql: str):
        return run_agent(
            question,
            db_id="test",
            db_path=str(self.db_path),
            initial_sql=sql,
            max_repairs=2,
            max_rows=5,
            use_cache=False,
            save_log=False,
        )


if __name__ == "__main__":
    unittest.main()
