import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.schema_index import format_schema_context, load_schema, retrieve_schema


class SchemaIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "ehr.sqlite"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE PATIENTS (
                    SUBJECT_ID INTEGER PRIMARY KEY,
                    GENDER TEXT,
                    AGE INTEGER
                );
                CREATE TABLE LABEVENTS (
                    ROW_ID INTEGER PRIMARY KEY,
                    SUBJECT_ID INTEGER,
                    ITEMID INTEGER,
                    VALUE TEXT
                );
                INSERT INTO PATIENTS VALUES (1, 'F', 67), (2, 'M', 75);
                INSERT INTO LABEVENTS VALUES (10, 1, 50809, 'glucose'), (11, 2, 50810, 'sodium');
                """
            )
        finally:
            conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_schema_contains_tables_columns_counts_and_samples(self):
        schema = load_schema("test", db_path=str(self.db_path), refresh=True)

        self.assertEqual(schema["table_count"], 2)
        patients = next(table for table in schema["tables"] if table["name"] == "PATIENTS")
        self.assertEqual(patients["row_count"], 2)
        gender = next(column for column in patients["columns"] if column["name"] == "GENDER")
        self.assertEqual(gender["type"], "TEXT")
        self.assertIn("F", gender["sample_values"])

    def test_retrieve_schema_is_deterministic_and_ranks_matches(self):
        first = retrieve_schema("patient glucose lab value", "test", db_path=str(self.db_path), top_k_tables=2)
        second = retrieve_schema("patient glucose lab value", "test", db_path=str(self.db_path), top_k_tables=2)

        self.assertEqual(first, second)
        self.assertEqual(first["tables"][0]["name"], "LABEVENTS")
        self.assertGreater(first["tables"][0]["score"], 0)
        self.assertIn("glucose", first["tables"][0]["matched_terms"])

    def test_format_schema_context_includes_prompt_ready_details(self):
        retrieved = retrieve_schema("age of patient", "test", db_path=str(self.db_path), top_k_tables=1)
        text = format_schema_context(retrieved)

        self.assertIn("Database: test", text)
        self.assertIn("Table PATIENTS", text)
        self.assertIn("AGE INTEGER", text)

    def test_keyword_boost_recalls_domain_tables(self):
        mimic_path = Path(__file__).resolve().parents[1] / "实验三材料" / "EHRSQL" / "mimic_iii.sqlite"
        if not mimic_path.exists():
            self.skipTest("real EHRSQL database is not available")

        retrieved = retrieve_schema("what drug route was administered", "mimic_iii", top_k_tables=3)
        names = [table["name"] for table in retrieved["tables"]]

        self.assertIn("PRESCRIPTIONS", names)


if __name__ == "__main__":
    unittest.main()
