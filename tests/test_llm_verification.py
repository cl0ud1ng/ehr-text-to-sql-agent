import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.planner import run_agent
from src.llm_client import LLMResponse


class FakeLLMClient:
    def __init__(self, payload=None, *, available=True):
        self.available = available
        self.model = "fake-model"
        self.payload = payload or {}
        self.calls = []

    def json_chat(self, messages, *, cache_parts=(), model=None, task_name="json"):
        if not self.available:
            raise AssertionError("LLM should not be called when unavailable")
        self.calls.append(
            {
                "messages": messages,
                "cache_parts": list(cache_parts),
                "model": model,
                "task_name": task_name,
            }
        )
        return LLMResponse(
            content=json.dumps(self.payload),
            parsed=self.payload,
            model=model or self.model,
            cache_hit=False,
            elapsed_ms=0,
        )


class LLMVerificationTests(unittest.TestCase):
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
                INSERT INTO admissions VALUES (75581, '2020-01-01', '2020-01-03');
                INSERT INTO admissions VALUES (75581, '2020-02-01', '2020-02-04');
                """
            )
        finally:
            conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_heuristic_sql_is_unchanged_when_llm_unavailable(self):
        client = FakeLLMClient(available=False)

        result = self._run_with_client(client)

        self.assertEqual(result["llm_verification"], {})
        self.assertEqual(result["generation"]["source"], "heuristic")
        self.assertTrue(result["execution"]["ok"])

    def test_heuristic_sql_is_verified_when_llm_is_available(self):
        client = FakeLLMClient(
            {
                "verdict": "valid",
                "reason": "The SQL answers the first hospital stay length question.",
                "corrected_sql": "",
                "confidence": 0.93,
            }
        )

        result = self._run_with_client(client)

        self.assertEqual(client.calls[0]["task_name"], "verify_sql")
        self.assertEqual(result["llm_verification"]["verdict"], "valid")
        self.assertTrue(result["generation"]["llm_verified"])
        self.assertEqual(result["generation"]["source"], "heuristic")
        self.assertTrue(result["execution"]["ok"])

    def test_llm_correction_is_applied_after_guard_validation(self):
        corrected_sql = (
            "select 99 as stay_days from admissions "
            "where admissions.subject_id = 75581 limit 1"
        )
        client = FakeLLMClient(
            {
                "verdict": "invalid",
                "reason": "Use the simpler corrected expression.",
                "corrected_sql": corrected_sql,
                "confidence": 0.88,
            }
        )

        result = self._run_with_client(client)

        self.assertEqual(result["generated_sql"], corrected_sql)
        self.assertEqual(result["generation"]["source"], "heuristic_llm_corrected")
        self.assertTrue(result["llm_verification"]["correction_applied"])
        self.assertTrue(result["llm_verification"]["correction_validation"]["ok"])
        self.assertEqual(result["execution"]["rows"], [[99]])

    def test_invalid_llm_correction_is_not_applied(self):
        client = FakeLLMClient(
            {
                "verdict": "invalid",
                "reason": "Unsafe correction.",
                "corrected_sql": "drop table admissions",
                "confidence": 0.5,
            }
        )

        result = self._run_with_client(client)

        self.assertEqual(result["generation"]["source"], "heuristic")
        self.assertFalse(result["llm_verification"]["correction_applied"])
        self.assertFalse(result["llm_verification"]["correction_validation"]["ok"])
        self.assertIn("select strftime", result["generated_sql"])
        self.assertTrue(result["execution"]["ok"])

    def _run_with_client(self, client):
        with patch("src.agent.planner.DeepSeekClient", return_value=client):
            return run_agent(
                "Could you provide the length of stay for patient 75581's initial hospital admission?",
                db_id="mimic_iii",
                db_path=str(self.db_path),
                max_repairs=0,
                max_rows=5,
                use_cache=False,
                save_log=False,
            )


if __name__ == "__main__":
    unittest.main()
