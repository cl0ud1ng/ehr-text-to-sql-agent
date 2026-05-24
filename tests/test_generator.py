from pathlib import Path
import unittest

from src.agent.generator import heuristic_generate_sql
from src.agent.generator import generate_sql
from src.llm_client import LLMResponse
from src.sql_executor import execute_sql


class FakeLLMClient:
    available = True

    def __init__(self):
        self.messages = None
        self.cache_parts = None

    def json_chat(self, messages, *, cache_parts=(), model=None, task_name="json"):
        self.messages = messages
        self.cache_parts = list(cache_parts)
        return LLMResponse(
            content='{"answerable": true, "sql": "select count(*) from patients", "used_tables": ["patients"]}',
            parsed={"answerable": True, "sql": "select count(*) from patients", "used_tables": ["patients"]},
            model=model or "fake",
            cache_hit=False,
            elapsed_ms=0,
        )


class GeneratorHeuristicTests(unittest.TestCase):
    def test_fewshot_prompt_injects_dynamic_examples(self):
        client = FakeLLMClient()

        result = generate_sql(
            "How many admissions are recorded?",
            "mimic_iii",
            "Table patients\nTable admissions",
            prompt_version="fewshot",
            llm_client=client,
            example_type="time",
            sample_metadata={"t_tag": ["exact-first"]},
        )

        system_prompt = client.messages[0]["content"]
        self.assertEqual(result["fewshot"]["example_type"], "time")
        self.assertIn("data/EHRSQL/示例数据", system_prompt)
        self.assertIn("Tag: exact-first", system_prompt)
        self.assertNotIn("fluconazole", system_prompt.lower())
        self.assertIn(result["fewshot"]["cache_key"], client.cache_parts)

    def test_initial_hospital_admission_maps_to_first_stay(self):
        db_path = Path(__file__).resolve().parents[1] / "data" / "EHRSQL" / "mimic_iii.sqlite"
        if not db_path.exists():
            self.skipTest("real EHRSQL database is not available")

        stay = heuristic_generate_sql(
            "Could you provide the length of stay for patient 75581's initial hospital admission?",
            "mimic_iii",
        )

        self.assertIsNotNone(stay)
        self.assertIn("order by admissions.admittime asc", stay["sql"])
        self.assertTrue(execute_sql(stay["sql"], "mimic_iii")["ok"])

    def test_mimic_route_template_normalizes_chemo_prefix(self):
        db_path = Path(__file__).resolve().parents[1] / "data" / "EHRSQL" / "mimic_iii.sqlite"
        if not db_path.exists():
            self.skipTest("real EHRSQL database is not available")

        route = heuristic_generate_sql("What are the methods of intake for the chemo syringe (chemo)?", "mimic_iii")

        self.assertIsNotNone(route)
        self.assertIn("syringe (chemo)", route["sql"])
        self.assertNotIn("chemo syringe (chemo)", route["sql"])
        self.assertTrue(execute_sql(route["sql"], "mimic_iii")["ok"])

    def test_mimic_lab_and_drug_cost_templates_execute(self):
        db_path = Path(__file__).resolve().parents[1] / "data" / "EHRSQL" / "mimic_iii.sqlite"
        if not db_path.exists():
            self.skipTest("real EHRSQL database is not available")

        lab = heuristic_generate_sql("What is the price of albumin?", "mimic_iii")
        implicit_lab = heuristic_generate_sql("What is the CD3 %?", "mimic_iii")
        drug = heuristic_generate_sql("How much does it cost to take multivitamin 12?", "mimic_iii")
        drug_with_duplicate_suffix = heuristic_generate_sql(
            "What is the price of a drug named amoxicillin oral susp. suspension?",
            "mimic_iii",
        )

        self.assertIsNotNone(lab)
        self.assertIn("d_labitems", lab["sql"])
        self.assertTrue(execute_sql(lab["sql"], "mimic_iii")["ok"])

        self.assertIsNotNone(implicit_lab)
        self.assertEqual(implicit_lab["used_tables"], ["COST", "LABEVENTS", "D_LABITEMS"])
        self.assertIn("d_labitems", implicit_lab["sql"])
        self.assertTrue(execute_sql(implicit_lab["sql"], "mimic_iii")["ok"])

        self.assertIsNotNone(drug)
        self.assertIn("prescriptions", drug["sql"])
        self.assertTrue(execute_sql(drug["sql"], "mimic_iii")["ok"])

        self.assertIsNotNone(drug_with_duplicate_suffix)
        self.assertIn("amoxicillin oral susp.", drug_with_duplicate_suffix["sql"])
        self.assertNotIn("amoxicillin oral susp. suspension", drug_with_duplicate_suffix["sql"])
        self.assertTrue(execute_sql(drug_with_duplicate_suffix["sql"], "mimic_iii")["ok"])

    def test_mimic_procedure_cost_template_executes(self):
        db_path = Path(__file__).resolve().parents[1] / "data" / "EHRSQL" / "mimic_iii.sqlite"
        if not db_path.exists():
            self.skipTest("real EHRSQL database is not available")

        procedure = heuristic_generate_sql("What is the cost for bronch/lung dx proc nec?", "mimic_iii")

        self.assertIsNotNone(procedure)
        self.assertEqual(procedure["used_tables"], ["COST", "PROCEDURES_ICD", "D_ICD_PROCEDURES"])
        self.assertIn("procedures_icd", procedure["sql"])
        self.assertTrue(execute_sql(procedure["sql"], "mimic_iii")["ok"])

    def test_eicu_treatment_lab_and_drug_cost_templates_execute(self):
        db_path = Path(__file__).resolve().parents[1] / "data" / "EHRSQL" / "eicu.sqlite"
        if not db_path.exists():
            self.skipTest("real EHRSQL database is not available")

        treatment = heuristic_generate_sql("What is the cost of nitroglycerin - oral?", "eicu")
        lab = heuristic_generate_sql("How much does it cost for folate lab tests?", "eicu")
        drug = heuristic_generate_sql("Can you tell me the costs of Klonopin?", "eicu")

        self.assertIsNotNone(treatment)
        self.assertEqual(treatment["used_tables"], ["cost", "treatment"])
        self.assertIn("treatment", treatment["sql"])
        self.assertNotIn("medication.drugname", treatment["sql"])
        self.assertTrue(execute_sql(treatment["sql"], "eicu")["ok"])

        self.assertIsNotNone(lab)
        self.assertEqual(lab["used_tables"], ["cost", "lab"])
        self.assertIn("cost.eventtype = 'lab'", lab["sql"])
        self.assertNotIn("folate lab tests", lab["sql"])
        self.assertTrue(execute_sql(lab["sql"], "eicu")["ok"])

        self.assertIsNotNone(drug)
        self.assertEqual(drug["used_tables"], ["cost", "medication"])
        self.assertIn("medication", drug["sql"])
        self.assertTrue(execute_sql(drug["sql"], "eicu")["ok"])


if __name__ == "__main__":
    unittest.main()
