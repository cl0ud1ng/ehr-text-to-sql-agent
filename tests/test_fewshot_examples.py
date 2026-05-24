import unittest

from src.fewshot_examples import build_fewshot_context, infer_example_type


class FewshotExamplesTests(unittest.TestCase):
    def test_basic_examples_are_deterministic_and_from_representatives(self):
        first = build_fewshot_context("mimic_iii", "What is the method of vancomycin intake?", example_type="basic")
        second = build_fewshot_context("mimic_iii", "What is the method of vancomycin intake?", example_type="basic")

        self.assertEqual(first["example_type"], "basic")
        self.assertEqual(first["examples"], second["examples"])
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.assertIn("data/EHRSQL/示例数据", first["prompt_block"])
        self.assertNotIn("fluconazole", first["prompt_block"].lower())

        source_groups = {example.get("source_group") for example in first["examples"]}
        self.assertIn("possible_examples", source_groups)
        self.assertIn("impossible_examples", source_groups)

    def test_time_examples_prefer_matching_tags_and_are_deterministic(self):
        metadata = {"t_tag": ["", "exact-first", "rel-year-since"]}

        first = build_fewshot_context("mimic_iii", "When was the first prescription?", example_type="time", sample_metadata=metadata)
        second = build_fewshot_context("mimic_iii", "When was the first prescription?", example_type="time", sample_metadata=metadata)

        patterns = [example for example in first["examples"] if example["kind"] == "time_pattern"]
        self.assertEqual(first["examples"], second["examples"])
        self.assertEqual(patterns[0]["tag"], "exact-first")
        self.assertEqual(patterns[1]["tag"], "rel-year-since")
        self.assertTrue(any(example["kind"] == "sql_example" for example in first["examples"]))
        self.assertIn("mimic_iii_test_split_tag_representatives.json", patterns[0]["source_file"])

    def test_auto_type_uses_metadata_before_question_heuristics(self):
        self.assertEqual(infer_example_type("How many patients are there?", sample_metadata={"t_tag": ["exact-last"]}), "time")
        self.assertEqual(infer_example_type("How many patients are there?", sample_metadata={"example_type": "basic"}), "basic")
        self.assertEqual(infer_example_type("How many patients were admitted since last year?"), "time")
        self.assertEqual(infer_example_type("How many patients are there?"), "basic")


if __name__ == "__main__":
    unittest.main()
