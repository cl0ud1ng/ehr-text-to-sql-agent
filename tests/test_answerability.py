import unittest

from src.agent.answerability import rule_based_answerability


class AnswerabilityRuleTests(unittest.TestCase):
    def test_cost_questions_are_answerable_with_event_schema(self):
        schema_context = """
        Table COST
          cost REAL
        Table D_LABITEMS
          label TEXT
        Table LABEVENTS
          itemid INTEGER
        """

        result = rule_based_answerability("What is the price of albumin?", schema_context)

        self.assertTrue(result["answerable"])
        self.assertEqual(result["source"], "rules")

    def test_unsupported_future_visit_question_is_not_answerable(self):
        result = rule_based_answerability("When is the earliest next hospital visit for patient 73652?", "Table admissions")

        self.assertFalse(result["answerable"])
        self.assertEqual(result["source"], "rules")

    def test_unsupported_advice_and_external_context_questions_are_not_answerable(self):
        questions = [
            "What is the effect of furosemide 10 mg/ml inj soln?",
            "What is the phone number of patient 011-55642's companion?",
            "What medication does patient 006-2586 take after being prescribed by our hospital's other department?",
        ]

        for question in questions:
            with self.subTest(question=question):
                result = rule_based_answerability(question, "Table patient")
                self.assertFalse(result["answerable"])
                self.assertEqual(result["source"], "rules")


if __name__ == "__main__":
    unittest.main()
