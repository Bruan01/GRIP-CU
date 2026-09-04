import unittest

from entity_decoder.records import build_prompt, annotate_rows


class RecordProtocolTest(unittest.TestCase):
    def test_answer_only_prompt_contains_question_but_no_graph_evidence(self):
        row = {"text": "Q", "path_nodes": ["gold_head", "gold_answer"], "answer": "gold_answer"}
        prompt = build_prompt(row, "answer_only")
        self.assertIn("Question: Q", prompt)
        self.assertNotIn("gold_head", prompt)
        self.assertNotIn("gold_answer", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_joint_slot_is_fixed_and_query_independent_except_question(self):
        row = {"text": "Q", "path_nodes": ["gold_head", "gold_mid", "gold_answer"], "answer": "gold_answer"}
        prompt = build_prompt(row, "joint_answer_slot")
        self.assertIn("Trace: <MASKED_INTERNAL_TRACE>", prompt)
        self.assertNotIn("gold_head", prompt)
        self.assertNotIn("gold_mid", prompt)
        self.assertNotIn("gold_answer", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_annotation_uses_train_only_composition_and_relation_counts(self):
        train = [{"path_relations": ["r1"], "answer": "a", "depth_label": 1, "text": "q", "task_id": "t"}]
        test = [{"path_relations": ["r2"], "answer": "abcd", "depth_label": 1, "text": "q", "task_id": "x"}]
        row = annotate_rows(train, test, answer_token_lengths={"abcd": 3}, prefix_ambiguities={"abcd": 4})[0]
        self.assertEqual(row["composition"], "novel-composition")
        self.assertEqual(row["relation_frequency_bucket"], "rare")
        self.assertEqual(row["answer_token_length"], 3)
        self.assertEqual(row["entity_prefix_ambiguity"], 4)


if __name__ == "__main__":
    unittest.main()
