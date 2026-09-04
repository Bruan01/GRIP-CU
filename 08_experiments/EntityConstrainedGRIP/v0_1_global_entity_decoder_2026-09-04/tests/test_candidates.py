import unittest

from entity_decoder.candidates import build_diagnostic_candidates


class CandidateTest(unittest.TestCase):
    def test_d3_contains_gold_and_three_train_only_same_depth_distractors(self):
        train = [
            {"task_id": f"t{i}", "depth_label": 2, "answer": f"e{i}"}
            for i in range(6)
        ] + [{"task_id": "other", "depth_label": 1, "answer": "wrong_depth"}]
        query = {"task_id": "q", "depth_label": 2, "answer": "gold"}
        candidates = build_diagnostic_candidates(query, train, distractor_count=3, seed=7)
        self.assertEqual(len(candidates), 4)
        self.assertIn("gold", candidates)
        self.assertNotIn("wrong_depth", candidates)
        self.assertEqual(len(set(candidates)), 4)


if __name__ == "__main__":
    unittest.main()
