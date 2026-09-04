import unittest

from priority_distill.candidates import build_candidate_pools
from priority_distill.supervision import METHODS, build_method_supervision
from tests.test_candidates import make_row


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [make_row(i, ["r1", f"r{i}"]) for i in range(6)]
        self.pools = build_candidate_pools(self.rows, distractor_count=3, seed=3)

    def test_registered_methods_have_answer_targets(self):
        for method in METHODS:
            records = build_method_supervision(self.rows, self.pools, method, seed=3)
            self.assertTrue(records)
            self.assertTrue(all(record["answer"] for record in records))

    def test_oracle_contains_only_gold_path(self):
        record = build_method_supervision(self.rows, self.pools, "oracle_priority_equal_token", seed=3)[0]
        self.assertEqual(record["evidence_kind"], "oracle_gold_path")
        self.assertEqual(record["candidate_count"], 1)
        self.assertIn("Path 1:", record["prompt"])
        self.assertNotIn("[GOLD PATH]", record["prompt"])

    def test_random_path_excludes_gold(self):
        record = build_method_supervision(self.rows, self.pools, "random_path_equal_token", seed=3)[0]
        self.assertEqual(record["candidate_count"], 1)
        self.assertNotIn("[GOLD PATH]", record["prompt"])
        self.assertNotEqual(record["selected_source_task_ids"], [record["task_id"]])

    def test_all_paths_hides_gold_label_and_randomizes_position(self):
        records = build_method_supervision(self.rows, self.pools, "all_paths_equal_token", seed=3)
        self.assertTrue(all("[GOLD PATH]" not in record["prompt"] for record in records))
        positions = {record["gold_position"] for record in records}
        self.assertGreater(len(positions), 1)

    def test_answer_only_prompt_has_no_graph_evidence(self):
        record = build_method_supervision(self.rows, self.pools, "answer_only", seed=3)[0]
        self.assertEqual(record["candidate_count"], 0)
        self.assertNotIn("Candidate evidence", record["prompt"])


if __name__ == "__main__":
    unittest.main()

# Explicit anti-copy invariants are intentionally separate from the legacy method tests.
class AntiCopySupervisionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [make_row(i, ["r1", f"r{i}"]) for i in range(6)]
        self.pools = build_candidate_pools(self.rows, distractor_count=3, seed=3)

    def test_terminal_mask_is_not_present_in_any_path_as_an_answer(self):
        records = build_method_supervision(
            self.rows, self.pools, "all_paths_terminal_masked_equal_token", seed=3
        )
        for record in records:
            self.assertEqual(record["candidate_count"], 4)
            self.assertIn("Candidate paths (terminal entity hidden):", record["prompt"])
            self.assertEqual(record["prompt"].count("<MASKED_TERMINAL>"), 4)
            self.assertIn("Unassociated candidate terminal entities:", record["prompt"])
            self.assertNotIn(f"--> {record['answer']}\n", record["prompt"])

    def test_terminal_mask_keeps_gold_position_hidden(self):
        records = build_method_supervision(
            self.rows, self.pools, "all_paths_terminal_masked_equal_token", seed=3
        )
        self.assertTrue(all(record["gold_position"] >= 0 for record in records))
        self.assertTrue(all("[GOLD PATH]" not in record["prompt"] for record in records))
