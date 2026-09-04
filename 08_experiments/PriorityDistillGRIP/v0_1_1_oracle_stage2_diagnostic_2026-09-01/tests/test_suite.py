import unittest

from priority_distill.suite import apply_gate


class GateTests(unittest.TestCase):
    def test_gate_requires_oracle_to_beat_every_registered_control(self):
        def metrics(accuracy, deep):
            return {"seed_count": 3, "test_accuracy": accuracy, "test_deep_3_4_accuracy": deep, "stage1_input_tokens": 1000, "stage1_truncated_example_rate": 0.0}
        aggregate = {
            "answer_only": metrics(0.50, 0.45),
            "more_qa_equal_token": metrics(0.51, 0.46),
            "random_path_equal_token": metrics(0.505, 0.455),
            "all_paths_equal_token": metrics(0.50, 0.44),
            "oracle_priority_equal_token": metrics(0.53, 0.49),
        }
        config = {"gate": {"minimum_seeds_for_final": 3, "oracle_over_answer_only": 0.02, "oracle_over_more_qa": 0.01, "oracle_over_random_path": 0.0, "oracle_over_all_paths": 0.0, "deep_3_4_improvement": 0.0}}
        self.assertEqual(apply_gate(aggregate, config)["status"], "GO_LEARNED_PRIORITIZER")

    def test_one_seed_can_only_issue_preliminary_decision(self):
        def metrics(accuracy):
            return {"seed_count": 1, "test_accuracy": accuracy, "test_deep_3_4_accuracy": accuracy, "stage1_input_tokens": 1000, "stage1_truncated_example_rate": 0.0}
        aggregate = {method: metrics(value) for method, value in {
            "answer_only": 0.50,
            "more_qa_equal_token": 0.51,
            "random_path_equal_token": 0.50,
            "all_paths_equal_token": 0.49,
            "oracle_priority_equal_token": 0.54,
        }.items()}
        config = {"gate": {"minimum_seeds_for_final": 3, "oracle_over_answer_only": 0.02, "oracle_over_more_qa": 0.01, "oracle_over_random_path": 0.0, "oracle_over_all_paths": 0.0, "deep_3_4_improvement": 0.0}}
        self.assertEqual(apply_gate(aggregate, config)["status"], "PRELIMINARY_GO")

    def test_tie_with_random_path_is_not_a_go(self):
        def metrics(accuracy):
            return {"seed_count": 3, "test_accuracy": accuracy, "test_deep_3_4_accuracy": accuracy, "stage1_input_tokens": 1000, "stage1_truncated_example_rate": 0.0}
        aggregate = {
            "answer_only": metrics(0.50),
            "more_qa_equal_token": metrics(0.51),
            "random_path_equal_token": metrics(0.53),
            "all_paths_equal_token": metrics(0.49),
            "oracle_priority_equal_token": metrics(0.53),
        }
        config = {"gate": {"minimum_seeds_for_final": 3, "oracle_over_answer_only": 0.02, "oracle_over_more_qa": 0.01, "oracle_over_random_path": 0.0, "oracle_over_all_paths": 0.0, "deep_3_4_improvement": 0.0, "max_stage1_token_relative_gap": 0.01}}
        self.assertEqual(apply_gate(aggregate, config)["status"], "STOP_PRIORITY_DISTILL")

    def test_prompt_truncation_forces_stop(self):
        def metrics(accuracy, truncation=0.0):
            return {"seed_count": 3, "test_accuracy": accuracy, "test_deep_3_4_accuracy": accuracy, "stage1_input_tokens": 1000, "stage1_truncated_example_rate": truncation}
        aggregate = {
            "answer_only": metrics(0.50),
            "more_qa_equal_token": metrics(0.51),
            "random_path_equal_token": metrics(0.50),
            "all_paths_equal_token": metrics(0.49, truncation=0.05),
            "oracle_priority_equal_token": metrics(0.54),
        }
        config = {"gate": {"minimum_seeds_for_final": 3, "oracle_over_answer_only": 0.02, "oracle_over_more_qa": 0.01, "oracle_over_random_path": 0.0, "oracle_over_all_paths": 0.0, "deep_3_4_improvement": 0.0, "max_stage1_token_relative_gap": 0.01, "max_stage1_truncated_example_rate": 0.0}}
        self.assertEqual(apply_gate(aggregate, config)["status"], "STOP_PRIORITY_DISTILL")

    def test_large_stage1_token_gap_forces_stop(self):
        def metrics(accuracy, tokens):
            return {"seed_count": 3, "test_accuracy": accuracy, "test_deep_3_4_accuracy": accuracy, "stage1_input_tokens": tokens, "stage1_truncated_example_rate": 0.0}
        aggregate = {
            "answer_only": metrics(0.50, 0),
            "more_qa_equal_token": metrics(0.51, 700),
            "random_path_equal_token": metrics(0.50, 1000),
            "all_paths_equal_token": metrics(0.49, 1000),
            "oracle_priority_equal_token": metrics(0.54, 1000),
        }
        config = {"gate": {"minimum_seeds_for_final": 3, "oracle_over_answer_only": 0.02, "oracle_over_more_qa": 0.01, "oracle_over_random_path": 0.0, "oracle_over_all_paths": 0.0, "deep_3_4_improvement": 0.0, "max_stage1_token_relative_gap": 0.01}}
        gate = apply_gate(aggregate, config)
        self.assertEqual(gate["status"], "STOP_PRIORITY_DISTILL")
        self.assertFalse(gate["checks"]["stage1_token_budget_match"]["pass"])


if __name__ == "__main__":
    unittest.main()
