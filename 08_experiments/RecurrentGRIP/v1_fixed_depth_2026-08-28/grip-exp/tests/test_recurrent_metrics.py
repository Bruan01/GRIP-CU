import unittest

from evaluation.recurrent_metrics import summarize_recurrent_predictions


class RecurrentMetricsTest(unittest.TestCase):
    def test_best_k_correlation_excludes_questions_never_solved(self):
        rows = [
            {"question_id": "q1", "true_hop": 1, "recurrence_k": 1, "adapter_control": "correct", "correct": True},
            {"question_id": "q1", "true_hop": 1, "recurrence_k": 4, "adapter_control": "correct", "correct": True},
            {"question_id": "q2", "true_hop": 4, "recurrence_k": 1, "adapter_control": "correct", "correct": False},
            {"question_id": "q2", "true_hop": 4, "recurrence_k": 4, "adapter_control": "correct", "correct": True},
            {"question_id": "q3", "true_hop": 3, "recurrence_k": 1, "adapter_control": "correct", "correct": False},
            {"question_id": "q3", "true_hop": 3, "recurrence_k": 4, "adapter_control": "correct", "correct": False},
        ]
        summary = summarize_recurrent_predictions(rows)
        self.assertEqual(summary["solved_question_count"], 2)
        self.assertEqual(summary["unsolved_question_count"], 1)
        self.assertEqual(summary["best_k_true_hop_spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
