import unittest
from priority_distill.metrics import parse_trace_output, score_predictions
from priority_distill.supervision import build_method_supervision, build_gold_trace_prompt


class JointSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "task_id": "q1", "split": "train", "text": "h --r1--> m --r2--> t",
            "answer": "t", "depth_label": 2, "path_nodes": ["h", "m", "t"],
            "path_relations": ["r1", "r2"],
        }
        gold = {**self.row, "source_task_id": "q1", "source_split": "train", "is_gold": True}
        d = {**self.row, "task_id": "q2", "source_task_id": "q2", "source_split": "train", "answer": "x", "path_nodes": ["h", "n", "x"], "is_gold": False}
        d2 = {**self.row, "task_id": "q3", "source_task_id": "q3", "source_split": "train", "answer": "y", "path_nodes": ["h", "p", "y"], "is_gold": False}
        d3 = {**self.row, "task_id": "q4", "source_task_id": "q4", "source_split": "train", "answer": "z", "path_nodes": ["h", "q", "z"], "is_gold": False}
        self.pools = {"q1": [gold, d, d2, d3]}

    def test_joint_target_has_weighted_segments(self):
        row = build_method_supervision([self.row], self.pools, "graph_free_trace_answer_joint", 3)[0]
        self.assertEqual(row["target_text"], "Trace: h -> m -> <MASKED_TERMINAL>\nAnswer: t")
        self.assertEqual(row["target_segments"][0]["weight"], 0.25)
        self.assertEqual(row["target_segments"][1]["weight"], 1.0)

    def test_gold_trace_hides_terminal(self):
        prompt = build_gold_trace_prompt(self.row)
        self.assertIn("h -> m -> <MASKED_TERMINAL>", prompt)
        self.assertNotIn("Gold intermediate trace (terminal hidden): h -> m -> t", prompt)

    def test_structured_metrics_score_trace_and_answer(self):
        rows = [{"depth_label": 2, "answer": "t", "expected_trace": "h -> m -> <MASKED_TERMINAL>", "prediction_trace": parse_trace_output("Trace: h -> m -> <MASKED_TERMINAL>\nAnswer: t"), "prediction_answer": "t"}]
        metrics = score_predictions(rows)
        self.assertEqual(metrics["trace"]["correct"], 1)
        self.assertEqual(metrics["joint"]["correct"], 1)

if __name__ == "__main__":
    unittest.main()
