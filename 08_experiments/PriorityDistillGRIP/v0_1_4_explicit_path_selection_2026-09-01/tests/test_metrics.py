import unittest

from priority_distill.metrics import parse_explicit_output, score_candidate_selection_predictions, score_predictions


class MetricTests(unittest.TestCase):
    def test_explicit_output_parser_and_selection_metrics(self):
        self.assertEqual(parse_explicit_output("selected PATH: 2\nAnswer: concept_x"), {"selected_path": 2, "answer": "concept_x"})
        rows = [
            {"answer": "concept_x", "prediction_text": "Selected path: 2\nAnswer: concept_x", "depth_label": 2, "selection_target": 2, "candidate_count": 4},
            {"answer": "concept_y", "prediction_text": "Selected path: 1\nAnswer: wrong", "depth_label": 3, "selection_target": 2, "candidate_count": 4},
        ]
        metrics = score_candidate_selection_predictions(rows)
        self.assertEqual(metrics["selection"]["accuracy"], 0.5)
        self.assertEqual(metrics["answer"]["accuracy"], 0.5)
        self.assertEqual(metrics["joint"]["accuracy"], 0.5)
        self.assertEqual(metrics["random_selection_baseline"], 0.25)

    def test_exact_entity_accuracy_and_depth_breakdown(self):
        rows = [
            {"answer": "entity_a", "prediction_text": " entity_a\n", "depth_label": 1},
            {"answer": "entity_b", "prediction_text": "wrong", "depth_label": 3},
        ]
        metrics = score_predictions(rows)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["accuracy_by_depth"]["1"], 1.0)
        self.assertEqual(metrics["accuracy_by_depth"]["3"], 0.0)


if __name__ == "__main__":
    unittest.main()
