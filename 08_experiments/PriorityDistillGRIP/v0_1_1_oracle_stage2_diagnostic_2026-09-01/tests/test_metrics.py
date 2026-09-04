import unittest

from priority_distill.metrics import score_predictions


class MetricTests(unittest.TestCase):
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
