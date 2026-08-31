import unittest

from structured_lora.metrics import normalize_entity, score_predictions


class MetricsTest(unittest.TestCase):
    def test_entity_extraction_uses_generated_identifier(self):
        self.assertEqual(
            normalize_entity("concept_stateorprovince_wisconsin\nextra text"),
            "concept_stateorprovince_wisconsin",
        )

    def test_depth_macro_is_not_micro_accuracy(self):
        rows = [
            {"depth_label": 1, "answer": "concept_a", "prediction_text": "concept_a"},
            {"depth_label": 1, "answer": "concept_b", "prediction_text": "concept_b"},
            {"depth_label": 2, "answer": "concept_c", "prediction_text": "concept_wrong"},
            {"depth_label": 3, "answer": "concept_d", "prediction_text": "concept_d"},
            {"depth_label": 4, "answer": "concept_e", "prediction_text": "concept_wrong"},
        ]
        metrics = score_predictions(rows)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["macro_depth_accuracy"], 0.5)
        self.assertEqual(metrics["worst_depth_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
