import unittest

from entity_decoder.metrics import (
    canonicalize_prediction,
    compare_decoder_predictions,
    ranking_metrics,
    score_prediction_rows,
)


class MetricsTest(unittest.TestCase):
    def test_canonicalizes_single_entity_embedded_in_output(self):
        vocab = ["concept_a", "concept_ab"]
        self.assertEqual(canonicalize_prediction("Answer: concept_a", vocab), "concept_a")
        self.assertEqual(canonicalize_prediction("concept_a and concept_ab", vocab), "")

    def test_scores_raw_canonical_valid_and_strict_prefix(self):
        vocab = ["concept_alpha", "concept_beta"]
        rows = [
            {"answer": "concept_alpha", "prediction_text": "concept_alpha", "depth_label": 1, "composition": "seen-composition"},
            {"answer": "concept_beta", "prediction_text": "Answer: concept_beta", "depth_label": 2, "composition": "novel-composition"},
            {"answer": "concept_beta", "prediction_text": "concept_bet", "depth_label": 2, "composition": "novel-composition"},
        ]
        result = score_prediction_rows(rows, vocab)
        self.assertAlmostEqual(result["raw_exact_match"], 1 / 3)
        self.assertAlmostEqual(result["canonical_entity_exact_match"], 2 / 3)
        self.assertAlmostEqual(result["valid_entity_rate"], 2 / 3)
        self.assertAlmostEqual(result["strict_prefix_error_rate"], 1 / 3)
        self.assertAlmostEqual(result["by_composition"]["novel-composition"]["canonical_entity_exact_match"], 1 / 2)

    def test_raw_exact_match_is_strict_and_reports_hallucination_alias(self):
        vocab = ["concept_alpha", "concept_beta"]
        rows = [
            {"answer": "concept_alpha", "prediction_text": "Concept_Alpha", "depth_label": 1},
            {"answer": "concept_beta", "prediction_text": "not_an_entity", "depth_label": 1},
        ]
        result = score_prediction_rows(rows, vocab)
        self.assertEqual(result["raw_exact_match"], 0.0)
        self.assertEqual(result["canonical_entity_exact_match"], 0.5)
        self.assertEqual(result["invalid_or_hallucinated_entity_rate"], 0.5)
        self.assertEqual(result["invalid_or_hallucinated_entity_rate"], result["invalid_entity_rate"])

    def test_compares_d0_to_d1_error_transitions_by_task_id(self):
        vocab = ["entity_a", "entity_b", "entity_c", "entity_d"]
        d0 = [
            {"task_id": "t1", "answer": "entity_a", "prediction_text": "garbage"},
            {"task_id": "t2", "answer": "entity_b", "prediction_text": "entity_c"},
            {"task_id": "t3", "answer": "entity_c", "prediction_text": "entity_c"},
            {"task_id": "t4", "answer": "entity_d", "prediction_text": "unknown"},
            {"task_id": "t5", "answer": "entity_a", "prediction_text": "entity_b"},
        ]
        d1 = [
            {"task_id": "t5", "answer": "entity_a", "prediction_text": "entity_d"},
            {"task_id": "t3", "answer": "entity_c", "prediction_text": "entity_d"},
            {"task_id": "t1", "answer": "entity_a", "prediction_text": "entity_a"},
            {"task_id": "t4", "answer": "entity_d", "prediction_text": "entity_b"},
            {"task_id": "t2", "answer": "entity_b", "prediction_text": "entity_b"},
        ]
        result = compare_decoder_predictions(d0, d1, vocab)
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["counts"]["d0_invalid_to_d1_correct"], 1)
        self.assertEqual(result["counts"]["d0_valid_wrong_to_d1_correct"], 1)
        self.assertEqual(result["counts"]["d0_correct_to_d1_wrong"], 1)
        self.assertEqual(result["counts"]["d0_wrong_unchanged"], 2)
        self.assertEqual(result["counts"]["d0_invalid_to_d1_wrong"], 1)
        self.assertEqual(result["counts"]["d0_valid_wrong_to_d1_wrong"], 1)
        self.assertEqual(result["counts"]["d0_correct_to_d1_correct"], 0)
        self.assertAlmostEqual(result["rates"]["d0_invalid_to_d1_correct"], 0.2)

    def test_transition_comparison_rejects_task_id_mismatch(self):
        with self.assertRaises(ValueError):
            compare_decoder_predictions(
                [{"task_id": "a", "answer": "entity_a", "prediction_text": "entity_a"}],
                [{"task_id": "b", "answer": "entity_a", "prediction_text": "entity_a"}],
                ["entity_a"],
            )

    def test_ranking_metrics(self):
        result = ranking_metrics([1, 2, 10, 11])
        self.assertEqual(result["hit_at_1"], 0.25)
        self.assertEqual(result["hit_at_5"], 0.5)
        self.assertEqual(result["hit_at_10"], 0.75)
        self.assertAlmostEqual(result["mrr"], (1 + 0.5 + 0.1 + 1 / 11) / 4)


if __name__ == "__main__":
    unittest.main()
