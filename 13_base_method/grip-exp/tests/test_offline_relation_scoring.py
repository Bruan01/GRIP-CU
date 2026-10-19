import unittest

from grip.offline_relation_scoring import (
    build_pair_relation_index,
    compute_candidate_diagnostics,
    resolve_pair_metadata,
    validate_candidate_diagnostics,
)


class OfflineRelationScoringTest(unittest.TestCase):
    def test_pair_index_and_alternative_true_relation_filter(self):
        edges = [
            ["head", "gold", "tail"],
            ["head", "alternative", "tail"],
            ["head", "other", "different-tail"],
        ]
        pair_relations = build_pair_relation_index(edges)
        metadata = resolve_pair_metadata(
            {"source_edge_index": 0}, edges, pair_relations,
        )
        self.assertEqual(metadata["all_true_relations"], ["alternative", "gold"])
        self.assertEqual(metadata["metadata_source"], "source_edge_indices")

        rows, summary = compute_candidate_diagnostics(
            ["gold", "alternative", "negative"],
            [-0.5, -0.4, -1.5],
            "gold",
            metadata["all_true_relations"],
        )
        self.assertFalse(rows[1]["is_valid_negative"])
        self.assertEqual(rows[1]["invalid_reason"], "alternative_true_relation")
        self.assertTrue(rows[2]["is_valid_negative"])
        self.assertEqual(summary["number_of_valid_negatives"], 1)
        self.assertAlmostEqual(rows[2]["negative_mass"], 1.0)

    def test_diagnostics_compute_gap_confusion_mass_and_ranks(self):
        rows, summary = compute_candidate_diagnostics(
            ["gold", "negative_a", "negative_b"],
            [-0.8, -0.9, -1.8],
            "gold",
            [],
            temperature=1.0,
        )
        self.assertAlmostEqual(rows[1]["score_gap"], 0.1)
        self.assertGreater(rows[1]["pairwise_confusion"], 0.4)
        self.assertLess(rows[1]["pairwise_confusion"], 0.5)
        self.assertAlmostEqual(sum(row["negative_mass"] for row in rows if row["negative_mass"] is not None), 1.0)
        self.assertEqual(summary["gold_rank"], 1)
        self.assertEqual([row["candidate_rank_all"] for row in rows], [1, 2, 3])

    def test_validation_checks_all_required_invariants(self):
        rows, _ = compute_candidate_diagnostics(
            ["gold", "negative"], [-0.8, -0.9], "gold", [], temperature=1.0,
        )
        rows[0]["known_true_relations"] = []
        rows[1]["known_true_relations"] = []
        validation = validate_candidate_diagnostics({"qa-1": rows})
        self.assertTrue(validation["all_passed"])
        self.assertTrue(all(validation["checks"].values()))


if __name__ == "__main__":
    unittest.main()
