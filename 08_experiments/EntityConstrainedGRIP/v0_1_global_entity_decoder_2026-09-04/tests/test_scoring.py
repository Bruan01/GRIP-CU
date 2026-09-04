import unittest

from entity_decoder.scoring import rank_entities, select_normalization


class ScoringTest(unittest.TestCase):
    def test_rank_is_deterministic_under_score_ties(self):
        result = rank_entities(["b", "a", "gold"], [1.0, 1.0, 0.5], "gold", top_k=2)
        self.assertEqual(result["rank"], 3)
        self.assertEqual([row["entity"] for row in result["top_entities"]], ["a", "b"])

    def test_selects_normalization_on_validation_hit_then_mrr(self):
        metrics = {
            "sum": {"hit_at_1": 0.4, "mrr": 0.5},
            "mean": {"hit_at_1": 0.4, "mrr": 0.6},
        }
        self.assertEqual(select_normalization(metrics, ["sum", "mean"]), "mean")


if __name__ == "__main__":
    unittest.main()
