import unittest

from scripts.compare_candidate_predictions import compare


class CompareCandidatePredictionTests(unittest.TestCase):
    def test_ranking_compare_uses_rank_one(self):
        direct = {
            "q1": {"rank": 1, "candidate_answers": ["a", "b", "c", "d"]},
            "q2": {"rank": 2, "candidate_answers": ["e", "f", "g", "h"]},
        }
        other = {
            "q1": {"rank": 2, "candidate_answers": ["a", "b", "c", "d"]},
            "q2": {"rank": 1, "candidate_answers": ["e", "f", "g", "h"]},
        }
        evaluation = {
            "q1": {"path_relations": ["r1"]},
            "q2": {"path_relations": ["r2"]},
        }
        report = compare(direct, other, evaluation, {("r1",)}, direct_name="direct", other_name="other", prediction_kind="ranking")
        self.assertEqual(report["overall"]["direct_correct"], 1)
        self.assertEqual(report["overall"]["other_correct"], 1)
        self.assertEqual(report["overall"]["other_wins"], 1)


if __name__ == "__main__":
    unittest.main()
