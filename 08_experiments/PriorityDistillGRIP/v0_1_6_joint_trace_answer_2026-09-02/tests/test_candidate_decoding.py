import contextlib
import io
import unittest

from scripts.select_sequence_candidate_answers import select


class CandidateDecodingTests(unittest.TestCase):
    def test_selects_highest_sequence_score(self):
        rows = [{
            "task_id": "q1",
            "depth_label": 1,
            "condition": "graph_free",
            "answer": "gold",
            "rank": 2,
            "candidate_scores": [
                {"answer": "wrong", "is_gold": False, "score": 0.9},
                {"answer": "gold", "is_gold": True, "score": 0.8},
                {"answer": "x", "is_gold": False, "score": 0.1},
                {"answer": "y", "is_gold": False, "score": -0.2},
            ],
        }]
        with contextlib.redirect_stdout(io.StringIO()):
            result = select(rows)
        self.assertEqual(result[0]["prediction_answer"], "wrong")
        self.assertFalse(result[0]["sequence_exact_match"])
        self.assertEqual(result[0]["candidate_answers"], ["wrong", "gold", "x", "y"])

    def test_rejects_duplicate_candidates(self):
        row = {
            "task_id": "q1", "depth_label": 1, "condition": "graph_free", "answer": "gold",
            "candidate_scores": [
                {"answer": "gold", "is_gold": True, "score": 1.0},
                {"answer": "gold", "is_gold": False, "score": 0.9},
                {"answer": "x", "is_gold": False, "score": 0.1},
                {"answer": "y", "is_gold": False, "score": 0.0},
            ],
        }
        with self.assertRaisesRegex(ValueError, "not unique"):
            with contextlib.redirect_stdout(io.StringIO()):
                select([row])


if __name__ == "__main__":
    unittest.main()
