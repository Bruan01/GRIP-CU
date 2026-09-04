import unittest

from scripts.analyze_composition_split import analyze


def row(task_id, split, relations, depth=1):
    return {
        "task_id": task_id,
        "split": split,
        "path_relations": relations,
        "depth_label": depth,
    }


class CompositionSplitTests(unittest.TestCase):
    def setUp(self):
        self.train = [row("train:1", "train", ["r1", "r2"], depth=2)]
        self.evaluation = [
            row("eval:seen", "test", ["r1", "r2"], depth=2),
            row("eval:novel", "test", ["r2", "r1"], depth=2),
        ]

    def test_strict_grouping_and_coverage(self):
        predictions = [
            {"task_id": "eval:seen", "candidate_answers": ["gold", "a", "b", "c"], "gold_in_candidate_set": True, "constrained_exact_match": True},
            {"task_id": "eval:novel", "candidate_answers": ["gold", "a", "b", "c"], "gold_in_candidate_set": True, "constrained_exact_match": False},
        ]
        report = analyze(self.train, self.evaluation, predictions, prediction_kind="constrained")
        self.assertEqual(report["prediction_count"], 2)
        self.assertEqual(report["groups"]["seen_composition"]["accuracy"], 1.0)
        self.assertEqual(report["groups"]["novel_composition"]["accuracy"], 0.0)

    def test_missing_prediction_is_rejected(self):
        predictions = [{"task_id": "eval:seen", "rank": 1, "condition": "graph_free"}]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            analyze(self.train, self.evaluation, predictions, prediction_kind="ranking")

    def test_duplicate_prediction_is_rejected(self):
        predictions = [
            {"task_id": "eval:seen", "rank": 1, "condition": "graph_free"},
            {"task_id": "eval:seen", "rank": 2, "condition": "graph_free"},
            {"task_id": "eval:novel", "rank": 1, "condition": "graph_free"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            analyze(self.train, self.evaluation, predictions, prediction_kind="ranking")

    def test_ranking_condition_filter_is_strict(self):
        predictions = [
            {"task_id": "eval:seen", "rank": 1, "condition": "graph_free"},
            {"task_id": "eval:novel", "rank": 2, "condition": "gold_trace"},
        ]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            analyze(self.train, self.evaluation, predictions, prediction_kind="ranking", condition="graph_free")


if __name__ == "__main__":
    unittest.main()
