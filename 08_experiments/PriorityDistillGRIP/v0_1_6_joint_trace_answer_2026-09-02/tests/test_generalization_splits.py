import unittest

from scripts.analyze_generalization_splits import analyze


def row(task_id, split, answer, head, relations, depth=None):
    depth = depth or len(relations)
    return {
        "task_id": task_id,
        "split": split,
        "answer": answer,
        "path_nodes": [head, "middle", answer],
        "path_relations": relations,
        "depth_label": depth,
    }


class GeneralizationSplitTests(unittest.TestCase):
    def test_axes_separate_composition_from_entity_novelty(self):
        train = [row("train:1", "train", "a", "h1", ["r1", "r2"])]
        evaluation = [
            row("eval:seen", "test", "a", "h1", ["r1", "r2"]),
            row("eval:novel_composition", "test", "b", "h2", ["r2", "r1"]),
            row("eval:unseen_relation", "test", "c", "h3", ["r3", "r1"]),
        ]
        predictions = [
            {"task_id": "eval:seen", "candidate_answers": ["a", "x", "y", "z"], "gold_in_candidate_set": True, "constrained_exact_match": True},
            {"task_id": "eval:novel_composition", "candidate_answers": ["b", "x", "y", "z"], "gold_in_candidate_set": True, "constrained_exact_match": False},
            {"task_id": "eval:unseen_relation", "candidate_answers": ["c", "x", "y", "z"], "gold_in_candidate_set": True, "constrained_exact_match": True},
        ]
        report = analyze(train, evaluation, predictions, prediction_kind="constrained")
        self.assertEqual(report["axes"]["composition"]["novel"]["count"], 2)
        self.assertEqual(report["axes"]["novel_composition_relation_coverage"]["all_relations_seen"]["count"], 1)
        self.assertEqual(report["axes"]["novel_composition_relation_coverage"]["contains_unseen_relation"]["count"], 1)
        self.assertEqual(report["axes"]["answer_seen_in_train"]["unseen"]["count"], 2)

    def test_ranking_condition_and_coverage_are_strict(self):
        train = [row("train:1", "train", "a", "h1", ["r1"])]
        evaluation = [row("eval:1", "test", "b", "h2", ["r1"])]
        predictions = [{"task_id": "eval:1", "condition": "gold_trace", "rank": 1}]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            analyze(train, evaluation, predictions, prediction_kind="ranking", condition="graph_free")


if __name__ == "__main__":
    unittest.main()
