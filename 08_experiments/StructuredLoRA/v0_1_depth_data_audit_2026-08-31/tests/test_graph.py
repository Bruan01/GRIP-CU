from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_lora_audit.depth import depth_bucket, label_split
from structured_lora_audit.graph import KnowledgeGraph


class GraphDepthTest(unittest.TestCase):
    def test_leave_one_out_removes_only_the_selected_pair(self) -> None:
        triples = [
            ("a", "direct", "b"),
            ("a", "to_c", "c"),
            ("c", "to_b", "b"),
        ]
        graph = KnowledgeGraph(triples)
        self.assertEqual(graph.leave_one_out_distance(triples[0], max_depth=4, directed=True), 2)
        self.assertEqual(graph.leave_one_out_distance(triples[0], max_depth=4, directed=False), 2)

    def test_parallel_relation_is_valid_one_step_alternative(self) -> None:
        triples = [("a", "r1", "b"), ("a", "r2", "b")]
        graph = KnowledgeGraph(triples)
        self.assertEqual(graph.leave_one_out_distance(triples[0], max_depth=4, directed=True), 1)
        self.assertEqual(graph.leave_one_out_distance(triples[0], max_depth=4, directed=False), 1)

    def test_direction_is_preserved(self) -> None:
        graph = KnowledgeGraph([("a", "r", "b"), ("c", "r", "b")])
        self.assertIsNone(graph.bounded_distance("a", "c", max_depth=3, directed=True))
        self.assertEqual(graph.bounded_distance("a", "c", max_depth=3, directed=False), 2)

    def test_censored_bucket_is_not_called_unreachable(self) -> None:
        self.assertEqual(depth_bucket(None, 4), ">4_or_unreachable")
        self.assertEqual(depth_bucket(4, 4), "4")

    def test_train_and_eval_label_definitions_differ(self) -> None:
        triples = [("a", "r", "b"), ("a", "x", "c"), ("c", "y", "b")]
        graph = KnowledgeGraph(triples)
        train = label_split(graph, [triples[0]], split="train", max_depth=4, leave_one_out=True)[0]
        test = label_split(graph, [triples[0]], split="test", max_depth=4, leave_one_out=False)[0]
        self.assertEqual(train["directed_distance"], 2)
        self.assertEqual(test["directed_distance"], 1)
        self.assertEqual(train["depth_definition"], "leave_one_edge_out")
        self.assertEqual(test["depth_definition"], "train_graph_support")


if __name__ == "__main__":
    unittest.main()
