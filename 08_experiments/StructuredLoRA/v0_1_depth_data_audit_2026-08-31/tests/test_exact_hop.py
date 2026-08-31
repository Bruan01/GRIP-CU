from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_lora_audit.exact_hop import sample_exact_hop_tasks
from structured_lora_audit.graph import KnowledgeGraph


class ExactHopTest(unittest.TestCase):
    def setUp(self) -> None:
        triples = []
        for chain in range(8):
            for depth in range(1, 5):
                triples.append((f"n{chain}_{depth-1}", f"r{chain}_{depth}", f"n{chain}_{depth}"))
        self.graph = KnowledgeGraph(triples)

    def test_generated_tasks_preserve_path_provenance(self) -> None:
        tasks, metadata = sample_exact_hop_tasks(
            self.graph,
            per_depth=2,
            max_depth=4,
            seed=7,
            max_attempts_per_depth=10_000,
        )
        self.assertEqual(len(tasks), 8)
        self.assertEqual(metadata["total_tasks"], 8)
        for task in tasks:
            depth = task["depth_label"]
            self.assertEqual(len(task["path_relations"]), depth)
            self.assertEqual(len(task["path_nodes"]), depth + 1)
            self.assertEqual(task["answers"], [task["path_nodes"][-1]])
            self.assertEqual(
                self.graph.bounded_distance(
                    task["path_nodes"][0], task["path_nodes"][-1], max_depth=depth, directed=True
                ),
                depth,
            )
            self.assertEqual(
                self.graph.follow_relations(task["path_nodes"][0], task["path_relations"]),
                {task["path_nodes"][-1]},
            )

    def test_generation_is_deterministic(self) -> None:
        first, _ = sample_exact_hop_tasks(self.graph, per_depth=1, max_depth=4, seed=11)
        second, _ = sample_exact_hop_tasks(self.graph, per_depth=1, max_depth=4, seed=11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
