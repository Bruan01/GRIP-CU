import unittest

from grip.tasks.recurrent_tasks.split_builder import build_graph_hop_split


class HopSplitTest(unittest.TestCase):
    def test_train_and_test_are_partitioned_by_true_shortest_hop(self):
        nodes = [f"S{i} has disabled access" for i in range(5)]
        pairs = [(0, 1), (1, 2), (0, 2), (2, 4), (0, 3), (1, 4), (0, 4)]
        record = {
            "id": "line",
            "title": "line graph",
            "graph": {
                "node_list": nodes,
                "edge_index": [[0, 1], [1, 2], [2, 3], [3, 4]],
                "edge_list": [],
            },
            "questions": [f"How many stations are between S{a} and S{b}?" for a, b in pairs],
            "answers": [str(max(abs(a - b) - 1, 0)) for a, b in pairs],
            "question_types": ["StationShortestCount"] * len(pairs),
        }
        enriched, _ = build_graph_hop_split(
            record,
            validation_fraction=0.5,
            max_questions_per_hop=0,
            seed=11,
        )
        samples = enriched["recurrent_questions"]
        self.assertTrue(any(item["split"] == "train" for item in samples))
        self.assertTrue(any(item["split"] == "validation" for item in samples))
        self.assertTrue(any(item["split"] == "test" for item in samples))
        for item in samples:
            if item["split"] in {"train", "validation"}:
                self.assertIn(item["true_hop"], {1, 2})
            else:
                self.assertIn(item["true_hop"], {3, 4})

    def test_split_is_deterministic_for_graph_and_seed(self):
        record = {
            "id": "g",
            "graph": {
                "node_list": [f"N{i} has x" for i in range(5)],
                "edge_index": [[0, 1], [1, 2], [2, 3], [3, 4]],
            },
            "questions": [
                "How many stations are between N0 and N1?",
                "How many stations are between N1 and N2?",
                "How many stations are between N0 and N2?",
                "How many stations are between N2 and N4?",
                "How many stations are between N0 and N3?",
                "How many stations are between N0 and N4?",
            ],
            "answers": ["0", "0", "1", "1", "2", "3"],
        }
        first, _ = build_graph_hop_split(record, validation_fraction=0.5, seed=2026)
        second, _ = build_graph_hop_split(record, validation_fraction=0.5, seed=2026)
        self.assertEqual(first["recurrent_questions"], second["recurrent_questions"])


if __name__ == "__main__":
    unittest.main()
