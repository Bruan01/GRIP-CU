import json
import tempfile
import unittest
from pathlib import Path

from priority_distill.records import load_exact_hop_jsonl, validate_splits


class RecordTests(unittest.TestCase):
    def row(self, task_id="train:1", split="train", depth=2):
        return {
            "task_id": task_id,
            "text": "Starting from A, follow r1 then r2.",
            "answer": "C",
            "depth_label": depth,
            "depth_source": "exact_directed_shortest_path",
            "path_nodes": ["A", "B", "C"],
            "path_relations": ["r1", "r2"],
            "path_edges": [
                {"head": "A", "relation": "r1", "tail": "B"},
                {"head": "B", "relation": "r2", "tail": "C"},
            ],
            "shortest_path_verified": True,
            "unique_relation_chain_answer": True,
            "split": split,
        }

    def test_loader_accepts_strict_path_and_preserves_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(self.row()) + "\n", encoding="utf-8")
            rows = load_exact_hop_jsonl(path)
        self.assertEqual(rows[0]["path_relations"], ["r1", "r2"])
        self.assertEqual(rows[0]["depth_label"], 2)

    def test_split_validation_rejects_task_leakage(self):
        row = self.row(task_id="shared")
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits({"train": [row], "validation": [{**row, "split": "validation"}], "test": []})


if __name__ == "__main__":
    unittest.main()
