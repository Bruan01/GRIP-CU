from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_lora_audit.predictions import analyze_prediction_files


class PredictionJoinTest(unittest.TestCase):
    def test_question_is_joined_without_using_previous_true_hop(self) -> None:
        label = {
            "split": "valid",
            "head": "entity_a",
            "tail": "entity_b",
            "relation": "rel",
            "undirected_bucket": "3",
            "undirected_distance": 3,
            "relation_train_frequency": 17,
        }
        prediction = {
            "question_id": "q1",
            "question": (
                "What is the relation between word node entity_a and word node entity_b? "
                "Selected from the following candidate answers: rel; other."
            ),
            "target": ["rel"],
            "true_hop": 1,
            "recurrent_train_k": 1,
            "recurrence_k": 2,
            "adapter_control": "correct",
            "correct": True,
            "metadata": {"split": "validation"},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            metrics, audit = analyze_prediction_files([path], [label])
        self.assertEqual(audit["prediction_rows_joined"], 1)
        self.assertEqual(metrics[0]["depth_bucket"], "3")
        self.assertEqual(metrics[0]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
