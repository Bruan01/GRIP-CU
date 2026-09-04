import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from entity_decoder.artifacts import import_d0_predictions


class D0ArtifactTest(unittest.TestCase):
    def _write(self, root, rows):
        path = root / "predictions.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path.write_text(payload, encoding="utf-8")
        return path, hashlib.sha256(payload.encode()).hexdigest()

    def test_imports_schema_variants_and_merges_registered_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, sha = self._write(root, [
                {"task_id": "t2", "answer": "entity_b", "prediction": "entity_a", "depth_label": 99},
                {"task_id": "t1", "answer": "entity_a", "prediction_text": "Answer: entity_a"},
            ])
            dataset = [
                {"task_id": "t1", "answer": "entity_a", "depth_label": 1, "composition": "seen-composition"},
                {"task_id": "t2", "answer": "entity_b", "depth_label": 2, "composition": "novel-composition"},
            ]
            rows, audit = import_d0_predictions(
                {"path": path.name, "sha256": sha}, root, dataset
            )
            self.assertEqual([row["task_id"] for row in rows], ["t1", "t2"])
            self.assertEqual(rows[0]["prediction_text"], "Answer: entity_a")
            self.assertEqual(rows[1]["prediction_text"], "entity_a")
            self.assertEqual(rows[1]["depth_label"], 2)
            self.assertEqual(rows[1]["composition"], "novel-composition")
            self.assertEqual(audit["row_count"], 2)
            self.assertEqual(audit["observed_sha256"], sha)
            self.assertEqual(audit["task_id_alignment"], "exact")

    def test_rejects_hash_mismatch_duplicate_ids_and_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, sha = self._write(root, [
                {"task_id": "t1", "answer": "entity_a", "prediction_text": "entity_a"},
                {"task_id": "t1", "answer": "entity_a", "prediction_text": "entity_a"},
            ])
            dataset = [{"task_id": "t1", "answer": "entity_a", "depth_label": 1}]
            with self.assertRaises(ValueError):
                import_d0_predictions({"path": path.name, "sha256": "0" * 64}, root, dataset)
            with self.assertRaises(ValueError):
                import_d0_predictions({"path": path.name, "sha256": sha}, root, dataset)

            path, sha = self._write(root, [
                {"task_id": "other", "answer": "entity_a", "prediction_text": "entity_a"}
            ])
            with self.assertRaises(ValueError):
                import_d0_predictions({"path": path.name, "sha256": sha}, root, dataset)


if __name__ == "__main__":
    unittest.main()
