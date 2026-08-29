import json
import tempfile
import unittest
from pathlib import Path

from recurrent_cross_audit import audit_recurrent_cross_run


class RecurrentCrossAuditTest(unittest.TestCase):
    def _build_run(self, root: Path, second_hash: str = "same") -> None:
        for train_k, selection_hash in ((1, "same"), (2, second_hash)):
            subdir = root / f"train_k{train_k}"
            adapter_dir = subdir / "adapters" / "nell23k"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "context_sampling_manifest.json").write_text(
                json.dumps(
                    {
                        "selection_sha256": selection_hash,
                        "selected_node_count": 32,
                        "selected_edge_count": 224,
                        "relation_count": 3,
                        "selected_relation_count": 3,
                        "relation_coverage": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            rows = []
            for control in ("correct", "none"):
                for eval_k in (1, 2):
                    for question_id, split in (("qv", "validation"), ("qt", "test")):
                        rows.append(
                            {
                                "graph_id": "nell23k",
                                "question_id": question_id,
                                "recurrent_train_k": train_k,
                                "recurrence_k": eval_k,
                                "adapter_control": control,
                                "correct": False,
                                "response": "r1",
                                "generated_token_count": 2,
                                "ended_with_eos": True,
                                "response_in_candidates": True,
                                "step_pooled_hidden_states": [
                                    [float(step), 0.0] for step in range(1, eval_k + 1)
                                ],
                                "metadata": {"split": split},
                            }
                        )
            with (subdir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")

    def test_accepts_complete_balanced_cross_with_identical_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_run(root)
            report = audit_recurrent_cross_run(root)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["prediction_count"], 16)
        self.assertEqual(report["selection_sha256"], "same")
        self.assertEqual(len(report["matrix_counts"]), 16)

    def test_rejects_trace_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_run(root)
            path = root / "train_k1" / "predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["step_pooled_hidden_states"] = []
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "trace length mismatch"):
                audit_recurrent_cross_run(root)

    def test_rejects_different_context_samples_between_train_depths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_run(root, second_hash="different")
            with self.assertRaisesRegex(ValueError, "selection_sha256"):
                audit_recurrent_cross_run(root)


if __name__ == "__main__":
    unittest.main()
