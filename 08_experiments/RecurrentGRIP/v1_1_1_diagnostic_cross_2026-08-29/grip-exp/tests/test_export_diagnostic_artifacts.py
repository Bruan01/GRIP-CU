import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "export_diagnostic_cross_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("export_diagnostic_cross_artifacts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DiagnosticArtifactExportTest(unittest.TestCase):
    def _build_run(self, root: Path) -> Path:
        run_dir = root / "run_fixture"
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "input_stats.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text("gpu=fixture\n", encoding="utf-8")

        rows = []
        for train_k in (1, 2):
            manifest_dir = run_dir / f"train_k{train_k}" / "adapters" / "nell23k"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "context_sampling_manifest.json").write_text(
                json.dumps({"selection_sha256": "same", "train_k": train_k}),
                encoding="utf-8",
            )
            for eval_k in (1, 2):
                for control in ("correct", "none"):
                    rows.append(
                        {
                            "graph_id": "nell23k",
                            "question_id": f"q-{train_k}-{eval_k}-{control}",
                            "question": "fixture question",
                            "target": ["r1"],
                            "true_hop": 2,
                            "recurrent_train_k": train_k,
                            "recurrence_k": eval_k,
                            "adapter_id": "nell23k" if control == "correct" else "none",
                            "adapter_control": control,
                            "raw_response": "r1",
                            "response": "r1",
                            "correct": True,
                            "latency_seconds": 0.1,
                            "peak_memory_bytes": 100,
                            "generated_token_count": 2,
                            "ended_with_eos": True,
                            "response_in_candidates": True,
                            "step_pooled_hidden_states": [[1.0, 2.0]] * eval_k,
                            "metadata": {"split": "validation"},
                        }
                    )
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

        audit = {
            "status": "pass",
            "prediction_count": len(rows),
            "selection_sha256": "same",
            "train_depths": [1, 2],
            "eval_depths": [1, 2],
            "adapter_controls": ["correct", "none"],
        }
        (run_dir / "cross_run_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        (analysis_dir / "summary.json").write_text(
            json.dumps({"count": len(rows)}), encoding="utf-8"
        )
        for relative in MODULE.REQUIRED_ANALYSIS_FILES:
            path = analysis_dir / relative
            if not path.exists():
                path.write_text("fixture\n", encoding="utf-8")
        return run_dir

    def test_exports_compact_predictions_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._build_run(root)
            output_dir = root / "export"
            manifest = MODULE.export_artifacts(run_dir, output_dir, root)

            exported_rows = [
                json.loads(line)
                for line in (output_dir / "predictions_audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest["prediction_count"], 8)
            self.assertEqual(len(exported_rows), 8)
            self.assertNotIn("step_pooled_hidden_states", exported_rows[0])
            self.assertTrue((output_dir / "context_manifests" / "train_k1.json").is_file())
            self.assertTrue((output_dir / "analysis" / "state_dynamics.json").is_file())
            self.assertTrue((output_dir / "artifact_manifest.json").is_file())

    def test_rejects_existing_export_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._build_run(root)
            output_dir = root / "export"
            output_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                MODULE.export_artifacts(run_dir, output_dir, root)

    def test_rejects_failed_cross_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._build_run(root)
            audit_path = run_dir / "cross_run_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["status"] = "fail"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status=pass"):
                MODULE.export_artifacts(run_dir, root / "export", root)


if __name__ == "__main__":
    unittest.main()
