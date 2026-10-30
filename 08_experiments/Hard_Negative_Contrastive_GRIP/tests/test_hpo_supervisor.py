"""CPU checks for campaign selection, restart, and stale evaluation safety."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supervise_nell_fb_hpo as guard


def make_adapter(path, value=b"weights"):
    path.mkdir(parents=True, exist_ok=True)
    guard.hpo.write_json(path / "adapter_config.json", {"r": 24, "lora_alpha": 48})
    (path / "adapter_model.safetensors").write_bytes(value)


class EvaluationCacheTests(unittest.TestCase):
    def test_mutated_adapter_cannot_reuse_old_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter, source, output = root / "adapter", root / "valid.json", root / "eval"
            make_adapter(adapter)
            guard.hpo.write_json(source, {"questions": ["q"], "answers": ["a"]})
            output.mkdir()
            guard.hpo.write_json(output / "summary.json", {
                "adapter": str(adapter), "eval_file": str(source), "all": {"count": 1, "em": 1.0}})
            (output / "predictions.jsonl").write_text('{}\n')
            self.assertEqual(guard.evaluate(adapter, source, output, "label"), 1.0)
            (adapter / "adapter_model.safetensors").write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                guard.evaluate(adapter, source, output, "label")

    def test_incomplete_predictions_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter, source, output = root / "adapter", root / "valid.json", root / "eval"
            make_adapter(adapter)
            guard.hpo.write_json(source, {"questions": ["q", "q2"], "answers": ["a", "b"]})
            output.mkdir()
            guard.hpo.write_json(output / "summary.json", {
                "adapter": str(adapter), "eval_file": str(source), "all": {"count": 2, "em": 1.0}})
            (output / "predictions.jsonl").write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                guard.evaluate(adapter, source, output, "label")

    def test_fb_rank_and_schedule_follow_actual_stage1_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_adapter(root / "s1")
            with patch.object(guard.hpo, "run_logged") as run:
                guard.hpo.train_rung(python=guard.PYTHON, dataset="fb15k237_2",
                    paths={"input": root / "tasks.json", "valid": root / "valid.json",
                           "raw": root / "raw", "expected_listed": 3109, "epochs": 5,
                           "b1_adapter": root / "b1", "entity_text_file": root / "entities.txt"},
                    adapter=root / "s1", sampler=root / "bank", model_cache=root,
                    trial_dir=root / "trial", trial_number=0, rung_step=16,
                    fresh=False, resume_checkpoint=root / "checkpoint-8",
                    params={"lambda_candidate": 1, "temperature": 1,
                            "learning_rate": 0.001, "weight_decay": 0.0001})
            environment = run.call_args.args[1]
            self.assertEqual(environment["TRAIN_LORA_R"], "24")
            self.assertEqual(environment["INVOLVE_QA_EPOCHS"], "5")
            self.assertEqual(environment["NO_RESUME"], "0")
            self.assertEqual(environment["B1_ADAPTER"], str(root / "b1"))
            self.assertEqual(environment["RELATION_TEXT_FILE"], str(root / "raw/rel2text.json"))
            self.assertEqual(environment["ENTITY_TEXT_FILE"], str(root / "entities.txt"))


class SearchDecisionTests(unittest.TestCase):
    def run_search(self, root, full_candidate_em, interrupt=False, spec_patch=None):
        work, s1, b1, _ = guard.fb_paths(root)
        make_adapter(s1)
        make_adapter(b1)
        valid, test = root / "valid.json", root / "test.json"
        for source in (valid, test):
            guard.hpo.write_json(source, {"questions": ["q"], "answers": ["a"]})
        paths = {"input": root / "tasks.json", "valid": valid, "test": test,
                 "raw": root / "raw", "expected_listed": 3109}
        spec = None
        if spec_patch is not None:
            spec = {"dataset": "codexm", "s1_adapter": str(s1), "b1_adapter": str(b1),
                "sampler": str(work / "assets/k9"), "search_dir": str(work / "search"),
                **{key: str(paths[key]) for key in ["input", "valid", "test", "raw"]},
                "epochs": 5, "expected_listed": 3109, "paper_reference": 0.9442,
                "rungs": [8, 16, 32], "trial_budget": 2, **spec_patch}
        evaluations = []

        def fake_train(**kwargs):
            folder, step = kwargs["trial_dir"], kwargs["rung_step"]
            make_adapter(folder / "train_run/listed/adapter", f"step-{step}".encode())
            checkpoint = folder / f"train_run/trainer_listed/checkpoint-{step}"
            guard.hpo.write_json(checkpoint / "trainer_state.json", {"global_step": step})

        def fake_evaluate(adapter, source, output, label):
            evaluations.append((adapter, source))
            if interrupt and len(evaluations) == 2:
                raise InterruptedError("simulated interruption after saving rung")
            if source == test:
                self.assertTrue((work / "search/selection.json").exists())
                return 0.9
            if source == valid:
                return 0.8 if adapter == b1 else full_candidate_em
            return 0.8 if adapter == b1 else 0.85

        with patch.object(guard.hpo, "dataset_paths", return_value=paths), \
             patch.object(guard.hpo, "validate_assets"), \
             patch.object(guard.hpo, "train_rung", side_effect=fake_train), \
             patch.object(guard.hpo, "write_study_exports"), \
             patch.object(guard, "evaluate", side_effect=fake_evaluate):
            guard.fb_search(root, spec)
        return evaluations

    def test_screen_gain_without_full_validation_gain_does_not_open_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = self.run_search(root, 0.79)
            self.assertFalse(any(source.name == "test.json" for _, source in calls))
            self.assertTrue((root / "fb15k237_2/search/no_go.json").exists())

    def test_selection_is_frozen_before_test_and_selected_weights_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = self.run_search(root, 0.86)
            tests = [(adapter, source) for adapter, source in calls if source.name == "test.json"]
            self.assertEqual(len(tests), 2)  # matched B1 plus one selected candidate
            selection = guard.read(root / "fb15k237_2/search/selection.json")
            self.assertEqual(str(tests[1][0]), selection["winner"]["adapter"])
            self.assertEqual(tests[1][0].joinpath("adapter_model.safetensors").read_bytes(), b"step-8")

    def test_interrupted_trial_is_resumed_without_spending_an_extra_trial(self):
        import optuna
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(InterruptedError):
                self.run_search(root, 0.86, interrupt=True)
            calls = self.run_search(root, 0.86)
            study = optuna.load_study(study_name="fb_sccl_validation_checkpoint_v1",
                storage=f"sqlite:///{root / 'fb15k237_2/search/study.sqlite3'}")
            self.assertEqual(len(study.trials), 6)
            self.assertFalse(any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials))
            self.assertTrue(any(source.name == "test.json" for _, source in calls))

    def test_partial_resource_turn_resumes_same_trials_and_defers_test(self):
        import optuna
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = self.run_search(root, 0.86, spec_patch={"resource_cap": 16, "search_only": True})
            self.assertFalse(any(source.name == "test.json" for _, source in calls))
            storage = f"sqlite:///{root / 'fb15k237_2/search/study.sqlite3'}"
            study = optuna.load_study(study_name="codexm_sccl_validation_checkpoint_v1", storage=storage)
            self.assertEqual(len(study.trials), 2)
            self.assertTrue(all(t.state == optuna.trial.TrialState.RUNNING for t in study.trials))
            calls = self.run_search(root, 0.86, spec_patch={"resource_cap": 32, "search_only": False})
            study = optuna.load_study(study_name="codexm_sccl_validation_checkpoint_v1", storage=storage)
            self.assertEqual(len(study.trials), 2)
            self.assertFalse(any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials))
            self.assertEqual(sum(source.name == "test.json" for _, source in calls), 2)

    def test_capped_rounds_do_not_repeat_startup_hyperparameters(self):
        import optuna
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for budget in (2, 4, 6):
                self.run_search(root, 0.86, spec_patch={"resource_cap": 16, "search_only": True,
                    "trial_budget": budget, "prune_after": 32})
            study = optuna.load_study(study_name="codexm_sccl_validation_checkpoint_v1",
                storage=f"sqlite:///{root / 'fb15k237_2/search/study.sqlite3'}")
            self.assertEqual(len(study.trials), 6)
            self.assertEqual(len({tuple(sorted(trial.params.items())) for trial in study.trials}), 6)


if __name__ == "__main__":
    unittest.main()
