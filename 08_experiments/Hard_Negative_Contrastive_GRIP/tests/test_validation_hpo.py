"""CPU-only HPO contract and recovery tests; no model or CUDA is loaded."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/search_validation_hparams.py"
spec = importlib.util.spec_from_file_location("validation_hpo", SCRIPT)
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)


def manifest_row():
    return {"question_id": "qa:0", "positive_relation": "gold", "negative_relations": ["a", "b"],
            "hard_negative_relations": ["a"], "uniform_negative_relations": ["b"],
            "negative_source": "shared_valid_pool_calibrated", "shared_pool": "train_only_valid_negatives",
            "lambda_candidate": 0.7, "concentration_c_q": 0.6}


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(hpo, "EXPECTED_VALIDATION_COUNT", 3)
    root = tmp_path / "paper"
    root.mkdir()
    result = argparse.Namespace(input_file=root / "tasks.json", eval_file=root / "eval.json",
        calibrated_manifest=root / "calibrated.jsonl", raw_dir=root / "raw", s1_adapter=root / "s1",
        search_dir=root / "hpo", pipeline_root=root, model_cache_dir=root / "cache", seed=2026, n_trials=6)
    for path in hpo.required_files(result):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    hpo.write_json(result.eval_file, {"recurrent_questions": [
        {"question_id": f"nell23k:validation:{i}", "split": "validation"} for i in range(3)
    ] + [{"question_id": "nell23k:test:0", "split": "test"}]})
    result.calibrated_manifest.write_text(json.dumps(manifest_row()) + "\n")
    return result


def test_full_validation_required(args, monkeypatch):
    pytest.importorskip("optuna")
    monkeypatch.setattr(hpo, "EXPECTED_VALIDATION_COUNT", 4951)
    with pytest.raises(ValueError, match="4951 questions"):
        hpo.run_search(args, execute=lambda *_: pytest.fail("Partial validation must not train"))


@pytest.mark.parametrize("method, expected", [("soft_mix", 0.5), ("calibrated", 0.35)])
def test_weights_preserve_frozen_negative_lists(args, method, expected):
    params = {"method": method, "lambda_scale": 0.5, "epochs": 3}
    output = args.pipeline_root / "trial.jsonl"
    hpo.freeze_manifest(args.calibrated_manifest, output, params)
    row = json.loads(output.read_text())
    assert row["lambda_candidate"] == pytest.approx(expected)
    for key in ("negative_relations", "hard_negative_relations", "uniform_negative_relations", "question_id"):
        assert row[key] == manifest_row()[key]
    hpo.freeze_manifest(args.calibrated_manifest, output, params)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        hpo.freeze_manifest(args.calibrated_manifest, output, dict(params, lambda_scale=0.2))


def test_rejects_fixed_lambda_source(args):
    row = dict(manifest_row(), variant="soft_mix")
    args.calibrated_manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="calibrated manifest"):
        hpo.freeze_manifest(args.calibrated_manifest, args.pipeline_root / "out.jsonl",
                            {"method": "calibrated", "lambda_scale": 1})


def test_validation_only_command_and_completed_adapter(args):
    params = {"epochs": 3}
    directory = args.search_dir / "trial_000"
    command = hpo.training_command(args, params, directory)
    assert command[command.index("--eval_splits") + 1] == "validation"
    assert "test" not in command and "--no_resume" not in command
    assert "--s2_stop_loss_threshold" not in command
    assert command[command.index("--save_steps") + 1] == "10"
    adapter = directory / "listed/adapter"
    adapter.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (adapter / name).touch()
    assert "--skip_train" in hpo.training_command(args, params, directory)
    record = hpo.validation_record(args.eval_file)
    assert len(record["recurrent_questions"]) == 3
    assert all(row["split"] == "validation" for row in record["recurrent_questions"])


def test_validation_rejects_partial_or_test_summary(tmp_path):
    path = tmp_path / "summary.json"
    summary = {"validation": {"count": 3, "official_harness_count": 2, "official_harness_em": 0.5}}
    hpo.write_json(path, summary)
    assert hpo.validation_value(path, 3) == 0.5
    with pytest.raises(ValueError, match="Incomplete"):
        hpo.validation_value(path, 4)
    hpo.write_json(path, dict(summary, test={"official_em": 0.9}))
    with pytest.raises(ValueError, match="test"):
        hpo.validation_value(path, 3)


def fake_training(command, directory):
    assert "test" not in command
    validation_path = command[command.index("--eval_file") + 1]
    assert all(row["split"] == "validation" for row in hpo.read_json(validation_path)["recurrent_questions"])
    hpo.write_json(directory / "listed/summary.json", {"validation": {
        "count": 3, "official_harness_count": 2, "official_harness_em": 0.8}})


def test_interrupted_trial_resumes_then_respects_total_budget(args):
    optuna = pytest.importorskip("optuna")
    calls = []

    def interrupt(command, directory):
        calls.append(directory)
        # Simulate a checkpoint produced before a process failure.
        checkpoint = directory / "trainer_listed/checkpoint-10"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        hpo.run_search(args, execute=interrupt)
    trial_before = hpo.read_json(calls[0] / "trial.json")
    manifest_before = (calls[0] / "negatives.jsonl").read_bytes()

    def resume(command, directory):
        if not calls or directory != calls[-1]:
            calls.append(directory)
        if directory == calls[0]:
            assert (directory / "trainer_listed/checkpoint-10/trainer_state.json").is_file()
            assert hpo.read_json(directory / "trial.json")["params"] == trial_before["params"]
            assert (directory / "negatives.jsonl").read_bytes() == manifest_before
        fake_training(command, directory)

    hpo.run_search(args, execute=resume)
    assert len(calls) == args.n_trials
    study = optuna.load_study(study_name="nell23k_validation_s2",
                              storage=f"sqlite:///{args.search_dir / 'study.sqlite3'}")
    assert len(study.trials) == args.n_trials
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert hpo.read_json(args.search_dir / "best.json")["best"]["number"] == 0
    hpo.run_search(args, execute=lambda *_: pytest.fail("Finished study must not start new work"))
    assert (args.search_dir / "VARIANTS.md").is_file()


def test_child_inherits_workflow_locks(args, monkeypatch):
    pytest.importorskip("optuna")
    args.n_trials = 4

    def run(command, *, check, stdout, stderr, pass_fds):
        assert check and len(pass_fds) == 2
        for path in (args.pipeline_root / "pipeline.lock", args.search_dir / "search.lock"):
            with path.open("a") as handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = Path(command[command.index("--output_dir") + 1])
        fake_training(command, directory)

    monkeypatch.setattr(hpo.subprocess, "run", run)
    hpo.run_search(args)


def test_config_mutation_is_rejected(args):
    pytest.importorskip("optuna")
    args.n_trials = 4
    hpo.run_search(args, execute=fake_training)
    args.input_file.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="frozen artifact differs"):
        hpo.run_search(args, execute=fake_training)


def test_paper_lock_blocks_search(args):
    pytest.importorskip("optuna")
    with (args.pipeline_root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="lock busy"):
            hpo.run_search(args, execute=lambda *_: pytest.fail("GPU must remain untouched"))


def test_dry_run_needs_no_prerequisites(tmp_path, capsys):
    argv = []
    for name in ("input_file", "eval_file", "s1_adapter", "calibrated_manifest", "raw_dir",
                 "model_cache_dir", "search_dir", "pipeline_root"):
        argv.extend(["--" + name, str(tmp_path / name)])
    hpo.main(argv + ["--dry_run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["test_evaluation"] is False
    assert plan["missing_prerequisites"]
    assert not (tmp_path / "search_dir").exists()
