"""CPU-only tests for fixed scheduling, adoption, deadline and restart safety."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("validation_budget", SCRIPTS / "search_validation_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
hpo = budget.hpo


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(hpo, "EXPECTED_VALIDATION_COUNT", 3)
    root = tmp_path / "paper"
    root.mkdir()
    result = argparse.Namespace(input_file=root / "tasks.json", eval_file=root / "eval.json",
        calibrated_manifest=root / "calibrated.jsonl", raw_dir=root / "raw", s1_adapter=root / "s1",
        search_dir=root / "budget", source_search_dir=root / "old", pipeline_root=root,
        model_cache_dir=root / "cache", seed=2026, source_supervisor_pid=900001,
        source_training_pid=900002, started_at=datetime.now(timezone.utc).isoformat(), budget_hours=60)
    for path in hpo.required_files(result):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    hpo.write_json(result.eval_file, {"recurrent_questions": [
        {"question_id": f"nell23k:validation:{i}", "split": "validation"} for i in range(3)
    ] + [{"question_id": "nell23k:test:0", "split": "test"}]})
    result.calibrated_manifest.write_text(json.dumps({
        "question_id": "qa:0", "positive_relation": "gold", "negative_relations": ["a", "b"],
        "hard_negative_relations": ["a"], "uniform_negative_relations": ["b"],
        "negative_source": "shared_valid_pool_calibrated", "shared_pool": "train_only_valid_negatives",
        "lambda_candidate": 0.7}) + "\n")
    source = budget.source_trial(result)
    source.mkdir(parents=True)
    child = argparse.Namespace(**vars(result))
    child.eval_file = result.source_search_dir / "validation_only.json"
    hpo.write_json(child.eval_file, hpo.validation_record(result.eval_file))
    hpo.write_json(source / "trial.json", {"params": budget.ANCHOR,
                                         "command": hpo.training_command(child, budget.ANCHOR, source)})
    hpo.freeze_manifest(result.calibrated_manifest, source / "negatives.jsonl", budget.ANCHOR)
    hpo.write_json(result.source_search_dir / "search_config.json", {
        "seed": 2026, "files": {str(result.input_file): hpo.sha256(result.input_file)}})
    summary(source, 0.99)  # Anchor wins overall, but must not choose the short-screen confirmation.
    return result


def summary(directory, value):
    hpo.write_json(directory / "listed/summary.json", {"validation": {
        "count": 3, "official_harness_count": 2, "official_harness_em": value}})


def test_fixed_schedule_selects_best_comparable_screen_and_preserves_source(args):
    original = (args.source_search_dir / "search_config.json").read_bytes()
    calls = []

    def execute(command, directory):
        calls.append(command)
        number = int(directory.name.rsplit("_", 1)[-1])
        summary(directory, {1: 0.81, 2: 0.84, 3: 0.84, 4: 0.79, 5: 0.86}[number])
        assert "test" not in command
        assert command[command.index("--save_steps") + 1] == "10"

    budget.verify_source(args)
    budget.run_search(args, execute=execute, adopt=lambda _: None)
    records = hpo.read_json(args.search_dir / "trials.json")
    assert len(calls) == 5 and len(records) == 6
    assert sum(r["params"]["epochs"] for r in records) == 27
    assert records[-1]["params"] == {"method": "soft_mix", "lambda_scale": 0.5, "epochs": 5}
    selected = hpo.read_json(args.search_dir / "best.json")
    assert selected["best"]["number"] == 0 and selected["finalized"]
    assert original == (args.source_search_dir / "search_config.json").read_bytes()
    budget.run_search(args, execute=lambda *_: pytest.fail("Complete run must not restart"),
                      adopt=lambda _: pytest.fail("Complete run must not touch source processes"))


def test_failure_resumes_same_trial_and_does_not_extend_budget(args):
    def fail(command, directory):
        checkpoint = directory / "trainer_listed/checkpoint-10"
        checkpoint.mkdir(parents=True)
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        budget.run_search(args, execute=fail, adopt=lambda _: None)
    first_config = (args.search_dir / "search_config.json").read_bytes()
    first_manifest = (args.search_dir / "trial_001/negatives.jsonl").read_bytes()
    calls = []

    def resume(command, directory):
        calls.append(directory.name)
        if directory.name == "trial_001":
            assert (directory / "trainer_listed/checkpoint-10").exists()
            assert (directory / "negatives.jsonl").read_bytes() == first_manifest
        summary(directory, 0.8)

    budget.run_search(args, execute=resume, adopt=lambda _: None)
    assert calls == [f"trial_{i:03d}" for i in range(1, 6)]
    assert (args.search_dir / "search_config.json").read_bytes() == first_config


def test_deadline_excludes_partial_results_and_restart_keeps_original_deadline(args, monkeypatch):
    def timeout(command, directory):
        summary(directory, 1.0)  # Even an existing summary cannot count after subprocess timeout.
        raise subprocess.TimeoutExpired(command, timeout=1)

    budget.run_search(args, execute=timeout, adopt=lambda _: None)
    result = hpo.read_json(args.search_dir / "best.json")
    assert result["status"] == "BUDGET_EXHAUSTED"
    assert result["best"]["number"] == 0
    assert hpo.read_json(args.search_dir / "trials.json")[1]["state"] == "BUDGET_EXHAUSTED"
    monkeypatch.setattr(budget.time, "time", lambda: budget.deadline(args) + 1)
    budget.run_search(args, execute=lambda *_: pytest.fail("Deadline must hold across restart"),
                      adopt=lambda _: pytest.fail("Expired run must not signal processes"))


def test_supervisor_only_paused_while_training_continues(args, monkeypatch):
    supervisor = {"pid": args.source_supervisor_pid, "command": [str(Path(hpo.__file__).resolve()),
                    "--search_dir", str(args.source_search_dir)], "start_ticks": "1"}
    training = {"pid": args.source_training_pid,
                "command": hpo.read_json(budget.source_trial(args) / "trial.json")["command"], "start_ticks": "2"}
    identities = {supervisor["pid"]: supervisor, training["pid"]: training}
    monkeypatch.setattr(budget, "identity", lambda pid: identities.get(pid))
    signals = []
    monkeypatch.setattr(budget.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(budget.time, "sleep", lambda _: identities.pop(training["pid"], None))
    budget.handoff(args)
    assert signals == [(supervisor["pid"], signal.SIGSTOP),
                       (supervisor["pid"], signal.SIGTERM), (supervisor["pid"], signal.SIGCONT)]


def test_rejects_source_mutation_before_signaling(args):
    args.input_file.write_text("changed")
    with pytest.raises(ValueError, match="Source input/code changed"):
        budget.verify_source(args)


def test_budget_lock_excludes_duplicate_runner(args):
    import fcntl
    args.search_dir.mkdir()
    with (args.search_dir / "search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            budget.run_search(args, adopt=lambda _: pytest.fail("Busy runner must not signal source"))


def test_incomplete_anchor_recovers_original_checkpoint(args):
    (budget.source_trial(args) / "listed/summary.json").unlink()
    checkpoint = budget.source_trial(args) / "trainer_listed/checkpoint-10"
    checkpoint.mkdir(parents=True)
    calls = []

    def execute(command, directory):
        calls.append(directory)
        if directory == budget.source_trial(args):
            assert checkpoint.exists() and "--no_resume" not in command
            assert command[command.index("--involve_qa_epochs") + 1] == "10"
        summary(directory, 0.8)

    budget.run_search(args, execute=execute, adopt=lambda _: None)
    assert calls[0] == budget.source_trial(args) and len(calls) == 6
    assert hpo.read_json(args.search_dir / "best.json")["status"] == "COMPLETE"


def test_frozen_budget_cannot_be_extended_on_restart(args):
    budget.run_search(args, execute=lambda _, directory: summary(directory, 0.8), adopt=lambda _: None)
    args.budget_hours = 70
    with pytest.raises(ValueError, match="frozen artifact differs"):
        budget.run_search(args, adopt=lambda _: pytest.fail("Mutation must not signal source"))


def test_reused_pid_never_receives_signal(monkeypatch):
    saved = {"pid": 1234, "command": ["old"], "start_ticks": "1"}
    monkeypatch.setattr(budget, "identity", lambda _: dict(saved, start_ticks="2"))
    monkeypatch.setattr(budget.os, "kill", lambda *_: pytest.fail("Reused PID must not be signaled"))
    budget.checked_signal(saved, signal.SIGSTOP)


def test_recovery_preserves_original_config_and_cutoff_on_restart(args, monkeypatch):
    import resume_validation_budget as recovery
    monkeypatch.setattr(recovery, "budget", budget)
    budget.run_search(args, execute=lambda _, directory: summary(directory, 0.8), adopt=lambda _: None)
    config_path = args.search_dir / "search_config.json"
    before = config_path.read_bytes()
    original, record = recovery.recovery_record(args, 18)
    hpo.write_json(args.search_dir / "recovery_config.json", record)
    monkeypatch.setattr(recovery.time, "time", lambda: 9999999999)
    assert recovery.recovery_record(args, 18) == (original, record)
    assert config_path.read_bytes() == before
    with pytest.raises(ValueError, match="Recovery configuration differs"):
        recovery.recovery_record(args, 20)


def test_recovery_rejects_modified_inputs(args, monkeypatch):
    import resume_validation_budget as recovery
    monkeypatch.setattr(recovery, "budget", budget)
    budget.run_search(args, execute=lambda _, directory: summary(directory, 0.8), adopt=lambda _: None)
    args.input_file.write_text("changed")
    with pytest.raises(ValueError, match="configuration/source hashes changed"):
        recovery.recovery_record(args, 18)
    assert not (args.search_dir / "recovery_config.json").exists()


def test_child_locks_and_timeout_are_enforced(args, monkeypatch):
    def execute(command, *, check, stdout, stderr, pass_fds, timeout):
        import fcntl
        assert check and len(pass_fds) == 2 and 0 < timeout <= 60 * 3600
        for path in (args.pipeline_root / "pipeline.lock", args.search_dir / "search.lock"):
            with path.open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary(Path(command[command.index("--output_dir") + 1]), 0.8)
    monkeypatch.setattr(budget.subprocess, "run", execute)
    budget.run_search(args, adopt=lambda _: None)
