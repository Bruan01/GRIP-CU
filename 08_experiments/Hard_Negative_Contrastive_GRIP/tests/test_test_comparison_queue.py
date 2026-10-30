"""CPU-only checks for test queue ordering, locks and winner reuse."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import queue_test_comparison as queue


def test_queue_serializes_and_reuses_selected_directory(tmp_path, monkeypatch):
    search = tmp_path / "search"
    records = [{"number": n, "state": "COMPLETE", "params": {"method": "soft_mix", "lambda_scale": 1, "epochs": 3},
                "validation_em": 0.89} for n in queue.ORDER]
    queue.hpo.write_json(search / "trials.json", records)
    queue.hpo.write_json(search / "best.json", {"finalized": True, "best": records[0]})
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(sys, "argv", ["queue", "--search_dir", str(search), "--eval_file", str(tmp_path / "data"),
                                     "--model_cache_dir", str(tmp_path / "cache")])
    calls = []

    def run(command, **kwargs):
        import fcntl
        number = int(command[command.index("--trial_number") + 1])
        output = Path(command[command.index("--output_dir") + 1])
        calls.append(number)
        assert "--wait_for_pipeline" in command
        assert kwargs["pass_fds"]
        with (search / "test_comparison/queue.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if number == 1:
            assert output == search / "selected_test"
        queue.hpo.write_json(output / "test_config.json", {"selection": {"number": number}, "test_count": 3})
        queue.hpo.write_json(output / "summary.json", {"test": {"count": 3, "official_harness_count": 2,
                                                                "official_harness_em": 0.8}})

    monkeypatch.setattr(queue.subprocess, "run", run)
    queue.main()
    assert calls == [1, 3, 2, 4]
    result = queue.hpo.read_json(search / "test_comparison/comparison.json")
    assert result["status"] == "COMPLETE"
    assert [r["number"] for r in result["rows"] if r["validation_selected"]] == [1]


def test_partial_summary_is_rejected(tmp_path):
    records = [{"number": n, "params": {}, "validation_em": 0.9} for n in queue.ORDER]
    output = tmp_path / "selected_test"
    queue.hpo.write_json(output / "test_config.json", {"selection": {"number": 1}, "test_count": 3})
    queue.hpo.write_json(output / "summary.json", {"test": {"count": 2, "official_harness_count": 1}})
    with pytest.raises(ValueError, match="Incomplete/mismatched"):
        queue.publish(tmp_path, records, 1, "RUNNING")
