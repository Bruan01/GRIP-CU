from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.archive_results import (  # noqa: E402
    archive_run,
    default_archive_root,
    should_copy,
    snapshot_run_name,
    unique_snapshot_dir,
)


def test_snapshot_run_name_keeps_nested_runs_apart(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    nested = runs_root / "20260923_offline_confusion_full" / "audit"
    nested.mkdir(parents=True)
    assert snapshot_run_name(nested, runs_root=runs_root) == "20260923_offline_confusion_full__audit"
    assert snapshot_run_name(runs_root / "listed_run", runs_root=runs_root) == "listed_run"


def test_default_archive_root_is_under_results_archive() -> None:
    root = default_archive_root()
    assert root.as_posix().endswith("Hard_Negative_Contrastive_GRIP/results/archive")


def test_unique_snapshot_dir_never_overwrites(tmp_path: Path) -> None:
    first = unique_snapshot_dir(tmp_path, "run_a", "20260927T000000Z")
    first.mkdir(parents=True)
    second = unique_snapshot_dir(tmp_path, "run_a", "20260927T000000Z")
    assert second != first
    assert second.name == "20260927T000000Z-2"


def test_should_copy_skips_weights_and_dumps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "listed" / "adapter").mkdir(parents=True)
    (run_dir / "trainer_listed" / "checkpoint-10").mkdir(parents=True)
    summary = run_dir / "listed" / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    weight = run_dir / "listed" / "adapter" / "adapter_model.safetensors"
    weight.write_bytes(b"x" * 16)
    dump = run_dir / "candidate_scores.jsonl"
    dump.write_text("{}\n", encoding="utf-8")
    ckpt = run_dir / "trainer_listed" / "checkpoint-10" / "trainer_state.json"
    ckpt.write_text("{}", encoding="utf-8")
    tokenizer = run_dir / "listed" / "adapter" / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    assert should_copy(summary, run_dir) is True
    assert should_copy(weight, run_dir) is False
    assert should_copy(dump, run_dir) is False
    assert should_copy(ckpt, run_dir) is False
    assert should_copy(tokenizer, run_dir) is False


def test_archive_run_copies_summaries_not_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260926_shared_pool_random_k_full"
    listed = run_dir / "listed"
    listed.mkdir(parents=True)
    (listed / "adapter").mkdir()
    comparison = {
        "listed": {"all": {"count": 96, "em": 0.90625}},
        "b1": {"all": {"count": 96, "em": 0.8645833333333334}},
        "listed_minus_b1_em": 0.04166666666666663,
    }
    (run_dir / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    (listed / "summary.json").write_text(json.dumps(comparison["listed"]), encoding="utf-8")
    (listed / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (listed / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (listed / "adapter" / "tokenizer.json").write_text("{}", encoding="utf-8")
    archive_root = tmp_path / "archive"
    dest = archive_run(run_dir, archive_root=archive_root, stamp="20260927T001544Z")
    assert dest == archive_root / run_dir.name / "20260927T001544Z"
    assert (dest / "comparison.json").is_file()
    assert (dest / "listed" / "summary.json").is_file()
    assert (dest / "listed" / "adapter" / "adapter_config.json").is_file()
    assert not (dest / "listed" / "adapter" / "adapter_model.safetensors").exists()
    assert not (dest / "listed" / "adapter" / "tokenizer.json").exists()
    again = archive_run(run_dir, archive_root=archive_root, stamp="20260927T001544Z")
    assert again == archive_root / run_dir.name / "20260927T001544Z-2"
    assert dest.is_dir()
    index = (archive_root / "INDEX.md").read_text(encoding="utf-8")
    assert "90.62%" in index or "90.63%" in index
    assert "20260927T001544Z-2" in index


def test_archive_run_nested_dir_stays_in_results_archive(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260923_offline_confusion_full" / "analysis"
    run_dir.mkdir(parents=True)
    (run_dir / "global_statistics.json").write_text("{}", encoding="utf-8")
    archive_root = tmp_path / "archive"
    dest = archive_run(
        run_dir,
        archive_root=archive_root,
        stamp="20260927T010000Z",
        runs_root=runs_root,
    )
    assert dest == archive_root / "20260923_offline_confusion_full__analysis" / "20260927T010000Z"
    assert dest.parent.parent == archive_root
