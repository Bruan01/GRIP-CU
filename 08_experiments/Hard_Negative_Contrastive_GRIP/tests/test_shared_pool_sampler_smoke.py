from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HNG = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from compare_shared_pool_sampler_runs import main as compare_main  # noqa: E402
from hard_negative_grip.shared_pool_samplers import (  # noqa: E402
    compact_wiring_payload,
    verify_listed_manifest_wiring,
)
from subset_matchable_task_qa import main as subset_main  # noqa: E402


def _summary(em: float, n: int = 96) -> dict:
    return {
        "all": {"count": n, "em": em},
        "wrong_in_list": 1,
        "wrong_out_of_list": 2,
        "test": {"count": 64, "em": em},
        "validation": {"count": 32, "em": em},
    }


def test_compare_shared_pool_sampler_runs(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    for name, em in (("random_k", 0.50), ("top_k_hard", 0.60), ("coverage_adaptive_k", 0.55)):
        listed = run_dir / name / "listed"
        listed.mkdir(parents=True)
        (listed / "summary.json").write_text(json.dumps(_summary(em)), encoding="utf-8")
    b1 = run_dir / "random_k" / "b1"
    b1.mkdir(parents=True)
    (b1 / "summary.json").write_text(json.dumps(_summary(0.40)), encoding="utf-8")
    output = run_dir / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        ["compare_shared_pool_sampler_runs.py", "--run_dir", str(run_dir), "--output", str(output)],
    )
    compare_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["deltas"]["top_k_hard_minus_random_k"] == pytest.approx(0.10)
    assert payload["variants"]["random_k"]["listed_minus_b1_em"] == pytest.approx(0.10)
    assert "64-QA / 10-step" in payload["note"]
    assert "12014 QA" in payload["note"]


def test_retired_shared_pool_sampler_smoke_script_exits() -> None:
    result = subprocess.run(
        ["bash", str(HNG / "configs/run_shared_pool_sampler_smoke.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "64-QA / 10-step" in result.stderr
    assert "run_shared_pool_random_k_full.sh" in result.stderr


def test_subset_matchable_task_qa_keeps_original_ids(tmp_path: Path, monkeypatch) -> None:
    task = {
        "context_samples": ["ctx"],
        "qa_samples": [
            "qa0",
            "qa1",
            "qa2",
        ],
    }
    input_path = tmp_path / "task.json"
    input_path.write_text(json.dumps(task), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "question_id": "task_qa:2",
            "positive_relation": "concept:gold",
            "hard_negative_relations": ["concept:hard"],
            "uniform_negative_relations": [],
            "negative_relations": ["concept:hard"],
        },
        {
            "question_id": "task_qa:0",
            "positive_relation": "concept:gold",
            "hard_negative_relations": ["concept:easy"],
            "uniform_negative_relations": [],
            "negative_relations": ["concept:easy"],
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "subset.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "subset_matchable_task_qa.py",
            "--input",
            str(input_path),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--max-samples",
            "2",
            "--seed",
            "2026",
        ],
    )
    subset_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["original_question_ids"] == ["task_qa:0", "task_qa:2"]
    assert payload["qa_samples"] == ["qa0", "qa2"]
    assert payload["subset_n_qa"] == 2


def _qa(question: str, gold: str) -> str:
    return (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n<answer>{gold}</answer><|im_end|>\n"
    )


def test_verify_listed_manifest_wiring_keeps_original_ids(tmp_path: Path) -> None:
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "context_samples": ["ctx"],
                "qa_samples": [
                    _qa("What is the relation between e and f?", "concept:gold")
                ],
                "original_question_ids": ["task_qa:2"],
                "original_qa_indices": [2],
                "subset_n_qa": 1,
            }
        ),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.txt").write_text(
        "a concept:gold b\nc concept:other d\ne concept:hard f\n",
        encoding="utf-8",
    )
    random_k = tmp_path / "random_k.jsonl"
    top_k = tmp_path / "top_k_hard.jsonl"
    adaptive = tmp_path / "coverage_adaptive_k.jsonl"
    random_k.write_text(
        json.dumps(
            {
                "question_id": "task_qa:2",
                "positive_relation": "concept:gold",
                "hard_negative_relations": [],
                "uniform_negative_relations": ["concept:other"],
                "negative_relations": ["concept:other"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    top_k.write_text(
        json.dumps(
            {
                "question_id": "task_qa:2",
                "positive_relation": "concept:gold",
                "hard_negative_relations": ["concept:hard"],
                "uniform_negative_relations": [],
                "negative_relations": ["concept:hard"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adaptive.write_text(
        json.dumps(
            {
                "question_id": "task_qa:2",
                "positive_relation": "concept:gold",
                "hard_negative_relations": ["concept:hard", "concept:other"],
                "uniform_negative_relations": [],
                "negative_relations": ["concept:hard", "concept:other"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = verify_listed_manifest_wiring(
        task_path=task,
        manifest_paths={
            "random_k": random_k,
            "top_k_hard": top_k,
            "coverage_adaptive_k": adaptive,
        },
        raw_dir=raw_dir,
        expected_n=1,
        expected_listed=1,
    )
    assert result["question_ids"] == ["task_qa:2"]
    assert result["n_listed"] == 1
    assert result["variants"]["random_k"]["k_min"] == 1
    assert result["variants"]["coverage_adaptive_k"]["k_max"] == 2
    assert result["_negatives_by_id"]["top_k_hard"]["task_qa:2"] == ["concept:hard"]


def test_verify_full_task_file_uses_positional_ids(tmp_path: Path) -> None:
    extra = (
        "<|im_start|>user\nIs this a yes-no question?<|im_end|>\n"
        "<|im_start|>assistant\n<answer>yes</answer><|im_end|>\n"
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "context_samples": ["ctx"],
                "qa_samples": [
                    _qa("What is the relation between e and f?", "concept:gold"),
                    extra,
                ]
                * 1627,
            }
        ),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.txt").write_text(
        "a concept:gold b\nc concept:other d\ne concept:hard f\n",
        encoding="utf-8",
    )
    rows = [
        json.dumps(
            {
                "question_id": f"task_qa:{index}",
                "positive_relation": "concept:gold",
                "hard_negative_relations": [],
                "uniform_negative_relations": ["concept:other"],
                "negative_relations": ["concept:other"],
            }
        )
        for index in range(0, 3254, 2)
    ]
    random_k = tmp_path / "random_k.jsonl"
    random_k.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = verify_listed_manifest_wiring(
        task_path=task,
        manifest_paths={"random_k": random_k},
        raw_dir=raw_dir,
        expected_listed=1627,
        variants=("random_k",),
    )
    assert result["question_id_source"] == "positional"
    assert result["n_qa"] == 3254
    assert result["n_listed"] == 1627
    compact = compact_wiring_payload(result)
    assert "question_ids" not in compact
    assert compact["n_listed"] == 1627
    assert compact["question_id_head"] == ["task_qa:0", "task_qa:1", "task_qa:2"]
    assert "question_ids" not in compact["variants"]["random_k"]
