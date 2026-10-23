from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/build_nell23k_train_ref.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_nell23k_train_ref", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_ref_reads_only_train_and_keeps_last_line(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "entity2text.json").write_text(json.dumps({e: {} for e in "abcdz"}))
    rels = [f"r{i}" for i in range(12)]
    lines = [f"a {rels[i]} b" for i in range(11)] + ["c r11 d"]
    (raw / "train.txt").write_text("\n".join(lines))  # no trailing newline
    (raw / "test.txt").write_text("z r0 a\n")
    record, manifest = _module().build_record(raw, seed=2026)
    assert len(record["graph"]["edge_list"]) == 12
    assert manifest["official_process_py_drops_last_train_line"] is True
    assert manifest["nodes"] == 5 and manifest["nodes_in_train_edges"] == 4
    assert record["answers"] == rels
    for (question, _), answer in zip(record["questions"], record["answers"]):
        candidates = question.split("candidate answers: ")[1].rstrip(".").split("; ")
        assert len(candidates) == 10 and answer in candidates
