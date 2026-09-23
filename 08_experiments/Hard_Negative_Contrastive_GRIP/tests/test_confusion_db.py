from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.confusion_db import (  # noqa: E402
    eligible_negative_relations,
    freeze_confusion_db,
    prefix_sha256,
    scorer_metadata,
    select_confusion_negatives,
    validate_score_table,
)


def _qa(question: str, answer: str) -> str:
    return (
        "<|im_start|>user\n"
        f"Given the context graph titled nell23k, please answer the following question: {question}"
        " Response in the following format:<answer>[answer]</answer>\n"
        f"<|im_start|>assistant\n<answer>{answer}</answer>"
    )


def test_validate_score_table_requires_full_official_vocab() -> None:
    order = ["concept:gold", "concept:hard", "concept:easy"]
    scores = [
        {"relation": "concept:hard", "score": 0.9, "rank": 1},
        {"relation": "concept:gold", "score": 0.5, "rank": 2},
        {"relation": "concept:easy", "score": 0.1, "rank": 3},
    ]
    mapped = validate_score_table(scores, order, "concept:gold")
    assert mapped["concept:hard"] == 0.9
    try:
        validate_score_table(scores[:2], order, "concept:gold")
    except ValueError as error:
        assert "198" not in str(error)
        assert "official relations" in str(error)
    else:
        raise AssertionError("incomplete table should fail")


def test_eligible_negatives_drop_gold_and_known_pair_relations() -> None:
    scores = {"concept:gold": 1.0, "concept:true": 0.8, "concept:hard": 0.7}
    assert eligible_negative_relations(
        scores,
        gold="concept:gold",
        known_relations=["concept:true"],
    ) == ["concept:hard"]


def test_select_confusion_negatives_is_frozen_top_score() -> None:
    order = ["concept:gold", "concept:true", "concept:hard", "concept:easy"]
    scores = [
        {"relation": "concept:true", "score": 0.9, "rank": 1},
        {"relation": "concept:hard", "score": 0.8, "rank": 2},
        {"relation": "concept:gold", "score": 0.5, "rank": 3},
        {"relation": "concept:easy", "score": 0.1, "rank": 4},
    ]
    row = {
        "matched_train_relation": "concept:gold",
        "all_candidate_scores": scores,
        "known_pair_relations": ["concept:true"],
    }
    assert select_confusion_negatives(row, order, k=9) == ["concept:hard", "concept:easy"]
    assert select_confusion_negatives(row, order, k=9, hard_k=1) == ["concept:hard"]


def test_freeze_confusion_db_annotates_prefix_hash_and_rejects_gaps(tmp_path: Path) -> None:
    task = tmp_path / "tasks.json"
    texts = [
        _qa("what is the relation between a and b?", "concept:gold"),
        _qa("Is Detroit a city?", "Yes"),
        _qa("what is the relation between c and d?", "not_in_vocab"),
    ]
    task.write_text(json.dumps({"qa_samples": texts}), encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.txt").write_text("a concept:gold b\na concept:true b\n", encoding="utf-8")
    (raw / "valid.txt").write_text("", encoding="utf-8")
    (raw / "test.txt").write_text("", encoding="utf-8")

    scores = [
        {"relation": "concept:gold", "score": 0.2, "rank": 2},
        {"relation": "concept:true", "score": 0.9, "rank": 1},
    ]
    # Rank by (-score, name): true=0.9 first, gold=0.2 second.
    scores = sorted(scores, key=lambda item: (-item["score"], item["relation"]))
    for rank, item in enumerate(scores, start=1):
        item["rank"] = rank
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "question_id": "task_qa:0",
                "positive_relation": "concept:gold",
                "matched_train_relation": "concept:gold",
                "all_candidate_scores": scores,
                "known_pair_relations": ["concept:gold", "concept:true"],
                "entity_pair": ["a", "b"],
                "teacher": "frozen_b1",
                "seed": 2026,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "confusion_db.jsonl"
    metadata = tmp_path / "confusion_db.json"
    summary = freeze_confusion_db(
        task_path=task,
        raw_dir=raw,
        input_jsonl=source,
        output_jsonl=output,
        metadata_path=metadata,
        b1_adapter="frozen/b1/adapter",
        candidate_batch_size=8,
        seed=2026,
    )
    assert summary["rows"] == 1
    assert summary["unmatched_relation_qa"] == 1
    assert summary["non_relation_qa"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["confusion_db"] is True
    assert row["vocab_size"] == 2
    assert row["scorer"] == "normalized_continuation_mean_logprob"
    assert row["scorer_spec"]["includes_eos"] is False
    assert row["prefix_sha256"] == prefix_sha256(
        texts[0][: texts[0].rfind("<answer>") + len("<answer>")]
    )
    assert "prefix_text" not in row
    extra = tmp_path / "extra.jsonl"
    extra.write_text(source.read_text(encoding="utf-8") * 2, encoding="utf-8")
    try:
        freeze_confusion_db(
            task_path=task,
            raw_dir=raw,
            input_jsonl=extra,
            output_jsonl=tmp_path / "bad.jsonl",
            metadata_path=tmp_path / "bad.json",
        )
    except ValueError as error:
        assert "duplicate question_id" in str(error)
    else:
        raise AssertionError("duplicate rows should fail")


def test_scorer_metadata_keeps_temperature_out_of_the_score() -> None:
    meta = scorer_metadata(vocab_size=198, seed=2026, candidate_batch_size=8)
    assert meta["scorer_spec"]["temperature_in_scorer"] is False
    assert meta["vocab_size"] == 198
