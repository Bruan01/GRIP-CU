import json

from hard_negative_grip.fb_pairs import (
    fb_known_pair_relations,
    fb_question_entity_pairs,
    load_fb_entity_text,
)
from hard_negative_grip.offline_scoring import parse_matchable_relation_qa
from hard_negative_grip.task_file import train_relation_alias_index


def _qa(span: str, gold: str) -> str:
    return (
        "<|im_start|>user\nGiven the context graph titled fb15k237_2, please answer the "
        f"following question: what is the relation between {span}? Response in the "
        "following format:<answer>[answer]</answer><|im_end|>\n<|im_start|>assistant\n"
        f"<answer>{gold}</answer><|im_end|>\n"
    )


def _raw(tmp_path):
    (tmp_path / "entity2wikidata.json").write_text(json.dumps({
        "/m/a": {"label": "Tom and Jerry", "description": "cartoon"},
        "/m/b": {"label": "Georgia", "description": "country in Eastern Europe"},
        "/m/c": {"label": "Georgia", "description": "country in Eastern Europe"},
    }))
    (tmp_path / "train.txt").write_text(
        "/m/a\t/r/x\t/m/b\n/m/a\t/r/y\t/m/c\n/m/a\t/r/z\t/m/u\n"
    )
    return {"/r/x": "Release Region", "/r/y": "Filmed In", "/r/z": "Award Nominee"}


def test_pair_split_handles_and_inside_entity_and_unknown(tmp_path):
    rel = _raw(tmp_path)
    entity_text = load_fb_entity_text(tmp_path)
    assert entity_text["/m/u"] == "Unknown"
    texts = set(entity_text.values())
    head, tail = "Tom and Jerry, cartoon", "Georgia, country in Eastern Europe"
    assert fb_question_entity_pairs(_qa(f"node {head} and {tail}", "x"), texts) == [(head, tail)]
    assert fb_question_entity_pairs(_qa(f"{head} and Unknown", "x"), texts) == [(head, "Unknown")]
    known = fb_known_pair_relations(
        tmp_path, entity_text=entity_text, relation_text=rel, splits=("train",)
    )
    # Two MIDs share the Georgia text: the filter unions their relations.
    assert known[(head, tail)] == {"Release Region", "Filmed In"}


def test_matchable_accepts_multiword_fb_relation(tmp_path):
    alias = train_relation_alias_index(["Release Region", "Filmed In"])
    text = _qa("A, x and B, y", "Release Region")
    item = parse_matchable_relation_qa(0, text, alias, pair_fn=lambda _t: ("A, x", "B, y"))
    assert item is not None
    assert item["matched_train_relation"] == "Release Region"
    assert item["entity_pair"] == ("A, x", "B, y")
