from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.confusion_vocab import (  # noqa: E402
    PAIR_FIELDS,
    aggregate_relation_pairs,
    beats_gold,
    confusion_neighbors,
    filter_pairs_for_display,
    gold_key,
    matrix_grid,
    write_confusion_vocab_outputs,
)
from hard_negative_grip.offline_scoring import build_qa_score_rows  # noqa: E402


def _qa(
    qa_id: str,
    gold_surface: str,
    matched: str,
    scores: dict[str, float],
    known: list[str] | None = None,
) -> list[dict]:
    order = list(scores)
    lengths = {rel: 1 for rel in order}
    rows, _summary = build_qa_score_rows(
        qa_id=qa_id,
        question="q",
        head_entity="a",
        tail_entity="b",
        gold_relation=gold_surface,
        matched_train_relation=matched,
        relation_order=order,
        scores=scores,
        token_lengths=lengths,
        known_relations=known or [matched],
        temperature=1.0,
        model_checkpoint="b1",
    )
    return [row for row in rows if row["is_valid_negative"]]


def test_gold_key_prefers_matched_train_relation() -> None:
    assert gold_key({"gold_relation": "parkincity", "matched_train_relation": "concept:parkincity"}) == (
        "concept:parkincity"
    )
    assert gold_key({"gold_relation": "concept:atdate"}) == "concept:atdate"


def test_beats_gold_is_strict() -> None:
    assert beats_gold({"candidate_score": -0.2, "gold_score": -1.0}) is True
    assert beats_gold({"candidate_score": -1.0, "gold_score": -1.0}) is False
    assert beats_gold({"score_gap": -0.1}) is True
    assert beats_gold({"score_gap": 0.0}) is False


def test_aggregate_maps_alias_onto_canonical_gold() -> None:
    order_scores = {
        "concept:gold": -1.0,
        "concept:hard": -0.2,
        "concept:easy": -4.0,
    }
    rows = []
    rows.extend(_qa("task_qa:0", "gold", "concept:gold", order_scores))
    rows.extend(
        _qa(
            "task_qa:1",
            "gold",
            "concept:gold",
            {"concept:gold": -1.0, "concept:hard": -0.4, "concept:easy": -3.0},
        )
    )
    payload = aggregate_relation_pairs(rows)
    by_pair = {
        (row["gold_relation"], row["candidate_relation"]): row for row in payload["pair_rows"]
    }
    hard = by_pair[("concept:gold", "concept:hard")]
    assert hard["sample_count"] == 2
    assert hard["beats_gold_count"] == 2
    assert hard["beats_gold_rate"] == 1.0
    assert hard["median_score_gap"] < 0
    assert 0.0 < hard["p90_pairwise_confusion"] <= 1.0
    assert payload["n_qa"] == 2
    assert payload["gold_qa_count"]["concept:gold"] == 2


def test_min_support_filters_display_not_raw(tmp_path: Path) -> None:
    rows = []
    rows.extend(
        _qa(
            "task_qa:0",
            "concept:gold",
            "concept:gold",
            {"concept:gold": -1.0, "concept:hard": -0.2, "concept:rare": -2.0},
        )
    )
    rows.extend(
        _qa(
            "task_qa:1",
            "concept:gold",
            "concept:gold",
            {"concept:gold": -1.0, "concept:hard": -0.3, "concept:rare": -2.5},
        )
    )
    rows.extend(
        _qa(
            "task_qa:2",
            "concept:gold",
            "concept:gold",
            {"concept:gold": -1.0, "concept:hard": -0.25, "concept:once": -2.2},
        )
    )
    payload = aggregate_relation_pairs(rows)
    raw = payload["pair_rows"]
    displayed = filter_pairs_for_display(raw, min_support=2)
    by_candidate = {row["candidate_relation"]: row for row in raw}
    assert by_candidate["concept:once"]["sample_count"] == 1
    assert "concept:once" in {row["candidate_relation"] for row in raw}
    assert all(row["sample_count"] >= 2 for row in displayed)
    assert {row["candidate_relation"] for row in displayed} == {"concept:hard", "concept:rare"}

    order = ["concept:gold", "concept:hard", "concept:rare", "concept:once"]
    grid = matrix_grid(raw, order, value_key="mean_pairwise_confusion", min_support=1)
    assert grid[0][0] is None
    assert grid[0][1] is not None
    filtered = matrix_grid(raw, order, value_key="beats_gold_rate", min_support=4)
    assert filtered[0][1] is None

    neighbors = confusion_neighbors(raw, top_k=20, min_support=1)
    assert neighbors["concept:gold"][0]["relation"] == "concept:hard"
    assert {item["relation"] for item in neighbors["concept:gold"]} >= {"concept:hard", "concept:rare", "concept:once"}
    write_confusion_vocab_outputs(
        pair_payload=payload,
        relation_order=order,
        output_dir=tmp_path,
        metadata={"checkpoint": "b1", "dataset": "toy", "split": "train", "T": 1.0, "scoring_version": "v1"},
        min_support=2,
        neighbor_k=20,
        heatmap_top_n=3,
        neighborhood_plot_n=1,
    )
    table = (tmp_path / "relation_pair_statistics.csv").read_text(encoding="utf-8")
    assert "concept:once" in table
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["raw_table_unfiltered"] is True
    assert meta["min_support"] == 2
    assert meta["min_support_applies_to"] == ["figures", "display filtering"]
    assert meta["gold_key"] == "matched_train_relation"
    assert meta["level"] == "relation-global"
    assert meta["neighbors_are_analysis_only"] is True
    header = table.splitlines()[0].split(",")
    assert header == list(PAIR_FIELDS)
    neighbors_path = tmp_path / "relation_confusion_neighbors.json"
    stored = json.loads(neighbors_path.read_text(encoding="utf-8"))
    first = stored["concept:gold"][0]
    assert first["relation"] == "concept:hard"
    assert {"mean_confusion", "median_confusion", "support"} <= set(first)
    for name in (
        "relation_confusion_mean.csv",
        "relation_confusion_median.csv",
        "relation_confusion_beats_gold.csv",
    ):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("gold_relation,")
        assert "concept:once" in text.splitlines()[0]
    figures_dir = tmp_path / "figures"
    if meta.get("figures_skipped"):
        assert not list(figures_dir.glob("*.png"))
    else:
        assert (figures_dir / "confusion_mean_heatmap_topn.png").is_file()
        assert (figures_dir / "support_heatmap_topn.png").is_file()
        assert (figures_dir / "neighborhood_concept_gold.png").is_file()
