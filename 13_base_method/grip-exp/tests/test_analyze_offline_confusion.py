import json

from scripts.analyze_offline_confusion import _build_relation_confusion_vocab, _load_and_aggregate


def test_load_and_aggregate_counts_invalid_negative_rows(tmp_path):
    rows = [
        {
            "qa_id": "q1",
            "global_index": 0,
            "gold_relation": "gold",
            "candidate_relation": "gold",
            "gold_score": -1.0,
            "candidate_score": -1.0,
            "score_gap": 0.0,
            "is_gold": True,
            "is_valid_negative": False,
            "all_true_relations": ["gold", "alternative"],
        },
        {
            "qa_id": "q1",
            "global_index": 0,
            "gold_relation": "gold",
            "candidate_relation": "alternative",
            "gold_score": -1.0,
            "candidate_score": -0.9,
            "score_gap": -0.1,
            "pairwise_confusion": 0.55,
            "negative_mass": 1.0,
            "negative_rank": 1,
            "is_gold": False,
            "is_valid_negative": False,
            "all_true_relations": ["gold", "alternative"],
        },
        {
            "qa_id": "q1",
            "global_index": 0,
            "gold_relation": "gold",
            "candidate_relation": "other",
            "gold_score": -1.0,
            "candidate_score": -2.0,
            "score_gap": 1.0,
            "pairwise_confusion": 0.1,
            "negative_mass": 1.0,
            "negative_rank": 2,
            "is_gold": False,
            "is_valid_negative": True,
            "all_true_relations": ["gold", "alternative"],
        },
    ]
    path = tmp_path / "candidate_scores.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    context, base = _load_and_aggregate(path)

    assert base["qa_count"] == 1
    assert context["valid_negative_rows"] == 1
    assert context["invalid_negative_rows"] == 1
    assert context["qa_states"]["q1"]["invalid_negative_count"] == 1


def test_relation_confusion_vocab_keeps_raw_pairs_and_filters_neighbors(tmp_path):
    pair_rows = [
        {
            "gold_relation": "gold",
            "candidate_relation": "hard",
            "count": 10,
            "mean_candidate_score": -1.0,
            "mean_score_gap": 0.1,
            "median_score_gap": 0.1,
            "mean_pairwise_confusion": 0.4,
            "median_pairwise_confusion": 0.3,
            "p90_pairwise_confusion": 0.8,
            "candidate_beats_gold_count": 2,
            "candidate_beats_gold_fraction": 0.2,
        },
        {
            "gold_relation": "gold",
            "candidate_relation": "rare",
            "count": 2,
            "mean_candidate_score": -2.0,
            "mean_score_gap": 1.0,
            "median_score_gap": 1.0,
            "mean_pairwise_confusion": 0.2,
            "median_pairwise_confusion": 0.2,
            "p90_pairwise_confusion": 0.2,
            "candidate_beats_gold_count": 0,
            "candidate_beats_gold_fraction": 0.0,
        },
    ]
    context = {
        "input_file": str(tmp_path / "candidate_scores.jsonl"),
        "candidate_vocabulary": {"hard", "rare"},
        "gold_vocabulary": {"gold"},
    }

    metadata = _build_relation_confusion_vocab(
        tmp_path / "confusion_vocab",
        pair_rows,
        context,
        metadata_path=None,
        min_support=5,
        neighbor_top_k=20,
    )

    output_dir = tmp_path / "confusion_vocab"
    assert metadata["raw_pair_count"] == 2
    assert metadata["display_pair_count"] == 1
    assert "p90_pairwise_confusion" in (output_dir / "relation_pair_statistics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert (output_dir / "relation_confusion_mean.csv").is_file()
    assert (output_dir / "relation_confusion_median.csv").is_file()
    assert (output_dir / "relation_confusion_beats_gold.csv").is_file()
    assert (output_dir / "relation_confusion_support.csv").is_file()
    neighbors = json.loads((output_dir / "relation_confusion_neighbors.json").read_text(encoding="utf-8"))
    assert [item["relation"] for item in neighbors["gold"]] == ["hard"]
