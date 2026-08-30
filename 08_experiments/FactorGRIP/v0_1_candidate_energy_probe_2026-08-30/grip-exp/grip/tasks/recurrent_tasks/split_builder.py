from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Iterable

from .path_sampler import infer_question_type, make_path_question


def _parallel_metadata(record: dict[str, Any], index: int) -> dict[str, Any]:
    metadata_list = record.get("question_metadata")
    if isinstance(metadata_list, list) and index < len(metadata_list):
        item = metadata_list[index]
        return item if isinstance(item, dict) else {}
    metadata: dict[str, Any] = {}
    for source_key, target_key in (
        ("question_types", "question_type"),
        ("question_groups", "question_group"),
        ("question_subgroups", "question_subgroup"),
    ):
        values = record.get(source_key)
        if isinstance(values, list) and index < len(values):
            metadata[target_key] = values[index]
    return metadata


def build_graph_hop_split(
    record: dict[str, Any],
    train_hops: Iterable[int] = (1, 2),
    validation_hops: Iterable[int] = (1, 2),
    test_hops: Iterable[int] = (3, 4),
    question_types: Iterable[str] = ("StationShortestCount",),
    validation_fraction: float = 0.2,
    max_questions_per_hop: int = 32,
    seed: int = 2026,
) -> tuple[dict[str, Any], Counter]:
    train_hops = set(int(value) for value in train_hops)
    validation_hops = set(int(value) for value in validation_hops)
    test_hops = set(int(value) for value in test_hops)
    allowed_types = set(question_types)
    if train_hops & test_hops:
        raise ValueError("train_hops and test_hops must be disjoint")

    graph_id = str(record.get("id", record.get("title", "graph")))
    questions = record.get("questions", [])
    answers = record.get("answers", [])
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")

    accepted_by_hop: dict[int, list] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for index, (question, answer) in enumerate(zip(questions, answers)):
        metadata = _parallel_metadata(record, index)
        question_type = metadata.get("question_type") or infer_question_type(str(question))
        if question_type not in allowed_types:
            stats["rejected_question_type"] += 1
            continue
        try:
            sample = make_path_question(
                graph_id=graph_id,
                local_index=index,
                graph=record["graph"],
                question=str(question),
                answer=answer,
                question_type=question_type,
            )
        except ValueError as error:
            reason = str(error).split(":", 1)[0]
            stats[f"rejected_{reason.replace(' ', '_')}"] += 1
            rejected.append({"index": index, "question": question, "reason": str(error)})
            continue
        if sample.true_hop not in train_hops | validation_hops | test_hops:
            stats["rejected_hop_outside_requested_range"] += 1
            continue
        accepted_by_hop[sample.true_hop].append(sample)

    rng = random.Random(f"{seed}:{graph_id}")
    output_samples = []
    for hop, samples in sorted(accepted_by_hop.items()):
        rng.shuffle(samples)
        if max_questions_per_hop > 0:
            samples = samples[:max_questions_per_hop]
        if hop in test_hops:
            output_samples.extend(replace(sample, split="test").to_dict() for sample in samples)
            stats[f"accepted_test_hop_{hop}"] += len(samples)
            continue
        validation_count = 0
        if hop in validation_hops and validation_fraction > 0 and len(samples) > 1:
            validation_count = min(len(samples) - 1, max(1, round(len(samples) * validation_fraction)))
        validation_samples = samples[:validation_count]
        train_samples = samples[validation_count:] if hop in train_hops else []
        output_samples.extend(replace(sample, split="validation").to_dict() for sample in validation_samples)
        output_samples.extend(replace(sample, split="train").to_dict() for sample in train_samples)
        stats[f"accepted_validation_hop_{hop}"] += len(validation_samples)
        stats[f"accepted_train_hop_{hop}"] += len(train_samples)

    enriched = dict(record)
    enriched["recurrent_questions"] = output_samples
    enriched["recurrent_rejections"] = rejected
    enriched["recurrent_split_stats"] = dict(stats)
    return enriched, stats


def build_dataset_hop_split(records: list[dict[str, Any]], max_graphs: int = 16, **kwargs):
    output = []
    total_stats: Counter = Counter()
    for record in records:
        enriched, stats = build_graph_hop_split(record, **kwargs)
        has_train = any(item["split"] == "train" for item in enriched["recurrent_questions"])
        has_test = any(item["split"] == "test" for item in enriched["recurrent_questions"])
        if not has_train or not has_test:
            total_stats["rejected_graph_missing_train_or_test"] += 1
            continue
        output.append(enriched)
        total_stats.update(stats)
        total_stats["accepted_graphs"] += 1
        if max_graphs > 0 and len(output) >= max_graphs:
            break
    return output, dict(total_stats)
