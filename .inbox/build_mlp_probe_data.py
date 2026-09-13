#!/usr/bin/env python3
"""从 NELL23K train.txt 构造平衡的有边/无边事实探针数据。

每条样本判断一个具体事实 (source, relation, target) 是否存在于 train graph：
  label=1: train.txt 中存在该有向三元组；
  label=0: 固定 source 和 relation，替换 target，且替换后的三元组不存在。

训练、验证、测试只是在不同的真实正边上采样；负样本使用同样的 source/relation
做 corruption，避免探针只学习实体或关系频率。输出为 JSONL，每行一个样本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields")
            triples.append((fields[0], fields[1], fields[2]))
    if not triples:
        raise ValueError(f"没有从 {path} 读取到三元组")
    return triples


def make_question(title: str, triple: tuple[str, str, str]) -> str:
    source, relation, target = triple
    return (
        f"In context graph {title}, is the following fact true: "
        f"word node {source} is {relation} word node {target}? Answer yes or no."
    )


def make_split(
    *,
    split: str,
    positive_pool: list[tuple[str, str, str]],
    all_entities: list[str],
    edge_set: set[tuple[str, str, str]],
    count: int,
    seed: int,
    title: str,
) -> list[dict]:
    if count <= 0:
        return []
    if len(positive_pool) < count:
        raise ValueError(
            f"{split} 正边不足: requested={count}, available={len(positive_pool)}"
        )
    rng = random.Random(f"{seed}:{split}")
    positives = rng.sample(positive_pool, count)
    result: list[dict] = []
    for index, triple in enumerate(positives):
        source, relation, target = triple
        result.append(
            {
                "question_id": f"mlp_probe:{split}:positive:{index:06d}",
                "text": make_question(title, triple),
                "label": 1,
                "label_name": "edge_exists",
                "split": split,
                "source_node": source,
                "relation": relation,
                "target_node": target,
                "triple": list(triple),
                "source_type": "real_train_edge",
            }
        )

        # Keep source/relation fixed and corrupt only target.
        negative = None
        for _ in range(1000):
            corrupted_target = rng.choice(all_entities)
            candidate = (source, relation, corrupted_target)
            if candidate not in edge_set:
                negative = candidate
                break
        if negative is None:
            raise RuntimeError(
                f"无法为 {triple!r} 构造负样本；请检查实体数量或图密度"
            )
        result.append(
            {
                "question_id": f"mlp_probe:{split}:negative:{index:06d}",
                "text": make_question(title, negative),
                "label": 0,
                "label_name": "edge_absent",
                "split": split,
                "source_node": negative[0],
                "relation": negative[1],
                "target_node": negative[2],
                "triple": list(negative),
                "positive_counterpart": list(triple),
                "source_type": "corrupted_target",
            }
        )
    rng.shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        default=(
            "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/"
            "RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/"
            "data/raw_datasets/nell23k/train.txt"
        ),
    )
    parser.add_argument("--output-file", default="sample.jsonl")
    parser.add_argument("--train-positive-count", type=int, default=100)
    parser.add_argument("--validation-positive-count", type=int, default=50)
    parser.add_argument("--test-positive-count", type=int, default=100)
    parser.add_argument("--title", default="nell23k")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train_file = Path(args.train_file).resolve()
    output_file = Path(args.output_file).resolve()
    triples = load_triples(train_file)
    unique_triples = list(dict.fromkeys(triples))
    entities = sorted({entity for triple in unique_triples for entity in (triple[0], triple[2])})
    edge_set = set(unique_triples)

    counts = {
        "train": args.train_positive_count,
        "validation": args.validation_positive_count,
        "test": args.test_positive_count,
    }
    total_requested = sum(counts.values())
    if total_requested > len(unique_triples):
        raise ValueError(
            f"真实正边总数不足: requested={total_requested}, available={len(unique_triples)}"
        )

    # Different splits use disjoint positive triples to prevent exact-triple leakage.
    shuffled = unique_triples.copy()
    random.Random(args.seed).shuffle(shuffled)
    pools: dict[str, list[tuple[str, str, str]]] = {}
    cursor = 0
    for split, count in counts.items():
        pools[split] = shuffled[cursor : cursor + count]
        cursor += count

    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        rows.extend(
            make_split(
                split=split,
                positive_pool=pools[split],
                all_entities=entities,
                edge_set=edge_set,
                count=counts[split],
                seed=args.seed,
                title=args.title,
            )
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(output_file.read_bytes()).hexdigest()
    summary = {
        "output_file": str(output_file),
        "source_train_file": str(train_file),
        "source_train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "output_sha256": digest,
        "total_rows": len(rows),
        "positive_rows": sum(row["label"] == 1 for row in rows),
        "negative_rows": sum(row["label"] == 0 for row in rows),
        "entities": len(entities),
        "unique_train_edges": len(unique_triples),
        "positive_counts": counts,
        "seed": args.seed,
        "label_definition": "directed edge/fact existence in train.txt",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"样本已写入: {output_file}")


if __name__ == "__main__":
    main()
