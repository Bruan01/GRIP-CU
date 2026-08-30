"""Integrity checks for the NELL23K RecurrentGRIP train/eval depth cross."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_recurrent_cross_run(
    run_dir: Path,
    train_depths: tuple[int, ...] = (1, 2),
    eval_depths: tuple[int, ...] = (1, 2),
    controls: tuple[str, ...] = ("correct", "none"),
) -> dict:
    manifests: dict[int, dict] = {}
    rows_by_train: dict[int, list[dict]] = {}
    question_sets: dict[tuple[int, str, str, int], set[tuple[str, str]]] = {}
    matrix_counts: dict[str, int] = {}

    for train_k in train_depths:
        subdir = run_dir / f"train_k{train_k}"
        manifest_paths = sorted(subdir.glob("adapters/*/context_sampling_manifest.json"))
        if len(manifest_paths) != 1:
            raise ValueError(
                f"train_k={train_k} expected exactly one context manifest, found {len(manifest_paths)}"
            )
        manifest = _load_json(manifest_paths[0])
        if int(manifest.get("selected_node_count", 0)) <= 0:
            raise ValueError(f"train_k={train_k} selected no node context")
        if int(manifest.get("selected_edge_count", 0)) <= 0:
            raise ValueError(f"train_k={train_k} selected no edge facts")
        expected_relation_count = min(
            int(manifest.get("relation_count", 0)),
            int(manifest.get("selected_edge_count", 0)),
        )
        if int(manifest.get("selected_relation_count", 0)) != expected_relation_count:
            raise ValueError(
                f"train_k={train_k} relation round-robin coverage is incomplete"
            )
        manifests[train_k] = manifest

        rows = _load_jsonl(subdir / "predictions.jsonl")
        if not rows:
            raise ValueError(f"train_k={train_k} has no predictions")
        rows_by_train[train_k] = rows
        seen: dict[tuple[str, str, int], set[tuple[str, str]]] = defaultdict(set)
        for row in rows:
            if int(row.get("recurrent_train_k", 0)) != train_k:
                raise ValueError(f"prediction train depth mismatch in train_k={train_k}")
            eval_k = int(row["recurrence_k"])
            control = str(row["adapter_control"])
            split = str(row.get("metadata", {}).get("split", "unknown"))
            if eval_k not in eval_depths:
                raise ValueError(f"unexpected eval depth: {eval_k}")
            if control not in controls:
                raise ValueError(f"unexpected adapter control: {control}")
            for field in (
                "generated_token_count",
                "ended_with_eos",
                "response_in_candidates",
                "step_pooled_hidden_states",
            ):
                if field not in row:
                    raise ValueError(f"prediction is missing audit field: {field}")
            trace = row["step_pooled_hidden_states"]
            if not isinstance(trace, list) or len(trace) != eval_k:
                raise ValueError(
                    f"prediction trace length mismatch: eval_k={eval_k}, "
                    f"trace_length={len(trace) if isinstance(trace, list) else 'invalid'}"
                )
            question_key = (str(row.get("graph_id", "unknown")), str(row["question_id"]))
            bucket = (split, control, eval_k)
            if question_key in seen[bucket]:
                raise ValueError(
                    f"duplicate prediction in train_k={train_k}, split={split}, "
                    f"adapter={control}, eval_k={eval_k}, question={question_key}"
                )
            seen[bucket].add(question_key)

        observed_controls = {control for _, control, _ in seen}
        observed_depths = {depth for _, _, depth in seen}
        if observed_controls != set(controls):
            raise ValueError(f"train_k={train_k} controls incomplete: {observed_controls}")
        if observed_depths != set(eval_depths):
            raise ValueError(f"train_k={train_k} eval depths incomplete: {observed_depths}")
        splits = {split for split, _, _ in seen}
        for split in splits:
            expected_questions: set[tuple[str, str]] | None = None
            for control in controls:
                for eval_k in eval_depths:
                    bucket = (split, control, eval_k)
                    questions = seen.get(bucket)
                    if not questions:
                        raise ValueError(
                            f"missing matrix cell train_k={train_k}, split={split}, "
                            f"adapter={control}, eval_k={eval_k}"
                        )
                    if expected_questions is None:
                        expected_questions = questions
                    elif questions != expected_questions:
                        raise ValueError(
                            f"question set mismatch in train_k={train_k}, split={split}"
                        )
                    key = (
                        f"split={split}|train_k={train_k}|eval_k={eval_k}|adapter={control}"
                    )
                    matrix_counts[key] = len(questions)
                    question_sets[(train_k, split, control, eval_k)] = questions

    hashes = {manifest.get("selection_sha256") for manifest in manifests.values()}
    if None in hashes or len(hashes) != 1:
        raise ValueError("selection_sha256 differs across train depths")

    reference_train_k = train_depths[0]
    for train_k in train_depths[1:]:
        for split, control, eval_k in {
            key[1:] for key in question_sets if key[0] == reference_train_k
        }:
            if question_sets[(train_k, split, control, eval_k)] != question_sets[
                (reference_train_k, split, control, eval_k)
            ]:
                raise ValueError(
                    f"question set differs between train depths for split={split}, "
                    f"adapter={control}, eval_k={eval_k}"
                )

    prediction_count = sum(len(rows) for rows in rows_by_train.values())
    return {
        "status": "pass",
        "run_dir": str(run_dir),
        "train_depths": list(train_depths),
        "eval_depths": list(eval_depths),
        "adapter_controls": list(controls),
        "prediction_count": prediction_count,
        "selection_sha256": next(iter(hashes)),
        "context_manifests": {
            f"train_k{train_k}": manifest for train_k, manifest in manifests.items()
        },
        "matrix_counts": dict(sorted(matrix_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    report = audit_recurrent_cross_run(Path(args.run_dir))
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
