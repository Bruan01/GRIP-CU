"""Import and provenance-check previously generated D0 prediction artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io_utils import read_jsonl, sha256_file
from .metrics import normalize_text


def import_d0_predictions(
    source: dict, repo_root: Path, dataset_rows: Iterable[dict]
) -> tuple[list[dict], dict]:
    """Align a frozen D0 artifact to registered dataset rows without trusting its metadata."""
    path = Path(source["path"])
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(f"missing D0 prediction artifact: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != source["sha256"]:
        raise ValueError(
            f"D0 artifact hash mismatch: expected {source['sha256']}, observed {observed_sha256}"
        )

    indexed: dict[str, dict] = {}
    for row in read_jsonl(path):
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"D0 artifact row missing task_id: {path}")
        if task_id in indexed:
            raise ValueError(f"duplicate D0 artifact task_id: {task_id}")
        indexed[task_id] = row

    dataset = list(dataset_rows)
    dataset_ids = [str(row.get("task_id", "")) for row in dataset]
    if not all(dataset_ids) or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("registered dataset task_ids must be present and unique")
    if set(dataset_ids) != set(indexed):
        missing = sorted(set(dataset_ids) - set(indexed))
        extra = sorted(set(indexed) - set(dataset_ids))
        raise ValueError(f"D0 artifact task_id mismatch: missing={missing[:5]} extra={extra[:5]}")

    imported = []
    for dataset_row in dataset:
        task_id = str(dataset_row["task_id"])
        source_row = indexed[task_id]
        if normalize_text(source_row.get("answer", "")) != normalize_text(dataset_row["answer"]):
            raise ValueError(f"D0 artifact answer mismatch for task_id: {task_id}")
        prediction_field = source.get("prediction_field")
        if prediction_field:
            prediction = source_row.get(prediction_field)
        else:
            prediction = source_row.get("prediction_answer", source_row.get("prediction_text", source_row.get("prediction")))
        if prediction is None:
            raise ValueError(f"D0 artifact prediction missing for task_id: {task_id}")
        row = dict(dataset_row)
        row.update({
            "prediction_text": str(prediction),
            "d0_artifact_source": str(path),
            "d0_artifact_sha256": observed_sha256,
        })
        imported.append(row)

    audit = {
        "source_path": str(path),
        "expected_sha256": source["sha256"],
        "observed_sha256": observed_sha256,
        "row_count": len(imported),
        "task_id_alignment": "exact",
        "metadata_authority": "registered_dataset_rows",
        "prediction_field_authority": "frozen_d0_artifact",
        "prediction_field": source.get("prediction_field", "auto"),
    }
    return imported, audit
