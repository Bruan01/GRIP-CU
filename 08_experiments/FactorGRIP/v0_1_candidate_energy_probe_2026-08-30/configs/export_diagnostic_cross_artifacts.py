#!/usr/bin/env python3
"""Export compact, auditable RecurrentGRIP diagnostic-cross artifacts.

The immutable run directory remains local/ignored because it may contain adapters,
trainer states, and full hidden-state traces. This exporter copies the analysis and
integrity metadata into a Git-trackable directory and writes a prediction audit file
with the large ``step_pooled_hidden_states`` field removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_ROOT_FILES = (
    "config.json",
    "environment.txt",
    "cross_run_audit.json",
)
OPTIONAL_ROOT_FILES = ("input_stats.json",)
REQUIRED_ANALYSIS_FILES = (
    "summary.json",
    "train_eval_depth_accuracy.csv",
    "split_train_k_eval_k_adapter_accuracy.csv",
    "transition_k1_k2.csv",
    "output_quality.json",
    "output_quality_by_train_eval.csv",
    "state_dynamics.json",
    "state_dynamics_by_train_eval.csv",
    "state_dynamics_by_split_train_eval.csv",
    "k_adapter_accuracy.csv",
    "split_k_adapter_accuracy.csv",
    "hop_k_accuracy.csv",
)
REQUIRED_PREDICTION_FIELDS = (
    "graph_id",
    "question_id",
    "recurrent_train_k",
    "recurrence_k",
    "adapter_control",
    "correct",
    "response",
    "generated_token_count",
    "ended_with_eos",
    "response_in_candidates",
    "step_pooled_hidden_states",
    "metadata",
)
REDACTED_FIELDS = ("step_pooled_hidden_states",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(start: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _validate_run(run_dir: Path) -> tuple[list[dict], dict, dict]:
    if not run_dir.is_dir():
        raise ValueError(f"run directory not found: {run_dir}")

    for relative in REQUIRED_ROOT_FILES:
        path = run_dir / relative
        if not path.is_file():
            raise ValueError(f"required run artifact missing: {path}")
    for relative in REQUIRED_ANALYSIS_FILES:
        path = run_dir / "analysis" / relative
        if not path.is_file():
            raise ValueError(f"required analysis artifact missing: {path}")

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise ValueError(f"predictions not found: {predictions_path}")

    rows: list[dict] = []
    with predictions_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"prediction line {line_number} missing fields: {', '.join(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("predictions.jsonl is empty")

    audit = _load_json(run_dir / "cross_run_audit.json")
    if audit.get("status") != "pass":
        raise ValueError("cross_run_audit.json does not report status=pass")
    if int(audit.get("prediction_count", -1)) != len(rows):
        raise ValueError("cross-run audit prediction count does not match predictions")

    summary = _load_json(run_dir / "analysis" / "summary.json")
    if int(summary.get("count", -1)) != len(rows):
        raise ValueError("analysis summary count does not match predictions")
    return rows, audit, summary


def export_artifacts(run_dir: Path, output_dir: Path, repository_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    repository_dir = repository_dir.resolve()
    rows, audit, summary = _validate_run(run_dir)

    if output_dir.exists():
        raise ValueError(f"export directory already exists: {output_dir}")

    output_dir.mkdir(parents=True)
    try:
        for relative in REQUIRED_ROOT_FILES + OPTIONAL_ROOT_FILES:
            source = run_dir / relative
            if source.is_file():
                shutil.copy2(source, output_dir / relative)

        analysis_output = output_dir / "analysis"
        analysis_output.mkdir()
        for relative in REQUIRED_ANALYSIS_FILES:
            shutil.copy2(run_dir / "analysis" / relative, analysis_output / relative)

        manifests_output = output_dir / "context_manifests"
        manifests_output.mkdir()
        for train_k in (1, 2):
            source = (
                run_dir
                / f"train_k{train_k}"
                / "adapters"
                / "nell23k"
                / "context_sampling_manifest.json"
            )
            if not source.is_file():
                raise ValueError(f"context sampling manifest missing: {source}")
            shutil.copy2(source, manifests_output / f"train_k{train_k}.json")

        prediction_output = output_dir / "predictions_audit.jsonl"
        with prediction_output.open("w", encoding="utf-8") as stream:
            for row in rows:
                compact = {key: value for key, value in row.items() if key not in REDACTED_FIELDS}
                stream.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")

        exported_files = sorted(
            path for path in output_dir.rglob("*") if path.is_file()
        )
        manifest = {
            "artifact_type": "RecurrentGRIP_v1.1.1_diagnostic_cross_audit_export",
            "run_id": run_dir.name,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_git_commit": _git_commit(repository_dir),
            "prediction_count": len(rows),
            "analysis_count": summary["count"],
            "cross_run_audit_status": audit["status"],
            "selection_sha256": audit["selection_sha256"],
            "train_depths": audit["train_depths"],
            "eval_depths": audit["eval_depths"],
            "adapter_controls": audit["adapter_controls"],
            "redacted_prediction_fields": list(REDACTED_FIELDS),
            "redaction_reason": "full recurrent hidden-state vectors are large; aggregate dynamics remain in analysis/",
            "files": {
                str(path.relative_to(output_dir)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in exported_files
            },
        }
        manifest_path = output_dir / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = export_artifacts(args.run_dir, args.output_dir, args.repository_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
