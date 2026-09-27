"""Copy a lightweight, never-overwritten experiment snapshot.

Weights, trainer checkpoints, tokenizers, and the 3253x198 score dump stay
on disk under ``results/runs/``. Git only receives JSON/Markdown summaries.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SKIP_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".tar",
    ".gz",
}
SKIP_NAMES = {
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
    "candidate_scores.jsonl",
    "confusion_db.jsonl",
}
SKIP_DIR_NAMES = {"trainer_listed", "trainer_s1", "trainer_b1"}
MAX_COPY_BYTES = 8 * 1024 * 1024
COPY_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".csv"}
COPY_LOG_NAMES = {
    "run.log",
    "decode.log",
    "cpu_audit.log",
    "live_20qa_rescore.log",
}


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_runs_root() -> Path:
    return package_root() / "results" / "runs"


def default_archive_root() -> Path:
    return package_root() / "results" / "archive"


def utc_stamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ")


def snapshot_run_name(run_dir: Path, *, runs_root: Path | None = None) -> str:
    """Stable archive folder name. Nested run dirs become ``parent__child``."""
    resolved = run_dir.resolve()
    root = (runs_root or default_runs_root()).resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return resolved.name
    parts = [part for part in rel.parts if part not in {".", ""}]
    return "__".join(parts) if parts else resolved.name


def unique_snapshot_dir(archive_root: Path, run_name: str, stamp: str) -> Path:
    base = archive_root / run_name / stamp
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = archive_root / run_name / f"{stamp}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def should_copy(path: Path, run_dir: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in SKIP_NAMES:
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    rel_parts = path.relative_to(run_dir).parts
    if any(part in SKIP_DIR_NAMES or part.startswith("checkpoint-") for part in rel_parts):
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > MAX_COPY_BYTES:
        return False
    if path.suffix.lower() in COPY_SUFFIXES:
        return True
    return path.name in COPY_LOG_NAMES


def _em_block(summary: dict | None) -> str:
    if not isinstance(summary, dict):
        return "n/a"
    all_block = summary.get("all") if isinstance(summary.get("all"), dict) else {}
    count = all_block.get("count")
    em = all_block.get("em")
    if em is None:
        return "n/a"
    try:
        pct = 100.0 * float(em)
    except (TypeError, ValueError):
        return "n/a"
    if count is None:
        return f"{pct:.2f}%"
    return f"{pct:.2f}% ({count})"


def metrics_from_run(run_dir: Path) -> dict:
    payload: dict = {
        "run_dir": str(run_dir),
        "listed_em": None,
        "b1_em": None,
        "listed_minus_b1_em": None,
        "eval_n": None,
    }
    comparison = run_dir / "comparison.json"
    listed_summary = run_dir / "listed" / "summary.json"
    b1_summary = run_dir / "b1" / "summary.json"
    if comparison.is_file():
        try:
            data = json.loads(comparison.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            listed = data.get("listed") if isinstance(data.get("listed"), dict) else {}
            b1 = data.get("b1") if isinstance(data.get("b1"), dict) else {}
            listed_all = listed.get("all") if isinstance(listed.get("all"), dict) else {}
            b1_all = b1.get("all") if isinstance(b1.get("all"), dict) else {}
            payload["listed_em"] = listed_all.get("em")
            payload["b1_em"] = b1_all.get("em")
            payload["eval_n"] = listed_all.get("count") or b1_all.get("count")
            payload["listed_minus_b1_em"] = data.get("listed_minus_b1_em")
            payload["comparison"] = data
            return payload
    listed = None
    b1 = None
    if listed_summary.is_file():
        try:
            listed = json.loads(listed_summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            listed = None
    if b1_summary.is_file():
        try:
            b1 = json.loads(b1_summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            b1 = None
    if isinstance(listed, dict):
        listed_all = listed.get("all") if isinstance(listed.get("all"), dict) else {}
        payload["listed_em"] = listed_all.get("em")
        payload["eval_n"] = listed_all.get("count")
    if isinstance(b1, dict):
        b1_all = b1.get("all") if isinstance(b1.get("all"), dict) else {}
        payload["b1_em"] = b1_all.get("em")
        payload["eval_n"] = payload["eval_n"] or b1_all.get("count")
    if payload["listed_em"] is not None and payload["b1_em"] is not None:
        payload["listed_minus_b1_em"] = float(payload["listed_em"]) - float(payload["b1_em"])
    return payload


def render_snapshot_markdown(*, dest: Path, run_dir: Path, metrics: dict, copied: list[str]) -> str:
    listed = _em_block((metrics.get("comparison") or {}).get("listed") if isinstance(metrics.get("comparison"), dict) else None)
    b1 = _em_block((metrics.get("comparison") or {}).get("b1") if isinstance(metrics.get("comparison"), dict) else None)
    if listed == "n/a" and metrics.get("listed_em") is not None:
        listed = f"{100.0 * float(metrics['listed_em']):.2f}%"
    if b1 == "n/a" and metrics.get("b1_em") is not None:
        b1 = f"{100.0 * float(metrics['b1_em']):.2f}%"
    delta = metrics.get("listed_minus_b1_em")
    delta_txt = "n/a"
    if isinstance(delta, (int, float)):
        delta_txt = f"{100.0 * float(delta):+.2f} pp"
    lines = [
        f"# Snapshot `{dest.name}`",
        "",
        f"- Source: `{run_dir}`",
        f"- Snapshot: `{dest}`",
        f"- Eval n: {metrics.get('eval_n')}",
        f"- Listed EM: {listed}",
        f"- Frozen B1 EM: {b1}",
        f"- Listed − B1: {delta_txt}",
        "",
        "Weights, trainer checkpoints, and score dumps are **not** copied.",
        "Re-running the same experiment writes a new stamp directory; old snapshots stay.",
        "",
        "## Copied files",
        "",
    ]
    for rel in copied:
        lines.append(f"- `{rel}`")
    lines.append("")
    return "\n".join(lines)


def append_index(index_path: Path, *, dest: Path, run_dir: Path, metrics: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.is_file():
        index_path.write_text(
            "# Experiment result snapshots\n\n"
            "Each row is a new directory. Snapshots are never overwritten.\n\n"
            "| stamp | run | listed EM | B1 EM | delta | path |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    listed = metrics.get("listed_em")
    b1 = metrics.get("b1_em")
    delta = metrics.get("listed_minus_b1_em")
    listed_txt = f"{100.0 * float(listed):.2f}%" if isinstance(listed, (int, float)) else ""
    b1_txt = f"{100.0 * float(b1):.2f}%" if isinstance(b1, (int, float)) else ""
    delta_txt = f"{100.0 * float(delta):+.2f} pp" if isinstance(delta, (int, float)) else ""
    row = (
        f"| {dest.name} | `{dest.parent.name}` | {listed_txt} | {b1_txt} | "
        f"{delta_txt} | `{dest.as_posix()}` |\n"
    )
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def archive_run(
    run_dir: Path,
    *,
    archive_root: Path | None = None,
    stamp: str | None = None,
    runs_root: Path | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    archive_root = (archive_root or default_archive_root()).resolve()
    run_name = snapshot_run_name(run_dir, runs_root=runs_root)
    dest = unique_snapshot_dir(archive_root, run_name, stamp or utc_stamp())
    dest.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not should_copy(path, run_dir):
            continue
        rel = path.relative_to(run_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(rel.as_posix())
    metrics = metrics_from_run(run_dir)
    (dest / "SNAPSHOT.md").write_text(
        render_snapshot_markdown(dest=dest, run_dir=run_dir, metrics=metrics, copied=copied),
        encoding="utf-8",
    )
    manifest = {
        "source_run": str(run_dir),
        "snapshot": str(dest),
        "copied": copied,
        "metrics": {
            "listed_em": metrics.get("listed_em"),
            "b1_em": metrics.get("b1_em"),
            "listed_minus_b1_em": metrics.get("listed_minus_b1_em"),
            "eval_n": metrics.get("eval_n"),
        },
        "note": "Lightweight Git snapshot. Adapter weights and score dumps remain in results/runs/.",
    }
    (dest / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_index(archive_root / "INDEX.md", dest=dest, run_dir=run_dir, metrics=metrics)
    return dest
