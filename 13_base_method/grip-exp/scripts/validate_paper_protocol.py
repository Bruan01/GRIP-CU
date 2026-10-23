"""Validate dataset/task artifacts before launching a paper-facing run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grip.paper_protocol import get_paper_protocol  # noqa: E402


def load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        return payload if isinstance(payload, list) else [payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wn18rr", "clegr_reasoning"], required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--task_file", type=Path, default=None)
    parser.add_argument("--allow_nonpaper_split", action="store_true")
    args = parser.parse_args()

    protocol = get_paper_protocol(args.dataset)
    test_rows = load_json_or_jsonl(args.test_file)
    expected_graphs = 1 if args.dataset == "wn18rr" else protocol.num_test
    print(f"dataset={args.dataset}")
    print(f"test_records={len(test_rows)} expected_for_paper={expected_graphs}")
    if expected_graphs is not None and len(test_rows) != expected_graphs:
        message = (
            f"{args.dataset} test split has {len(test_rows)} records, while the paper protocol "
            f"requires {expected_graphs}. This is a data/split mismatch, not a model result."
        )
        if not args.allow_nonpaper_split:
            raise SystemExit(message + " Pass --allow_nonpaper_split only for an explicitly labelled pilot.")
        print("WARNING: " + message)

    if args.task_file is not None:
        payload = json.loads(args.task_file.read_text(encoding="utf-8"))
        config = payload.get("generation_config") or {}
        expected = {
            "num_context_qa": protocol.context_qa,
            "num_reason_qa": protocol.reasoning_qa,
            "num_summarization": protocol.summarization,
            "format_as_instruction": protocol.format_as_instruction,
        }
        for key, value in expected.items():
            actual = config.get(key)
            status = "OK" if actual == value else "MISMATCH"
            print(f"{status}: {key}: actual={actual!r}, expected={value!r}")
        if any(config.get(key) != value for key, value in expected.items()):
            raise SystemExit("Task file generation settings do not match the selected paper protocol.")

    print("Paper protocol preflight passed for the checked artifacts.")


if __name__ == "__main__":
    main()
