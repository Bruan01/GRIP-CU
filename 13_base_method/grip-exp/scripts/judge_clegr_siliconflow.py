"""Evaluate CLEGR predictions with the paper's Qwen-32B LLM-as-Judge.

The API credential is read only from an environment variable.  Results are
cached by a hash of the question, candidate answer, target, model, and prompt
version, so a long 6k-question evaluation can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Keep these strings in sync with ``evaluation/metrics.py`` from the released
# GRIP implementation.  The examples matter because the paper's metric is a
# model-judged semantic match, rather than an arbitrary new judge rubric.
PAPER_JUDGE_SYSTEM = """
You are a helpful assistant. your task is to provide accurate, direct, and concise answers to user queries. If unable to
answer the question, respond with "I don't know". Do not guess the answer.
"""

PAPER_JUDGE_PROMPT = """
    You will be given one question, one predicted answer from an llm candidate model, and the ground truth answers.
    Be as best as you can to evaluate whether the answer generated from the model match one of the ground truth. If
    be included, response yes, otherwise no. A match is not required to be exactly the same,
    semantically included can be regarded as a match.
    Please DON'T output quotes when outputting your evaluation.
    Here is some examples:
    --Example 1--
    Question: What is in the box?
    Candidate answer: a pizza slice.
    Ground truth: [pizza, pizza slice].
    Evaluation: yes

    --Example 2--
    Question: On which side of the image are the chairs?
    Candidate answer: The chairs are on the left side of the image.
    Ground truth: [right]
    Evaluation: no


    --Example 3--
    Question: Who is wearing the hat?
    Candidate answer: the guy at coordinate (410, 163)
    Ground truth: [guy]
    Evaluation: yes


    --Real task--
    Question: {question}
    Candidate answer: {pred}
    Ground truth: {target}
    Evaluation:
    """.strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return payload
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def as_targets(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def request_key(row: dict[str, Any], model: str, prompt_version: str) -> str:
    payload = {
        "question": str(row.get("question", "")),
        "candidate": str(row.get("response", row.get("prediction", ""))),
        "target": as_targets(row.get("target")),
        "model": model,
        "prompt_version": prompt_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize_paper_answer(response: str) -> str:
    """Match the released GRIP ``normalize_answer`` behavior for judge text."""

    # Python's stdlib ``string.punctuation`` is exactly what the released
    # implementation uses; keep the dependency local to this standalone tool.
    import string
    normalization_map = {"man": "men", "woman": "women"}
    text = response.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(normalization_map.get(word, word) for word in text.split())


def parse_paper_label(response: str) -> str:
    """Return the label after the released paper's answer normalization."""

    normalized = normalize_paper_answer(response)
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    return "invalid"


def call_judge(client: Any, row: dict[str, Any], model: str, prompt_version: str, retries: int) -> dict[str, Any]:
    question = str(row.get("question", ""))
    candidate = str(row.get("response", row.get("prediction", "")))
    targets = as_targets(row.get("target"))
    prompt = PAPER_JUDGE_PROMPT.format(
        question=question,
        pred=candidate,
        target=str(targets),
    )
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PAPER_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=8,
            )
            raw = str(response.choices[0].message.content or "").strip()
            label = parse_paper_label(raw)
            return {
                "cache_key": request_key(row, model, prompt_version),
                "id": row.get("id"),
                "question": question,
                "candidate_answer": candidate,
                "target": targets,
                "judge_response": raw,
                "judge_label": label,
                "paper_score": int(label == "yes"),
                "model": model,
                "prompt_version": prompt_version,
            }
        except Exception as exc:  # API libraries expose several transient exception types.
            last_error = exc
            if attempt < retries:
                time.sleep(min(60.0, 2.0 ** attempt))
    raise RuntimeError(f"Judge request failed after {retries + 1} attempts: {last_error}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_file", type=Path, default=None)
    parser.add_argument("--cache_file", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--base_url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api_key_env", default="SILICONFLOW_API_KEY")
    parser.add_argument(
        "--env_file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
        help="dotenv file loaded before reading --api_key_env",
    )
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_workers < 1 or args.retries < 0 or args.limit < 0:
        raise ValueError("max_workers must be positive; retries and limit must be non-negative")
    env_file = args.env_file.expanduser().resolve()
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()
    summary_file = (args.summary_file or output_file.with_name("judge_summary.json")).resolve()
    cache_file = (args.cache_file or output_file.with_name("judge_cache.jsonl")).resolve()
    rows = load_rows(input_file)
    if args.limit:
        rows = rows[:args.limit]
    prompt_version = hashlib.sha256(
        (PAPER_JUDGE_SYSTEM + "\n" + PAPER_JUDGE_PROMPT).encode("utf-8")
    ).hexdigest()[:16]

    if args.dry_run:
        print(json.dumps({
            "input_rows": len(rows),
            "model": args.model,
            "base_url": args.base_url,
            "prompt_version": prompt_version,
            "api_key_env": args.api_key_env,
            "env_file": str(env_file),
            "would_call_api": bool(rows),
        }, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Export it as {args.api_key_env}; the key is never read from a repository file."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The guardenv OpenAI package is required for SiliconFlow judging.") from exc

    existing: dict[str, dict[str, Any]] = {}
    if not args.overwrite:
        for path in (cache_file, output_file):
            if path.exists():
                for result in load_rows(path):
                    if result.get("cache_key"):
                        existing[str(result["cache_key"])] = result

    keyed_rows = [(row, request_key(row, args.model, prompt_version)) for row in rows]
    missing = [(row, key) for row, key in keyed_rows if key not in existing]
    client = OpenAI(api_key=api_key, base_url=args.base_url, max_retries=0, timeout=120.0)
    started = time.time()
    api_calls = 0
    print(f"Judge rows={len(rows)}, cache_hits={len(rows) - len(missing)}, pending={len(missing)}")

    if missing:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(call_judge, client, row, args.model, prompt_version, args.retries): (row, key)
                for row, key in missing
            }
            for future in as_completed(futures):
                row, key = futures[future]
                result = future.result()
                existing[key] = result
                api_calls += 1
                if api_calls % 20 == 0 or api_calls == len(missing):
                    atomic_jsonl(cache_file, list(existing.values()))
                    print(f"completed {api_calls}/{len(missing)} API calls")

    ordered = [existing[key] for _, key in keyed_rows]
    atomic_jsonl(output_file, ordered)
    atomic_jsonl(cache_file, list(existing.values()))
    valid = [row for row in ordered if row.get("judge_label") in {"yes", "no"}]
    correct = sum(int(row.get("judge_label") == "yes") for row in ordered)
    summary = {
        "status": "complete",
        "finished_at": now_iso(),
        "elapsed_seconds": time.time() - started,
        "count": len(ordered),
        "correct": correct,
        "llm_as_judge": correct / len(ordered) if ordered else 0.0,
        "valid_label_count": len(valid),
        "invalid_label_count": len(ordered) - len(valid),
        "cache_hit_count": len(rows) - len(missing),
        "api_call_count": api_calls,
        "model": args.model,
        "base_url": args.base_url,
        "prompt_version": prompt_version,
        "input_file": str(input_file),
        "output_file": str(output_file),
    }
    atomic_json(summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
