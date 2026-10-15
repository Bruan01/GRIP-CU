#!/usr/bin/env python3
"""Download GRIP's FB15K-237 raw split without building 10-way candidates.

Source is WFRaain/TAG_datasets (the same Hugging Face dataset GRIP's
``process_fb15k237_2`` reads). Direct huggingface.co is often unreachable
from this machine, so the default endpoint is hf-mirror.com.

This only fetches the raw files. Candidate lists stay unsampled until
``process.py`` is run with an explicit seed.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
REPO = "WFRaain/TAG_datasets"
SUBFOLDER = "FB15K237"
FILES = (
    "entity2wikidata.json",
    "rel2text.json",
    "fb15k237.json",
    "train.txt",
    "valid.txt",
    "test.txt",
)
EXPECTED_LINES = {"train.txt": 272115, "valid.txt": 17535, "test.txt": 20466}
EXPECTED_RELATIONS = 237


def resolve_url(endpoint: str, filename: str) -> str:
    return f"{endpoint}/datasets/{REPO}/resolve/main/{SUBFOLDER}/{filename}"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if dest.exists() and dest.stat().st_size > 0:
        print(f"keep {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"get  {dest.name} <- {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "grip-fb15k237-download"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(dest)
    print(f"done {dest.name} ({dest.stat().st_size} bytes)")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify(out_dir: Path) -> dict:
    relations = set()
    counts = {}
    for name, expected in EXPECTED_LINES.items():
        got = count_lines(out_dir / name)
        counts[name] = got
        if got != expected:
            raise SystemExit(f"{name}: expected {expected} triples, got {got}")
        with (out_dir / name).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2:
                    relations.add(parts[1])
    rel_text = json.loads((out_dir / "rel2text.json").read_text(encoding="utf-8"))
    entities = json.loads((out_dir / "entity2wikidata.json").read_text(encoding="utf-8"))
    if len(relations) != EXPECTED_RELATIONS:
        raise SystemExit(f"expected {EXPECTED_RELATIONS} relations, got {len(relations)}")
    if len(rel_text) != EXPECTED_RELATIONS:
        raise SystemExit(f"rel2text.json has {len(rel_text)} entries, expected {EXPECTED_RELATIONS}")
    report = {
        "source": f"{REPO}/{SUBFOLDER}",
        "triples": counts,
        "relations_in_splits": len(relations),
        "rel2text": len(rel_text),
        "entities": len(entities),
    }
    (out_dir / "download_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw_datasets" / "fb15k237",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        download(resolve_url(args.endpoint, filename), args.out_dir / filename)
    report = verify(args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
