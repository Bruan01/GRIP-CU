"""Small dependency-free I/O helpers for the StructuredLoRA data audit."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import io
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

Triple = tuple[str, str, str]


def read_triples(path: Path) -> list[Triple]:
    rows: list[Triple] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields, got {len(fields)}")
            rows.append((fields[0], fields[1], fields[2]))
    if not rows:
        raise ValueError(f"No triples found in {path}")
    return rows


def iter_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if path.suffix == ".gz":
        raw = path.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        stream = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    else:
        stream = path.open("w", encoding="utf-8", newline="")
    try:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    finally:
        stream.close()
    return count


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
