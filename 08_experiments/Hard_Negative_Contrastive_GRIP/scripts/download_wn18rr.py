#!/usr/bin/env python3
"""Fetch and validate WFRaain/TAG_datasets/WN18RR raw data on CPU.

The default endpoint is hf-mirror.com (override with --endpoint/HF_ENDPOINT).
Each file is validated before atomic replacement; existing files are validated
before reuse. Network errors deliberately omit URLs, headers and credentials.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request

from prepare_wn18rr import (
    ROOT, REQUIRED_FILES, EXPECTED_COUNTS, SPLIT_FILES, WAY,
    atomic_write_json, sha256_file, load_entities, load_triples, load_raw,
    official_relation_text,
)

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
REPO = "WFRaain/TAG_datasets"
SUBFOLDER = "WN18RR"
FILES = REQUIRED_FILES
EXPECTED_LINES = {SPLIT_FILES[split]: count for split, count in EXPECTED_COUNTS.items()}
EXPECTED_RELATIONS = WAY


def resolve_url(endpoint: str, filename: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{REPO}/resolve/main/{SUBFOLDER}/{filename}"


def validate_file(path: Path, filename: str):
    if filename == "entity2text.txt":
        return len(load_entities(path))
    triples = load_triples(path)
    if len(triples) != EXPECTED_LINES[filename]:
        raise ValueError(f"{filename}: expected {EXPECTED_LINES[filename]} triples, got {len(triples)}")
    relations = {relation for _, relation, _ in triples}
    texts = {official_relation_text(relation) for relation in relations}
    if len(relations) > WAY or len(texts) != len(relations):
        raise ValueError(f"{filename}: invalid relation IDs or relation text collisions")
    if filename == "train.txt" and len(relations) != WAY:
        raise ValueError(f"{filename}: expected {WAY} training relations")
    return len(triples)


def download(url: str, dest: Path, *, timeout: float = 30):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        try:
            validate_file(dest, dest.name)
        except (ValueError, UnicodeError):
            print(f"replace invalid {dest.name}")
        else:
            print(f"keep {dest.name} ({dest.stat().st_size} bytes; validated)")
            return
    staged = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "grip-wn18rr-download"})
        print(f"get {dest.name}", flush=True)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code not in (401, 403):
                raise
            # Mirrors can cache expired signed redirects. One fresh resolver
            # request avoids reusing that redirect without logging its URL.
            separator = "&" if "?" in url else "?"
            fresh_url = f"{url}{separator}download=true&cache_bust={time.time_ns()}"
            fresh = urllib.request.Request(fresh_url, headers={"User-Agent": "grip-wn18rr-download"})
            response = urllib.request.urlopen(fresh, timeout=timeout)
        with response:
            with tempfile.NamedTemporaryFile(mode="wb", dir=dest.parent, prefix=f".{dest.name}.", suffix=".partial", delete=False) as handle:
                staged = Path(handle.name)
                while block := response.read(1 << 20):
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        validate_file(staged, dest.name)
        os.replace(staged, dest)
        print(f"done {dest.name} ({dest.stat().st_size} bytes; validated)")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        # Some transport exceptions include signed URLs. Keep only the type.
        raise RuntimeError(f"download failed for {dest.name}: {type(error).__name__}; check endpoint/network") from None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def verify(out_dir: Path) -> dict:
    source_hashes = {name: sha256_file(out_dir / name) for name in FILES}
    entities, triples, relations, train_unknown = load_raw(out_dir)
    if any(sha256_file(out_dir / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("source changed during download verification")
    report = {
        "source": f"{REPO}/{SUBFOLDER}",
        "triples": {SPLIT_FILES[split]: len(rows) for split, rows in triples.items()},
        "relations_in_splits": len(relations), "entities": len(entities),
        "train_unknown_entities": train_unknown,
        "heldout_entity_coverage_verified": True, "heldout_relation_coverage_verified": True,
        "source_sha256": source_hashes,
    }
    atomic_write_json(out_dir / "download_report.json", report, jsonl=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data/raw_datasets/wn18rr")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=30, help="Per-request network timeout in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        for filename in FILES:
            download(resolve_url(args.endpoint, filename), args.out_dir / filename, timeout=args.timeout)
        report = verify(args.out_dir)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
