#!/usr/bin/env python3
"""Verify/import offline CoDEx-M inputs, or fetch the pinned public sources."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from prepare_fb15k237 import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CODEX_REV = '3132e426c2a6b643b70bad679905a3a6270be440'
GRIP_REV = 'd423742ca41212c4a75f35bec11d7117b1921a1a'
TRIPLE_SHA = {
    'train.txt': 'd99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556',
    'valid.txt': '11c323096367354940846b9ed940c5dabddc1d20a70474222babe7e7fd28b28d',
    'test.txt': '0575ce05e4ce915e395bb4f708e8df9547989dd4c8c71407cbfd78c19daffcf7',
}
# Canonical JSON digests allow offline copies with different indentation/key order.
MAPPING_SHA = {
    'entity2text.json': 'e2a6d4da999aa693c07a334b4402c989991766bf2e6f12b94bf76ae76c1c5bb8',
    'codexm.json': '0457e2c403a8d9a4807db642cb8833a928942b00b9b935a2e3886944eca073ea',
}
FILES = tuple(TRIPLE_SHA) + tuple(MAPPING_SHA)


def source_url(name):
    if name in TRIPLE_SHA:
        return f'https://raw.githubusercontent.com/tsafavi/codex/{CODEX_REV}/data/triples/codex-m/{name}'
    return f'https://raw.githubusercontent.com/JiaruiFeng/GRIP/{GRIP_REV}/data/raw_datasets/codexm/{name}'


def canonical_hash(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def verify_file(path, name):
    if name in TRIPLE_SHA:
        actual, expected = sha256_file(path), TRIPLE_SHA[name]
    else:
        actual = canonical_hash(json.loads(path.read_text(encoding='utf-8')))
        expected = MAPPING_SHA[name]
    if actual != expected:
        raise ValueError(f'{name}: pinned source checksum mismatch; existing data retained')


def acquire(out_dir, *, import_dir=None, offline=False, timeout=30):
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = out_dir / name
        if dest.exists():
            verify_file(dest, name)
            continue
        source = None if import_dir is None else import_dir / name
        if source is None and offline:
            raise FileNotFoundError(f'{name}: missing offline source; use --import_dir')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=out_dir, prefix='.' + name,
                                             suffix='.partial', delete=False) as handle:
                temporary = Path(handle.name)
                if source is not None:
                    verify_file(source, name)
                    with source.open('rb') as stream:
                        shutil.copyfileobj(stream, handle)
                else:
                    with urllib.request.urlopen(source_url(name), timeout=timeout) as response:
                        shutil.copyfileobj(response, handle)
                handle.flush()
                os.fsync(handle.fileno())
            verify_file(temporary, name)
            os.replace(temporary, dest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    report = {'dataset': 'CoDEx-M', 'codex_revision': CODEX_REV, 'grip_revision': GRIP_REV,
              'mapping_check': 'canonical JSON SHA256; official field values must match exactly',
              'source_urls': {name: source_url(name) for name in FILES},
              'source_sha256': {name: sha256_file(out_dir / name) for name in FILES},
              'mapping_canonical_sha256': MAPPING_SHA,
              'verified_negative_files': 'not used by this relation-prediction task'}
    atomic_write_json(out_dir / 'source_manifest.json', report, jsonl=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', type=Path, default=ROOT / 'data/raw_datasets/codexm')
    parser.add_argument('--import_dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(acquire(args.out_dir, import_dir=args.import_dir,
                             offline=args.offline, timeout=args.timeout), indent=2))


if __name__ == '__main__':
    main()
