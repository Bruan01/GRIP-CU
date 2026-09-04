#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/build_entity_vocabulary.py
python3 scripts/validate_setup.py
python3 -m py_compile entity_decoder/*.py scripts/*.py
printf 'STATIC_CHECKS_PASSED\n'
