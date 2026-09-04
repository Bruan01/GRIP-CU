#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_BIN" scripts/validate_setup.py
"$PYTHON_BIN" scripts/runtime_self_test.py
nvidia-smi
echo QUERY_GRAPH_LORA_WSL_PREFLIGHT_PASSED
