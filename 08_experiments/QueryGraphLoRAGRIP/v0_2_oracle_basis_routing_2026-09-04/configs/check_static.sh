#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_BIN" scripts/validate_setup.py
if rg -n 'nell23k_exact_hop_test|predictions_test' query_graph_lora scripts configs --glob '!check_static.sh';then echo TEST_REFERENCE_DETECTED >&2;exit 2;fi
echo QUERY_GRAPH_LORA_STATIC_CHECK_PASSED
