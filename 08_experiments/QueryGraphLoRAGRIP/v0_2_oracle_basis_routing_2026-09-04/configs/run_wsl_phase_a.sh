#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";cd "$ROOT"
: "${RUN_ID:?Set a new immutable RUN_ID}"
OUT="$ROOT/results/runs/$RUN_ID";test ! -e "$OUT" || { echo "RUN_ID_EXISTS: $OUT" >&2;exit 3; }
ARGS=();if [[ -n "${MODEL_OVERRIDE:-}" ]];then ARGS+=(--model "$MODEL_OVERRIDE");fi
python scripts/run_suite.py --config configs/phase_a_oracle.json --run-root "$OUT" --minimal "${ARGS[@]}"
echo "PHASE_A_FINISHED: $OUT"
