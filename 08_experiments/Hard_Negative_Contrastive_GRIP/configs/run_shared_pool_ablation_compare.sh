#!/usr/bin/env bash
# CPU-only five-arm comparison for Prompt-6.
# Does not load a model. Omits variants that still lack listed/summary.json.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python}"
OUTPUT="${OUTPUT:-$HNG/results/runs/20260929_shared_pool_mixture_samplers/ablation_comparison.json}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$HNG/scripts/compare_shared_pool_sampler_runs.py" \
  --output "$OUTPUT" \
  "$@"
echo "[compare] wrote $OUTPUT"
bash "$HNG/configs/archive_and_push_results.sh" "$(dirname "$OUTPUT")"
