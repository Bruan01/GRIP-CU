#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/kieran/miniconda3/envs/guardenv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
RUN_ID="${RUN_ID:-wsl3090_entity_decoder_validation_01}"
RUN_MODE="${RUN_MODE:-phase_a}"
cd "$ROOT"

case "$RUN_MODE" in
  phase_a)
    DECODERS=(D0 D1)
    DEFAULT_CHECKPOINTS=(direct_answer_only_seed43 direct_answer_only_seed44 more_qa_equal_token_seed42)
    ;;
  d2)
    DECODERS=(D2)
    DEFAULT_CHECKPOINTS=(direct_answer_only_seed43 direct_answer_only_seed44 more_qa_equal_token_seed42)
    if [[ ! -f "$ROOT/results/runs/$RUN_ID/suite_summary.json" ]]; then
      echo "D2 requires a completed Phase-A suite_summary.json" >&2
      exit 2
    fi
    decision="$($PYTHON - "$ROOT/results/runs/$RUN_ID/suite_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["gate"]["decision"])
PY
)"
    if [[ "$decision" != "PRELIMINARY_GO_D2" ]]; then
      echo "D2 blocked by gate decision=$decision" >&2
      exit 2
    fi
    ;;
  diagnostic)
    DECODERS=(D3)
    DEFAULT_CHECKPOINTS=(direct_answer_only_seed43 direct_answer_only_seed44 more_qa_equal_token_seed42)
    ;;
  *) echo "unknown RUN_MODE=$RUN_MODE (expected phase_a, d2, or diagnostic)" >&2; exit 2 ;;
esac
if [[ -n "${CHECKPOINTS:-}" ]]; then
  read -r -a SELECTED <<< "$CHECKPOINTS"
else
  SELECTED=("${DEFAULT_CHECKPOINTS[@]}")
fi

CHECKPOINT_ARGS=()
for checkpoint in "${SELECTED[@]}"; do
  CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done
RUNTIME_SELF_TEST_CHECKPOINT="${SELECTED[0]}" bash configs/check_wsl_runtime.sh "${CHECKPOINT_ARGS[@]}"
RUN_ROOT="$ROOT/results/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/$RUN_MODE"
for checkpoint in "${SELECTED[@]}"; do
  output="$RUN_ROOT/$RUN_MODE/$checkpoint"
  if [[ -f "$output/run_summary.json" ]]; then
    echo "SKIP complete $RUN_MODE/$checkpoint"
    continue
  fi
  "$PYTHON" - "$output" <<'PY'
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
PY
  "$PYTHON" scripts/run_decoder_smoke.py \
    --checkpoint "$checkpoint" \
    --decoders "${DECODERS[@]}" \
    --model-name-or-path "$MODEL_NAME_OR_PATH" \
    --output-dir "$output"
done
"$PYTHON" scripts/summarize_smoke.py --run-dir "$RUN_ROOT"
printf 'RUN_COMPLETE mode=%s output=%s\n' "$RUN_MODE" "$RUN_ROOT"
