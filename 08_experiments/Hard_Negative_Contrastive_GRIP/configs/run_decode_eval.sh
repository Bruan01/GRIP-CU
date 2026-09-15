#!/usr/bin/env bash
# Closed-set 10-way eval on saved 7B B1/listed adapters, then pilot decode.
#
# Does not train. Smoke writes closed-set files next to the 20260913 run.
# Pilot writes a new run directory with generate + closed-set on 640 questions.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
ADAPTER_RUN="${ADAPTER_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
SMOKE_EVAL="${SMOKE_EVAL:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
PILOT_EVAL="${PILOT_EVAL:-$HNG/data/nell23k/recurrent_relation_prediction_pilot.aligned.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PILOT_DIR="${PILOT_DIR:-$HNG/results/runs/${RUN_ID}_qwen7b_pilot_decode}"
SKIP_PILOT="${SKIP_PILOT:-0}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$ADAPTER_RUN/b1/adapter" || ! -d "$ADAPTER_RUN/listed/adapter" ]]; then
  echo "error: missing 7B adapters under $ADAPTER_RUN" >&2
  exit 1
fi
if [[ ! -f "$SMOKE_EVAL" || ! -f "$PILOT_EVAL" ]]; then
  echo "error: missing aligned eval files" >&2
  exit 1
fi

mkdir -p "$ADAPTER_RUN" "$PILOT_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

eval_one() {
  local adapter_dir="$1"
  local output_dir="$2"
  local eval_file="$3"
  local decode="$4"
  echo "[decode] adapter=$adapter_dir decode=$decode eval=$eval_file"
  "$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
    --adapter_dir "$adapter_dir" \
    --output_dir "$output_dir" \
    --eval_file "$eval_file" \
    --model_name qwen-7b \
    --model_cache_dir "$CODE_DIR/model_cache" \
    --decode "$decode" \
    --gen_max_length 32
}

echo "[decode] smoke closed-set start $(date --iso-8601=seconds)" | tee -a "$ADAPTER_RUN/decode.log"
eval_one "$ADAPTER_RUN/b1/adapter" "$ADAPTER_RUN/b1" "$SMOKE_EVAL" closed_set 2>&1 | tee -a "$ADAPTER_RUN/decode.log"
eval_one "$ADAPTER_RUN/listed/adapter" "$ADAPTER_RUN/listed" "$SMOKE_EVAL" closed_set 2>&1 | tee -a "$ADAPTER_RUN/decode.log"
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --compare \
  --eval_file "$SMOKE_EVAL" \
  --output_dir "$ADAPTER_RUN" 2>&1 | tee -a "$ADAPTER_RUN/decode.log"
echo "[decode] smoke closed-set done $(date --iso-8601=seconds)" | tee -a "$ADAPTER_RUN/decode.log"

if [[ "$SKIP_PILOT" == "1" ]]; then
  echo "[decode] skip pilot" | tee -a "$ADAPTER_RUN/decode.log"
  exit 0
fi

mkdir -p "$PILOT_DIR/b1" "$PILOT_DIR/listed"
{
  echo "adapter_run=$ADAPTER_RUN"
  echo "pilot_dir=$PILOT_DIR"
  echo "eval_file=$PILOT_EVAL"
  date --iso-8601=seconds
} >> "$PILOT_DIR/environment.txt"

echo "[decode] pilot generate+closed-set start $(date --iso-8601=seconds)" | tee -a "$PILOT_DIR/decode.log"
eval_one "$ADAPTER_RUN/b1/adapter" "$PILOT_DIR/b1" "$PILOT_EVAL" both 2>&1 | tee -a "$PILOT_DIR/decode.log"
eval_one "$ADAPTER_RUN/listed/adapter" "$PILOT_DIR/listed" "$PILOT_EVAL" both 2>&1 | tee -a "$PILOT_DIR/decode.log"
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --compare \
  --eval_file "$PILOT_EVAL" \
  --output_dir "$PILOT_DIR" | tee -a "$PILOT_DIR/decode.log"
echo "$PILOT_DIR" > "$HNG/results/LAST_PILOT_DECODE_RUN.txt"
echo "[decode] pilot done $(date --iso-8601=seconds)" | tee -a "$PILOT_DIR/decode.log"
echo "decode eval completed smoke=$ADAPTER_RUN pilot=$PILOT_DIR"
