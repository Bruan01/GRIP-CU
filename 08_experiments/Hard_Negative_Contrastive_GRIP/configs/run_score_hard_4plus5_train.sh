#!/usr/bin/env bash
# Train the frozen-B1 score-hard mixed variant after full mining completes.
# The manifest is rebuilt offline as 4 B1-score hard + 5 uniform negatives.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
OLD_RUN="${OLD_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
S1_ADAPTER="${S1_ADAPTER:-$OLD_RUN/s1_adapter}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
MINING_SESSION="${MINING_SESSION:-score-hard-full-20260921}"
BASE_MANIFEST="${BASE_MANIFEST:-$HNG/results/runs/20260921_031844_score_hard_mining/score_hard_manifest.jsonl}"
MANIFEST="${MANIFEST:-$HNG/results/runs/20260921_031844_score_hard_mining/score_hard_4hard5uniform_manifest.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-3253}"
RUN_ID="${RUN_ID:-20260921_score_hard_4hard5uniform}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}}"
TMUX_SESSION="${TMUX_SESSION:-score-hard-4plus5-train-20260921}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$S1_ADAPTER" || ! -f "$INPUT_FILE" || ! -f "$EVAL_FILE" ]]; then
  echo "error: missing Stage-1 adapter or task/eval input" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR"
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "s1_adapter=$S1_ADAPTER"
  echo "base_manifest=$BASE_MANIFEST"
  echo "manifest=$MANIFEST"
  echo "hard_k=4"
  echo "uniform_k=5"
  echo "expected_rows=$EXPECTED_ROWS"
  echo "mining_session=$MINING_SESSION"
  echo "tmux_session=$TMUX_SESSION"
  date --iso-8601=seconds
} >> "$RUN_DIR/environment.txt"

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$CODE_DIR"
echo "[wait] waiting for complete mining session=$MINING_SESSION" | tee -a "$RUN_DIR/run.log"
while tmux has-session -t "$MINING_SESSION" 2>/dev/null; do
  lines=0
  if [[ -f "$BASE_MANIFEST" ]]; then
    lines=$(wc -l < "$BASE_MANIFEST")
  fi
  echo "[wait] mining_rows=$lines expected=$EXPECTED_ROWS $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/run.log"
  sleep 60
done

echo "[manifest] rebuilding 4 hard + 5 uniform" | tee -a "$RUN_DIR/run.log"
"$PYTHON" "$HNG/scripts/rebuild_score_hard_manifest.py" \
  --input "$BASE_MANIFEST" \
  --output "$MANIFEST" \
  --hard_k 4 \
  --total_k 9 \
  --seed 2026 \
  --expected_rows "$EXPECTED_ROWS" \
  2>&1 | tee -a "$RUN_DIR/run.log"

LISTED_RUN="$RUN_DIR" B1_RUN="$OLD_RUN" bash "$HNG/configs/attach_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"

TRAIN_FLAGS=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--no_resume)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_FLAGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "[launch] score-hard mixed train hard=4 uniform=5" | tee -a "$RUN_DIR/run.log"
set +e
"$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
  --input_file "$INPUT_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$RUN_DIR" \
  --stage listed \
  --s1_adapter "$S1_ADAPTER" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --lora_r 4 \
  --lora_alpha 8 \
  --target_modules down_proj up_proj gate_proj \
  --num_train_epochs 1 \
  --involve_qa_epochs 10 \
  --s1_stop_loss_threshold 0.15 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 512 \
  --s1_gradient_checkpointing \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --lambda_candidate 1.0 \
  --temperature 1.0 \
  --gen_max_length 32 \
  --seed 2026 \
  --listed_negative_k 9 \
  --listed_negative_source score_hard \
  --score_hard_manifest "$MANIFEST" \
  --raw_dir "$RAW_DIR" \
  --save_steps "${SAVE_STEPS:-10}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  "${TRAIN_FLAGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] score-hard mixed finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/run.log"
if [[ "$rc" -eq 0 ]]; then
  echo "[launch] compare mixed adapter vs frozen B1" | tee -a "$RUN_DIR/run.log"
  LISTED_RUN="$RUN_DIR" B1_RUN="$OLD_RUN" bash "$HNG/configs/compare_listed_to_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"
fi
exit "$rc"
