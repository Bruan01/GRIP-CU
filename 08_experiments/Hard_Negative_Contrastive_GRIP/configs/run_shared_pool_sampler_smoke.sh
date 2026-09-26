#!/usr/bin/env bash
# Listed-contrastive smoke on frozen shared-pool sampler manifests.
#
# Random-K vs Top-K Hard vs Coverage-Adaptive K. Same 64 paper-task QA,
# same 20260913 Stage-1 adapter, same frozen B1. Eval stays on the aligned
# 32/64 smoke split. Do not launch SCALE=full from this script.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SCALE="${SCALE:-smoke}"
VARIANT="${VARIANT:-all}"
OLD_RUN="${OLD_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
S1_ADAPTER="${S1_ADAPTER:-$OLD_RUN/s1_adapter}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
SAMPLER_DIR="${SAMPLER_DIR:-$HNG/results/runs/20260926_shared_pool_samplers}"
SEED="${SEED:-2026}"
LAST_RUN_FILE="$HNG/results/LAST_SHARED_POOL_SAMPLER_SMOKE_RUN.txt"

case "$SCALE" in
  smoke)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
    ;;
  pilot)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_pilot.aligned.json}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
    ;;
  full)
    echo "error: this launcher is smoke/pilot only; do not start full from here" >&2
    echo "  after smoke shows a direction, use a dedicated full listed-only script" >&2
    exit 1
    ;;
  *)
    echo "error: SCALE must be smoke or pilot, got: $SCALE" >&2
    exit 1
    ;;
esac

RUN_ID="${RUN_ID:-20260926_shared_pool_sampler_${SCALE}}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}}"
if [[ "${RESUME_LAST:-0}" == "1" ]]; then
  if [[ ! -f "$LAST_RUN_FILE" ]]; then
    echo "error: RESUME_LAST=1 but missing $LAST_RUN_FILE" >&2
    exit 1
  fi
  RUN_DIR="$(tr -d '\n' < "$LAST_RUN_FILE")"
fi
TMUX_SESSION="${TMUX_SESSION:-shared-pool-sampler-${SCALE}-20260926}"

VARIANTS=()
case "$VARIANT" in
  all)
    VARIANTS=(random_k top_k_hard coverage_adaptive_k)
    ;;
  random_k|top_k_hard|coverage_adaptive_k)
    VARIANTS=("$VARIANT")
    ;;
  *)
    echo "error: VARIANT must be all, random_k, top_k_hard, or coverage_adaptive_k" >&2
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
  echo "error: PyTorch cannot see a CUDA device; refusing to run CPU experiment" >&2
  exit 1
fi
MODEL_CACHE="$CODE_DIR/model_cache"
MODEL_DIR="$MODEL_CACHE/Qwen--Qwen2.5-7B-Instruct"
if ! PYTHONPATH="$CODE_DIR" "$PYTHON" -c "
from pathlib import Path
from models.utils import _is_complete_model_dir
raise SystemExit(0 if _is_complete_model_dir(Path(r'$MODEL_DIR')) else 1)
"; then
  echo "error: Qwen2.5-7B cache missing or incomplete: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$S1_ADAPTER/adapter_config.json" || ! -f "$S1_ADAPTER/adapter_model.safetensors" ]]; then
  echo "error: incomplete Stage-1 adapter: $S1_ADAPTER" >&2
  exit 1
fi
for required in "$INPUT_FILE" "$EVAL_FILE" "$RAW_DIR/train.txt" "$SAMPLER_DIR/policy.json"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done
for name in random_k top_k_hard coverage_adaptive_k; do
  if [[ ! -f "$SAMPLER_DIR/${name}.jsonl" ]]; then
    echo "error: missing frozen manifest: $SAMPLER_DIR/${name}.jsonl" >&2
    exit 1
  fi
done
if [[ "${FORCE_NEW:-0}" == "1" && -e "$RUN_DIR" ]]; then
  echo "error: FORCE_NEW=1 but run directory already exists: $RUN_DIR" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$LAST_RUN_FILE"
TASK_FILE="$RUN_DIR/task_${SCALE}${MAX_TRAIN_SAMPLES}.json"

{
  echo "scale=$SCALE"
  echo "variant=$VARIANT"
  echo "run_dir=$RUN_DIR"
  echo "s1_adapter=$S1_ADAPTER"
  echo "input_file=$INPUT_FILE"
  echo "eval_file=$EVAL_FILE"
  echo "sampler_dir=$SAMPLER_DIR"
  echo "max_train_samples=$MAX_TRAIN_SAMPLES"
  echo "seed=$SEED"
  echo "tmux_session=$TMUX_SESSION"
  echo "listed_negative_source=score_hard"
  echo "note=listed-only 64/32/64 smoke on frozen shared-pool manifests; B1 is 20260913"
  date --iso-8601=seconds
  "$PYTHON" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  for name in random_k top_k_hard coverage_adaptive_k; do
    echo -n "manifest_${name}_sha256="
    sha256sum "$SAMPLER_DIR/${name}.jsonl" | awk '{print $1}'
  done
} >> "$RUN_DIR/environment.txt" 2>&1

export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

log() {
  echo "$*" | tee -a "$RUN_DIR/run.log"
}

if [[ ! -f "$TASK_FILE" ]]; then
  log "[subset] keep original task_qa IDs; n=$MAX_TRAIN_SAMPLES seed=$SEED"
  "$PYTHON" "$HNG/scripts/subset_matchable_task_qa.py" \
    --input "$INPUT_FILE" \
    --manifest "$SAMPLER_DIR/random_k.jsonl" \
    --output "$TASK_FILE" \
    --max-samples "$MAX_TRAIN_SAMPLES" \
    --seed "$SEED" 2>&1 | tee -a "$RUN_DIR/run.log"
fi
log "[wire] attach frozen Random-K / Top-K Hard / Coverage-Adaptive K to $TASK_FILE"
"$PYTHON" "$HNG/scripts/verify_shared_pool_sampler_wiring.py" \
  --task_file "$TASK_FILE" \
  --sampler_dir "$SAMPLER_DIR" \
  --raw_dir "$RAW_DIR" \
  --expected_n "$MAX_TRAIN_SAMPLES" \
  --output "$RUN_DIR/wiring.json" 2>&1 | tee -a "$RUN_DIR/run.log"

TRAIN_FLAGS=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--no_resume)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_FLAGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

status=0
for name in "${VARIANTS[@]}"; do
  variant_dir="$RUN_DIR/$name"
  mkdir -p "$variant_dir"
  LISTED_RUN="$variant_dir" B1_RUN="$OLD_RUN" bash "$HNG/configs/attach_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"
  manifest="$SAMPLER_DIR/${name}.jsonl"
  skip_flags=()
  if [[ "${SKIP_TRAIN:-0}" == "1" && -f "$variant_dir/listed/adapter/adapter_config.json" ]]; then
    skip_flags+=(--skip_train)
  fi
  log "[launch] variant=$name manifest=$manifest"
  set +e
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
    --input_file "$TASK_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$variant_dir" \
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
    --seed "$SEED" \
    --listed_negative_source score_hard \
    --score_hard_manifest "$manifest" \
    --raw_dir "$RAW_DIR" \
    --save_steps "${SAVE_STEPS:-10}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    "${TRAIN_FLAGS[@]}" \
    "${skip_flags[@]}" \
    2>&1 | tee -a "$RUN_DIR/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  log "[launch] variant=$name finished exit=$rc $(date --iso-8601=seconds)"
  if [[ "$rc" -ne 0 ]]; then
    status="$rc"
    break
  fi
  LISTED_RUN="$variant_dir" B1_RUN="$OLD_RUN" INPUT_FILE="$TASK_FILE" S1_ADAPTER="$S1_ADAPTER" \
    bash "$HNG/configs/compare_listed_to_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"
done

if [[ "$status" -eq 0 && "$VARIANT" == "all" ]]; then
  log "[compare] three-arm smoke vs frozen B1"
  "$PYTHON" "$HNG/scripts/compare_shared_pool_sampler_runs.py" \
    --run_dir "$RUN_DIR" \
    --output "$RUN_DIR/comparison.json" 2>&1 | tee -a "$RUN_DIR/run.log"
fi
exit "$status"
