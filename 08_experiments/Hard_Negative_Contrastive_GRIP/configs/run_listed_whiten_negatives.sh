#!/usr/bin/env bash
# Experiment I2: same listed/embed_sim recipe as run_listed_embed_negatives.sh, but
# the Stage-1 relation table is first passed through a linear transform (default:
# partial PCA-whitening, full rank, alpha=0.5) because the raw table lives in a
# narrow cone (mean pairwise cosine ~0.88) and the cosine softmax is therefore
# nearly uniform over the pool.
#
# Only the space changes: same Stage-1 adapter, same 198 train-graph vocabulary,
# same pool/temperature/seed/loss. B1 is frozen, listed is retrained.
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

EMBED_RAW_FILE="${EMBED_RAW_FILE:-$HNG/results/relation_embeddings/s1_qwen7b_20260913.npz}"
EMBED_MODE="${EMBED_MODE:-pca_whiten}"
EMBED_K="${EMBED_K:-197}"
EMBED_ALPHA="${EMBED_ALPHA:-0.5}"
EMBED_REMOVE="${EMBED_REMOVE:-1}"
case "$EMBED_MODE" in
  pca_whiten) EMBED_TAG="pca_whiten_k${EMBED_K}_a${EMBED_ALPHA}" ;;
  allbuttop) EMBED_TAG="allbuttop_m${EMBED_REMOVE}" ;;
  *) EMBED_TAG="$EMBED_MODE" ;;
esac
EMBED_FILE="${EMBED_FILE:-${EMBED_RAW_FILE%.npz}_${EMBED_TAG}.npz}"

# Pool/temperature stay at the shipped embed_sim values so the transform is the
# only changed variable. EMBED_POOL_SIZE=197 turns the arm into "uniform support,
# whitened tilt", which is a different experiment.
EMBED_POOL_SIZE="${EMBED_POOL_SIZE:-40}"
EMBED_TEMPERATURE="${EMBED_TEMPERATURE:-0.1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}_qwen7b_whiten_negatives}"
LAST_RUN_FILE="$HNG/results/LAST_WHITEN_NEGATIVES_RUN.txt"
if [[ "${RESUME_LAST:-0}" == "1" ]]; then
  if [[ ! -f "$LAST_RUN_FILE" ]]; then
    echo "error: RESUME_LAST=1 but missing $LAST_RUN_FILE" >&2
    exit 1
  fi
  RUN_DIR="$(tr -d '\n' < "$LAST_RUN_FILE")"
fi
TMUX_SESSION="${TMUX_SESSION:-whiten-neg-${RUN_ID}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$S1_ADAPTER" ]]; then
  echo "error: missing Stage-1 adapter: $S1_ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "error: training input not found: $INPUT_FILE" >&2
  exit 1
fi
if [[ ! -f "$EVAL_FILE" ]]; then
  echo "error: eval file not found: $EVAL_FILE" >&2
  exit 1
fi
if [[ ! -f "$RAW_DIR/train.txt" ]]; then
  echo "error: missing $RAW_DIR/train.txt" >&2
  exit 1
fi
if [[ "${FORCE_NEW:-0}" == "1" && -e "$RUN_DIR" ]]; then
  echo "error: FORCE_NEW=1 but run directory already exists: $RUN_DIR" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR" "$(dirname "$EMBED_FILE")"
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "old_run=$OLD_RUN"
  echo "s1_adapter=$S1_ADAPTER"
  echo "listed_negative_source=embed_sim"
  echo "embed_raw_file=$EMBED_RAW_FILE"
  echo "embed_file=$EMBED_FILE"
  echo "embed_transform_mode=$EMBED_MODE"
  echo "embed_transform_k=$EMBED_K"
  echo "embed_transform_alpha=$EMBED_ALPHA"
  echo "embed_transform_remove=$EMBED_REMOVE"
  echo "embed_pool_size=$EMBED_POOL_SIZE"
  echo "embed_sample_temperature=$EMBED_TEMPERATURE"
  echo "input_file=$INPUT_FILE"
  echo "eval_file=$EVAL_FILE"
  echo "raw_dir=$RAW_DIR"
  echo "tmux_session=${TMUX_SESSION:-}"
  echo "save_steps=${SAVE_STEPS:-10}"
  echo "save_total_limit=${SAVE_TOTAL_LIMIT:-2}"
  echo "note=retrain listed only; negatives are whitened Stage-1 cosine neighbours"
  date --iso-8601=seconds
  "$PYTHON" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
} >> "$RUN_DIR/environment.txt" 2>&1

if [[ ! -f "$EMBED_RAW_FILE" ]]; then
  echo "[launch] precompute raw relation embeddings -> $EMBED_RAW_FILE" | tee -a "$RUN_DIR/run.log"
  "$PYTHON" "$HNG/scripts/precompute_relation_embeddings.py" \
    --s1_adapter "$S1_ADAPTER" \
    --raw_dir "$RAW_DIR" \
    --output "$EMBED_RAW_FILE" \
    --model_name qwen-7b \
    --model_cache_dir "$CODE_DIR/model_cache" \
    2>&1 | tee -a "$RUN_DIR/run.log"
fi

if [[ ! -f "$EMBED_FILE" || "${FORCE_WHITEN:-0}" == "1" ]]; then
  echo "[launch] $EMBED_MODE k=$EMBED_K alpha=$EMBED_ALPHA -> $EMBED_FILE" | tee -a "$RUN_DIR/run.log"
  "$PYTHON" "$HNG/scripts/whiten_relation_embeddings.py" \
    --input "$EMBED_RAW_FILE" \
    --output "$EMBED_FILE" \
    --mode "$EMBED_MODE" \
    --k "$EMBED_K" \
    --alpha "$EMBED_ALPHA" \
    --remove "$EMBED_REMOVE" \
    --pool_size "$EMBED_POOL_SIZE" \
    --temperature "$EMBED_TEMPERATURE" \
    2>&1 | tee -a "$RUN_DIR/run.log"
fi
cp -f "${EMBED_FILE%.npz}_whiten.json" "$RUN_DIR/embed_transform.json" 2>/dev/null || true

LISTED_RUN="$RUN_DIR" B1_RUN="$OLD_RUN" bash "$HNG/configs/attach_frozen_b1.sh"

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

TRAIN_FLAGS=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--no_resume)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_FLAGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "[launch] listed whitened embed_sim negatives tmux=${TMUX_SESSION:-none}" | tee -a "$RUN_DIR/run.log"
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
  --listed_negative_source embed_sim \
  --relation_embedding_file "$EMBED_FILE" \
  --embed_pool_size "$EMBED_POOL_SIZE" \
  --embed_sample_temperature "$EMBED_TEMPERATURE" \
  --raw_dir "$RAW_DIR" \
  --save_steps "${SAVE_STEPS:-10}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  "${TRAIN_FLAGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] listed finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/run.log"
echo "$RUN_DIR" > "$LAST_RUN_FILE"
if [[ "$rc" -eq 0 ]]; then
  echo "[launch] compare listed vs frozen B1" | tee -a "$RUN_DIR/run.log"
  LISTED_RUN="$RUN_DIR" B1_RUN="$OLD_RUN" bash "$HNG/configs/compare_listed_to_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"
fi
exit "$rc"
