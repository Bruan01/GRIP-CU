#!/usr/bin/env bash
# Listed-negative experiment runner for WSL.
#
# The primary route is Structure-Counterfactual-9 + InfoNCE. The historical
# ErrorMemory-9 route remains available by setting NEGATIVE_SOURCE=error_memory
# explicitly and is treated as an ablation.
#
# This script is intentionally staged. The default STEP=preflight performs
# read-only checks only. GPU work is explicit:
#   STEP=rollout  -> legacy ErrorMemory-only B1 free-generation errors on train QA
#   STEP=score    -> frozen-B1 candidate scores with train-only filtering
#   STEP=freeze   -> CPU-only selected negative-bank construction
#   STEP=wire     -> CPU-only full-task manifest attachment check
#   STEP=train    -> Stage-2 GRIP + InfoNCE training, only with ALLOW_TRAIN=1
#
# Example (after the user has approved the corresponding expensive step):
#   STEP=score bash configs/run_structure_counterfactual_wsl.sh
#   STEP=freeze bash configs/run_structure_counterfactual_wsl.sh
#   STEP=wire bash configs/run_structure_counterfactual_wsl.sh
#   ALLOW_TRAIN=1 STEP=train SWANLAB_PROJECT=grip-negative-sampling \
#     bash configs/run_structure_counterfactual_wsl.sh
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
BASELINE_DIR="${GRIP_BASELINE_DIR:-$REPO_ROOT/13_base_method/grip-exp}"
CODE_DIR="${CODE_DIR:-$BASELINE_DIR}"
PYTHON="${PYTHON:-/home/kieran/miniconda3/envs/guardenv/bin/python}"
RECORD_SCRIPT="$HNG/scripts/record_run_artifacts.py"

STEP="${STEP:-preflight}"
SEED="${SEED:-2026}"
DATASET="${DATASET:-nell23k}"
NEGATIVE_SOURCE="${NEGATIVE_SOURCE:-structure_counterfactual}"
LISTED_NEGATIVE_K="${LISTED_NEGATIVE_K:-9}"
if [[ ! "$LISTED_NEGATIVE_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: LISTED_NEGATIVE_K must be a positive integer, got $LISTED_NEGATIVE_K" >&2
  exit 2
fi
if [[ "$NEGATIVE_SOURCE" == "error_memory" && "$LISTED_NEGATIVE_K" != "9" ]]; then
  echo "error: the historical error_memory sampler still requires LISTED_NEGATIVE_K=9" >&2
  exit 2
fi
COUNTERFACTUAL_UNIFORM_MIX="${COUNTERFACTUAL_UNIFORM_MIX:-0.3333333333333333}"
COUNTERFACTUAL_STRUCTURE_WEIGHT="${COUNTERFACTUAL_STRUCTURE_WEIGHT:-1.0}"
COUNTERFACTUAL_VIOLATION_WEIGHT="${COUNTERFACTUAL_VIOLATION_WEIGHT:-1.0}"
COUNTERFACTUAL_TEMPERATURE="${COUNTERFACTUAL_TEMPERATURE:-0.25}"
LAMBDA_CANDIDATE="${LAMBDA_CANDIDATE:-1.0}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-0}"
DEFER_EVAL="${DEFER_EVAL:-0}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-1.0}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-1e-3}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-4}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
RUN_VARIANT="${RUN_VARIANT:-}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/04_benchmark/data/$DATASET}"
MODEL_NAME="${MODEL_NAME:-qwen-7b}"
if [[ -d /mnt/e/GRIP-CU/model_cache ]]; then
  DEFAULT_MODEL_CACHE_DIR=/mnt/e/GRIP-CU/model_cache
else
  DEFAULT_MODEL_CACHE_DIR="$CODE_DIR/model_cache"
fi
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$DEFAULT_MODEL_CACHE_DIR}"
TORCH_DTYPE="${TORCH_DTYPE:-${GRIP_TORCH_DTYPE:-bfloat16}}"
INPUT_FILE="${INPUT_FILE:-$DATA_ROOT/tasks/grip_${DATASET}_tasks.json}"
EVAL_FILE="${EVAL_FILE:-$DATA_ROOT/processed/processed_test.json}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw}"
case "$DATASET" in
  nell23k)
    DEFAULT_S1_ADAPTER="$CODE_DIR/artifacts/nell23k_paper_7b/lora_adapter"
    DEFAULT_B1_ADAPTER="$CODE_DIR/artifacts/nell23k_paper_7b/lora_adapter"
    DEFAULT_EXPECTED_LISTED=3253
    ;;
  fb15k237_2)
    # E: retains FB provenance/configuration, but not adapter weight files.
    DEFAULT_S1_ADAPTER=""
    DEFAULT_B1_ADAPTER=""
    DEFAULT_EXPECTED_LISTED=3109
    ;;
  wn18rr)
    # WN18RR is the one-graph paper benchmark.  The baseline adapter is the
    # full-task, no-graph-context GRIP run; the listed-negative fork starts
    # from its Stage-1 adapter and uses WN entity text to recover KG pairs.
    DEFAULT_S1_ADAPTER="$CODE_DIR/outputs/paper_runs/wn18rr_taskfile_3090/stage1_adapter"
    DEFAULT_B1_ADAPTER="$CODE_DIR/outputs/paper_runs/wn18rr_taskfile_3090/stage2_adapter"
    DEFAULT_EXPECTED_LISTED=2709
    ;;
  codexm)
    # CoDEx-M is run through the task-file pipeline built for this benchmark.
    # Its relation-QA count is derived from the generated task artifact, so the
    # wiring gate checks complete coverage rather than a hard-coded count.
    DEFAULT_S1_ADAPTER=""
    DEFAULT_B1_ADAPTER=""
    DEFAULT_EXPECTED_LISTED=-1
    ;;
  *)
    echo "error: unsupported DATASET=$DATASET (expected nell23k, fb15k237_2, wn18rr, or codexm)" >&2
    exit 1
    ;;
esac
if [[ "$DATASET" == "wn18rr" ]]; then
  DEFAULT_TRAIN_LORA_R=24
  DEFAULT_TRAIN_LORA_ALPHA=48
  DEFAULT_INVOLVE_QA_EPOCHS=5
  DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=4
  DEFAULT_GRADIENT_ACCUMULATION_STEPS=128
else
  DEFAULT_TRAIN_LORA_R=4
  DEFAULT_TRAIN_LORA_ALPHA=8
  DEFAULT_INVOLVE_QA_EPOCHS=10
  DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=1
  DEFAULT_GRADIENT_ACCUMULATION_STEPS=512
fi
TRAIN_LORA_R="${TRAIN_LORA_R:-$DEFAULT_TRAIN_LORA_R}"
TRAIN_LORA_ALPHA="${TRAIN_LORA_ALPHA:-$DEFAULT_TRAIN_LORA_ALPHA}"
INVOLVE_QA_EPOCHS="${INVOLVE_QA_EPOCHS:-$DEFAULT_INVOLVE_QA_EPOCHS}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
S1_ADAPTER="${S1_ADAPTER:-$DEFAULT_S1_ADAPTER}"
B1_ADAPTER="${B1_ADAPTER:-$DEFAULT_B1_ADAPTER}"
RELATION_TEXT_FILE="${RELATION_TEXT_FILE:-}"
ENTITY_TEXT_FILE="${ENTITY_TEXT_FILE:-}"
if [[ "$DATASET" == "codexm" ]]; then
  if [[ -z "$RELATION_TEXT_FILE" ]]; then
    RELATION_TEXT_FILE="$RAW_DIR/rel2text.json"
  fi
  if [[ -z "$ENTITY_TEXT_FILE" ]]; then
    ENTITY_TEXT_FILE="$RAW_DIR/entity2text.txt"
  fi
fi
if [[ "$DATASET" == "wn18rr" && -z "$ENTITY_TEXT_FILE" ]]; then
  ENTITY_TEXT_FILE="$RAW_DIR/entity2text.txt"
fi
if [[ "$DATASET" == "nell23k" ]]; then
  DEFAULT_ARTIFACT_ROOT="$HNG/results/runs/structure_counterfactual_artifacts"
else
  DEFAULT_ARTIFACT_ROOT="$HNG/results/runs/${DATASET}_structure_counterfactual_artifacts"
fi
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$DEFAULT_ARTIFACT_ROOT}"
SCORES_DIR="${SCORES_DIR:-$ARTIFACT_ROOT/b1_train_scores}"
ROLLOUT_MANIFEST="${ROLLOUT_MANIFEST:-$ARTIFACT_ROOT/b1_train_rollout.jsonl}"
slug_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/x}"
  value="$(printf '%s' "$value" | sed -E 's/[^[:alnum:]_]/_/g')"
  printf '%s' "$value"
}
if [[ "$NEGATIVE_SOURCE" == "structure_counterfactual" ]]; then
  SAMPLER_TAG=""
  if [[ "$COUNTERFACTUAL_UNIFORM_MIX" != "0.3333333333333333" || \
        "$COUNTERFACTUAL_STRUCTURE_WEIGHT" != "1.0" || \
        "$COUNTERFACTUAL_VIOLATION_WEIGHT" != "1.0" || \
        "$COUNTERFACTUAL_TEMPERATURE" != "0.25" || "$SEED" != "2026" ]]; then
    SAMPLER_TAG="_mix$(slug_value "$COUNTERFACTUAL_UNIFORM_MIX")_struct$(slug_value "$COUNTERFACTUAL_STRUCTURE_WEIGHT")_viol$(slug_value "$COUNTERFACTUAL_VIOLATION_WEIGHT")_temp$(slug_value "$COUNTERFACTUAL_TEMPERATURE")_s${SEED}"
  fi
  if [[ "$LISTED_NEGATIVE_K" == "9" && -z "$SAMPLER_TAG" ]]; then
    DEFAULT_SAMPLER_DIR="$ARTIFACT_ROOT/structure_counterfactual_bank"
  else
    DEFAULT_SAMPLER_DIR="$ARTIFACT_ROOT/structure_counterfactual_bank_k${LISTED_NEGATIVE_K}${SAMPLER_TAG}"
  fi
  DEFAULT_MANIFEST_NAME="counterfactual_bank.jsonl"
else
  DEFAULT_SAMPLER_DIR="$ARTIFACT_ROOT/error_memory_bank"
  DEFAULT_MANIFEST_NAME="error_memory_k.jsonl"
fi
SAMPLER_DIR="${SAMPLER_DIR:-$DEFAULT_SAMPLER_DIR}"
BANK_SOURCE_DIR="${BANK_SOURCE_DIR:-}"
MANIFEST="${MANIFEST:-$SAMPLER_DIR/$DEFAULT_MANIFEST_NAME}"
COUNTERFACTUAL_METADATA="${COUNTERFACTUAL_METADATA:-$SAMPLER_DIR/metadata.json}"
EXPECTED_LISTED="${EXPECTED_LISTED:-$DEFAULT_EXPECTED_LISTED}"
CONFIG_SUFFIX=""
if [[ "$LISTED_NEGATIVE_K" != "9" || "$SEED" != "2026" || \
      "$MODEL_NAME" != "qwen-7b" || "$TORCH_DTYPE" != "bfloat16" || \
      "$LAMBDA_CANDIDATE" != "1.0" || "$CONTRASTIVE_TEMPERATURE" != "1.0" || \
      "$TRAIN_LEARNING_RATE" != "1e-3" || "$TRAIN_WEIGHT_DECAY" != "1e-4" || \
      "$TRAIN_MAX_GRAD_NORM" != "1.0" || \
      "$COUNTERFACTUAL_UNIFORM_MIX" != "0.3333333333333333" || \
      "$COUNTERFACTUAL_STRUCTURE_WEIGHT" != "1.0" || \
      "$COUNTERFACTUAL_VIOLATION_WEIGHT" != "1.0" || \
      "$COUNTERFACTUAL_TEMPERATURE" != "0.25" || \
      "$INVOLVE_QA_EPOCHS" != "$DEFAULT_INVOLVE_QA_EPOCHS" || \
      "$TRAIN_LORA_R" != "$DEFAULT_TRAIN_LORA_R" || \
      "$TRAIN_LORA_ALPHA" != "$DEFAULT_TRAIN_LORA_ALPHA" || \
      "$PER_DEVICE_TRAIN_BATCH_SIZE" != "$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE" || \
      "$GRADIENT_ACCUMULATION_STEPS" != "$DEFAULT_GRADIENT_ACCUMULATION_STEPS" || \
      -n "$RUN_VARIANT" ]]; then
  CONFIG_SUFFIX="_s${SEED}_model$(slug_value "$MODEL_NAME")_dtype$(slug_value "$TORCH_DTYPE")_lc$(slug_value "$LAMBDA_CANDIDATE")_ct$(slug_value "$CONTRASTIVE_TEMPERATURE")_lr$(slug_value "$TRAIN_LEARNING_RATE")_wd$(slug_value "$TRAIN_WEIGHT_DECAY")_gn$(slug_value "$TRAIN_MAX_GRAD_NORM")_ep$(slug_value "$INVOLVE_QA_EPOCHS")_r${TRAIN_LORA_R}_a${TRAIN_LORA_ALPHA}_b${PER_DEVICE_TRAIN_BATCH_SIZE}_g${GRADIENT_ACCUMULATION_STEPS}"
  if [[ "$NEGATIVE_SOURCE" == "structure_counterfactual" ]]; then
    CONFIG_SUFFIX+="_mix$(slug_value "$COUNTERFACTUAL_UNIFORM_MIX")_struct$(slug_value "$COUNTERFACTUAL_STRUCTURE_WEIGHT")_viol$(slug_value "$COUNTERFACTUAL_VIOLATION_WEIGHT")_st$(slug_value "$COUNTERFACTUAL_TEMPERATURE")"
  fi
  if [[ -n "$RUN_VARIANT" ]]; then
    CONFIG_SUFFIX+="_$(slug_value "$RUN_VARIANT")"
  fi
fi
RUN_PREFIX="$NEGATIVE_SOURCE"
if [[ "$DATASET" != "nell23k" ]]; then
  RUN_PREFIX="${DATASET}_${NEGATIVE_SOURCE}"
fi
RUN_ID="${RUN_ID:-${RUN_PREFIX}_k${LISTED_NEGATIVE_K}_full_$(date +%Y%m%d)${CONFIG_SUFFIX}}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/$RUN_ID}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$RUN_ID}"
SWANLAB_GROUP="${SWANLAB_GROUP:-${NEGATIVE_SOURCE}-k${LISTED_NEGATIVE_K}}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"

if [[ "$NEGATIVE_SOURCE" != "error_memory" && "$NEGATIVE_SOURCE" != "structure_counterfactual" ]]; then
  echo "error: NEGATIVE_SOURCE must be error_memory or structure_counterfactual" >&2
  exit 1
fi
if [[ -n "$BANK_SOURCE_DIR" && "$NEGATIVE_SOURCE" != "structure_counterfactual" ]]; then
  echo "error: BANK_SOURCE_DIR is only supported for structure_counterfactual banks" >&2
  exit 2
fi

case "$TORCH_DTYPE" in
  bfloat16|float16|float32) ;;
  *) echo "error: unsupported TORCH_DTYPE=$TORCH_DTYPE (choose bfloat16, float16, or float32)" >&2; exit 1 ;;
esac

if [[ "$STEP" != "preflight" && "$STEP" != "rollout" && "$STEP" != "score" && \
      "$STEP" != "freeze" && "$STEP" != "wire" && "$STEP" != "train" ]]; then
  echo "error: STEP must be preflight, rollout, score, freeze, wire, or train" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "error: WSL Python environment not found: $PYTHON" >&2
  exit 1
fi
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export GRIP_BASELINE_DIR="$BASELINE_DIR"
export GRIP_TORCH_DTYPE="$TORCH_DTYPE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export STEP SEED DATASET NEGATIVE_SOURCE DATA_ROOT MODEL_NAME MODEL_CACHE_DIR
export TORCH_DTYPE INPUT_FILE EVAL_FILE RAW_DIR S1_ADAPTER B1_ADAPTER
export ENTITY_TEXT_FILE
export ARTIFACT_ROOT SCORES_DIR ROLLOUT_MANIFEST SAMPLER_DIR MANIFEST
export COUNTERFACTUAL_METADATA EXPECTED_LISTED RUN_ID RUN_DIR BANK_SOURCE_DIR
export SWANLAB_PROJECT SWANLAB_EXPERIMENT_NAME SWANLAB_GROUP SWANLAB_MODE
export LISTED_NEGATIVE_K STOP_AFTER_STEP DEFER_EVAL COUNTERFACTUAL_UNIFORM_MIX
export COUNTERFACTUAL_STRUCTURE_WEIGHT COUNTERFACTUAL_VIOLATION_WEIGHT
export COUNTERFACTUAL_TEMPERATURE LAMBDA_CANDIDATE CONTRASTIVE_TEMPERATURE
export TRAIN_LEARNING_RATE TRAIN_WEIGHT_DECAY TRAIN_MAX_GRAD_NORM
export TRAIN_LORA_R TRAIN_LORA_ALPHA INVOLVE_QA_EPOCHS
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS RUN_VARIANT
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
export CANDIDATE_BATCH_SIZE PROGRESS_EVERY
RELATION_TEXT_ARGS=()
if [[ -n "$RELATION_TEXT_FILE" ]]; then
  RELATION_TEXT_ARGS+=(--relation_text_file "$RELATION_TEXT_FILE")
fi
ENTITY_TEXT_ARGS=()
if [[ -n "$ENTITY_TEXT_FILE" ]]; then
  ENTITY_TEXT_ARGS+=(--entity_text_file "$ENTITY_TEXT_FILE")
fi
ELIGIBILITY_ARGS=()
if [[ -n "${RELATION_ELIGIBILITY:-}" ]]; then
  ELIGIBILITY_ARGS+=(--relation_eligibility "$RELATION_ELIGIBILITY")
fi

RUN_COMMAND="${RUN_COMMAND:-STEP=$STEP DATASET=$DATASET NEGATIVE_SOURCE=$NEGATIVE_SOURCE LISTED_NEGATIVE_K=$LISTED_NEGATIVE_K BANK_SOURCE_DIR=$BANK_SOURCE_DIR SAMPLER_DIR=$SAMPLER_DIR RUN_ID=$RUN_ID RUN_VARIANT=$RUN_VARIANT SEED=$SEED LAMBDA_CANDIDATE=$LAMBDA_CANDIDATE STOP_AFTER_STEP=$STOP_AFTER_STEP DEFER_EVAL=$DEFER_EVAL CONTRASTIVE_TEMPERATURE=$CONTRASTIVE_TEMPERATURE TRAIN_LEARNING_RATE=$TRAIN_LEARNING_RATE TRAIN_WEIGHT_DECAY=$TRAIN_WEIGHT_DECAY TRAIN_MAX_GRAD_NORM=$TRAIN_MAX_GRAD_NORM COUNTERFACTUAL_UNIFORM_MIX=$COUNTERFACTUAL_UNIFORM_MIX COUNTERFACTUAL_STRUCTURE_WEIGHT=$COUNTERFACTUAL_STRUCTURE_WEIGHT COUNTERFACTUAL_VIOLATION_WEIGHT=$COUNTERFACTUAL_VIOLATION_WEIGHT COUNTERFACTUAL_TEMPERATURE=$COUNTERFACTUAL_TEMPERATURE ARTIFACT_ROOT=$ARTIFACT_ROOT MODEL_CACHE_DIR=$MODEL_CACHE_DIR bash $0 $*}"
RECORD_PATH_ARGS=(
  --input_file "$INPUT_FILE"
  --eval_file "$EVAL_FILE"
  --raw_dir "$RAW_DIR"
  --model_cache_dir "$MODEL_CACHE_DIR"
  --scores_dir "$SCORES_DIR"
  --benchmark_dir "$ARTIFACT_ROOT/score_benchmark"
  --rollout_manifest "$ROLLOUT_MANIFEST"
  --sampler_dir "$SAMPLER_DIR"
  --manifest "$MANIFEST"
  --counterfactual_metadata "$COUNTERFACTUAL_METADATA"
)
if [[ -n "$RELATION_TEXT_FILE" ]]; then
  RECORD_PATH_ARGS+=(--relation_text_file "$RELATION_TEXT_FILE")
fi
if [[ -n "$ENTITY_TEXT_FILE" ]]; then
  RECORD_PATH_ARGS+=(--entity_text_file "$ENTITY_TEXT_FILE")
fi
if [[ -n "$S1_ADAPTER" ]]; then
  RECORD_PATH_ARGS+=(--s1_adapter "$S1_ADAPTER")
fi
if [[ -n "$B1_ADAPTER" ]]; then
  RECORD_PATH_ARGS+=(--b1_adapter "$B1_ADAPTER")
fi
if [[ -n "$BANK_SOURCE_DIR" ]]; then
  RECORD_PATH_ARGS+=(--bank_source_dir "$BANK_SOURCE_DIR")
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "error: missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "error: missing required directory: $1" >&2
    exit 1
  fi
}

require_adapter_dir() {
  local label="$1"
  local path="$2"
  # A config-only directory is not a usable LoRA adapter. Check the weight
  # file here so an expensive GPU stage cannot start with incomplete assets.
  if [[ -z "$path" ]]; then
    echo "error: $label adapter is not configured; set its path explicitly" >&2
    exit 1
  fi
  require_file "$path/adapter_config.json"
  require_file "$path/adapter_model.safetensors"
}

check_cuda_and_model() {
  if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
    echo "error: PyTorch cannot see a CUDA device" >&2
    exit 1
  fi
  local model_dir="$MODEL_CACHE_DIR/Qwen--Qwen2.5-7B-Instruct"
  if ! PYTHONPATH="$CODE_DIR" "$PYTHON" -c "
from pathlib import Path
from models.utils import _is_complete_model_dir
raise SystemExit(0 if _is_complete_model_dir(Path(r'$model_dir')) else 1)
"; then
    echo "error: incomplete Qwen2.5-7B cache: $model_dir" >&2
    exit 1
  fi
}

write_environment_record() {
  mkdir -p "$RUN_DIR"
  local git_head=""
  local path=""
  local -a recorded_paths=()
  local -A hashed_paths=()
  git_head="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
  # Windows-created worktrees keep a Windows gitdir pointer that WSL git
  # cannot resolve. Fall back to the Windows Git executable so the run
  # provenance still records the exact commit.
  if [[ -z "$git_head" ]] && command -v git.exe >/dev/null 2>&1; then
    local repo_win=""
    repo_win="$(wslpath -w "$REPO_ROOT" 2>/dev/null || true)"
    if [[ -n "$repo_win" ]]; then
      git_head="$(git.exe -C "$repo_win" rev-parse HEAD 2>/dev/null || true)"
    fi
  fi
  {
    echo "step=$STEP"
    echo "run_dir=$RUN_DIR"
    echo "git_head=$git_head"
    echo "baseline_dir=$BASELINE_DIR"
    echo "code_dir=$CODE_DIR"
    echo "input_file=$INPUT_FILE"
    echo "eval_file=$EVAL_FILE"
    echo "raw_dir=$RAW_DIR"
    echo "negative_source=$NEGATIVE_SOURCE"
    echo "listed_negative_k=$LISTED_NEGATIVE_K"
    echo "relation_text_file=$RELATION_TEXT_FILE"
    echo "entity_text_file=$ENTITY_TEXT_FILE"
    echo "s1_adapter=$S1_ADAPTER"
    echo "b1_adapter=$B1_ADAPTER"
    echo "scores_dir=$SCORES_DIR"
    echo "rollout_manifest=$ROLLOUT_MANIFEST"
    echo "sampler_dir=$SAMPLER_DIR"
    echo "bank_source_dir=$BANK_SOURCE_DIR"
    echo "counterfactual_metadata=$COUNTERFACTUAL_METADATA"
    echo "seed=$SEED"
    echo "lambda_candidate=$LAMBDA_CANDIDATE"
    echo "stop_after_step=$STOP_AFTER_STEP"
    echo "defer_eval=$DEFER_EVAL"
    echo "contrastive_temperature=$CONTRASTIVE_TEMPERATURE"
    echo "train_learning_rate=$TRAIN_LEARNING_RATE"
    echo "train_weight_decay=$TRAIN_WEIGHT_DECAY"
    echo "train_max_grad_norm=$TRAIN_MAX_GRAD_NORM"
    echo "involve_qa_epochs=$INVOLVE_QA_EPOCHS"
    echo "train_lora_r=$TRAIN_LORA_R"
    echo "train_lora_alpha=$TRAIN_LORA_ALPHA"
    echo "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
    echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
    echo "swanlab_project=$SWANLAB_PROJECT"
    echo "swanlab_group=$SWANLAB_GROUP"
    echo "swanlab_mode=$SWANLAB_MODE"
    date --iso-8601=seconds
    uname -a
    "$PYTHON" -c "import sys,torch; print('python', sys.version); print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
    recorded_paths=(
      "$INPUT_FILE" "$EVAL_FILE" "$RAW_DIR/train.txt" "$ENTITY_TEXT_FILE"
      "$S1_ADAPTER/adapter_model.safetensors" "$B1_ADAPTER/adapter_model.safetensors"
      "$SCORES_DIR/candidate_scores.jsonl" "$SCORES_DIR/metadata.json"
      "$ROLLOUT_MANIFEST" "$MANIFEST" "$COUNTERFACTUAL_METADATA"
    )
    if [[ -n "$BANK_SOURCE_DIR" ]]; then
      recorded_paths+=(
        "$BANK_SOURCE_DIR/counterfactual_bank.jsonl"
        "$BANK_SOURCE_DIR/metadata.json"
      )
    fi
    for path in "${recorded_paths[@]}"; do
      if [[ -f "$path" && -z "${hashed_paths[$path]+seen}" ]]; then
        sha256sum "$path"
        hashed_paths["$path"]=1
      fi
    done
  } >> "$RUN_DIR/environment.txt"
}

check_common_assets() {
  require_file "$INPUT_FILE"
  require_file "$RAW_DIR/train.txt"
  if [[ -n "$RELATION_TEXT_FILE" ]]; then require_file "$RELATION_TEXT_FILE"; fi
  if [[ -n "$ENTITY_TEXT_FILE" ]]; then require_file "$ENTITY_TEXT_FILE"; fi
  require_dir "$HNG/src"
  require_dir "$BASELINE_DIR"
}

check_train_assets() {
  check_common_assets
  require_file "$EVAL_FILE"
  require_adapter_dir "S1" "$S1_ADAPTER"
  require_file "$MANIFEST"
  check_cuda_and_model
}

enable_stage_log() {
  mkdir -p "$RUN_DIR"
  # Keep one durable launcher log per stage.  The stage recorder also stores
  # structured status, while this captures validation errors and stdout/stderr.
  exec > >(tee -a "$RUN_DIR/${STEP}.log") 2>&1
}

record_stage_start() {
  "$PYTHON" "$RECORD_SCRIPT" \
    --run_dir "$RUN_DIR" \
    --repo_root "$REPO_ROOT" \
    --baseline_dir "$BASELINE_DIR" \
    --phase start \
    --step "$STEP" \
    --status running \
    --exit_code 0 \
    --run_id "$RUN_ID" \
    --dataset "$DATASET" \
    --negative_source "$NEGATIVE_SOURCE" \
    --command "$RUN_COMMAND" \
    "${RECORD_PATH_ARGS[@]}"
}

record_stage_finish() {
  local code=$?
  trap - EXIT
  local stage_status=completed
  if [[ "$code" -ne 0 ]]; then
    stage_status=failed
  fi
  "$PYTHON" "$RECORD_SCRIPT" \
    --run_dir "$RUN_DIR" \
    --repo_root "$REPO_ROOT" \
    --baseline_dir "$BASELINE_DIR" \
    --phase finish \
    --step "$STEP" \
    --status "$stage_status" \
    --exit_code "$code" \
    --run_id "$RUN_ID" \
    --dataset "$DATASET" \
    --negative_source "$NEGATIVE_SOURCE" \
    --command "$RUN_COMMAND" \
    --hash_large \
    "${RECORD_PATH_ARGS[@]}" || true
  exit "$code"
}

start_stage_recording() {
  enable_stage_log
  record_stage_start
  trap 'record_stage_finish' EXIT
}

case "$STEP" in
  preflight)
    start_stage_recording
    check_common_assets
    echo "[preflight] read-only algorithm preflight passed"
    echo "[preflight] code=$HNG"
    if [[ "$NEGATIVE_SOURCE" == "structure_counterfactual" ]]; then
      echo "[preflight] mainline steps: score -> freeze -> wire -> train (K=$LISTED_NEGATIVE_K)"
      echo "[preflight] rollout is not required for structure counterfactuals"
    else
      echo "[preflight] ablation steps: rollout -> score -> freeze -> wire -> train"
    fi
    echo "[preflight] no GPU work was started"
    ;;
  rollout)
    start_stage_recording
    if [[ "$NEGATIVE_SOURCE" == "structure_counterfactual" ]]; then
      echo "error: rollout is not part of the structure-counterfactual mainline; use STEP=score" >&2
      exit 2
    fi
    check_common_assets
    require_adapter_dir "B1" "$B1_ADAPTER"
    check_cuda_and_model
    mkdir -p "$(dirname "$ROLLOUT_MANIFEST")"
    cd "$CODE_DIR"
    "$PYTHON" "$HNG/scripts/mine_rollout_hard_negatives.py" \
      --input_file "$INPUT_FILE" \
      --b1_adapter "$B1_ADAPTER" \
      --raw_dir "$RAW_DIR" \
      "${RELATION_TEXT_ARGS[@]}" \
      --output "$ROLLOUT_MANIFEST" \
      --model_name "$MODEL_NAME" \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --dtype "$TORCH_DTYPE" \
      --total_k 9 \
      --max_new_tokens "${MAX_NEW_TOKENS:-32}" \
      --seed "$SEED" \
      --filter_splits train \
      --progress_every "${PROGRESS_EVERY:-25}" \
      --resume
    ;;
  score)
    start_stage_recording
    check_common_assets
    require_adapter_dir "B1" "$B1_ADAPTER"
    check_cuda_and_model
    mkdir -p "$SCORES_DIR"
    cd "$CODE_DIR"
    "$PYTHON" "$HNG/scripts/score_relation_candidates_offline.py" \
      --task_file "$INPUT_FILE" \
      "${ELIGIBILITY_ARGS[@]}" \
      --raw_dir "$RAW_DIR" \
      "${RELATION_TEXT_ARGS[@]}" \
      "${ENTITY_TEXT_ARGS[@]}" \
      --b1_adapter "$B1_ADAPTER" \
      --output_dir "$SCORES_DIR" \
      --model_name "$MODEL_NAME" \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --dtype "$TORCH_DTYPE" \
      --split train \
      --filter_splits train \
      --candidate_batch_size "${CANDIDATE_BATCH_SIZE:-8}" \
      --temperature "${SCORE_TEMPERATURE:-1.0}" \
      --seed "$SEED" \
      --progress_every "${PROGRESS_EVERY:-10}" \
      --resume
    ;;
  freeze)
    start_stage_recording
    check_common_assets
    if [[ "$NEGATIVE_SOURCE" == "structure_counterfactual" && -n "$BANK_SOURCE_DIR" ]]; then
      require_file "$BANK_SOURCE_DIR/counterfactual_bank.jsonl"
      require_file "$BANK_SOURCE_DIR/metadata.json"
      require_file "$BANK_SOURCE_DIR/sampler_audit.jsonl"
      require_file "$BANK_SOURCE_DIR/relation_profiles.json"
    else
      require_file "$SCORES_DIR/candidate_scores.jsonl"
      require_file "$SCORES_DIR/metadata.json"
    fi
    mkdir -p "$SAMPLER_DIR"
    FREEZE_FLAGS=()
    if [[ "${OVERWRITE_BANK:-0}" == "1" ]]; then
      FREEZE_FLAGS+=(--overwrite)
    fi
    if [[ "$NEGATIVE_SOURCE" == "error_memory" ]]; then
      require_file "$ROLLOUT_MANIFEST"
      "$PYTHON" "$HNG/scripts/mine_error_memory_negatives.py" \
        --scores "$SCORES_DIR/candidate_scores.jsonl" \
        --metadata "$SCORES_DIR/metadata.json" \
        --rollout_manifest "$ROLLOUT_MANIFEST" \
        --output_dir "$SAMPLER_DIR" \
        --raw_dir "$RAW_DIR" \
        "${RELATION_TEXT_ARGS[@]}" \
        --total_k 9 \
        --uniform_mix "${ERROR_MEMORY_UNIFORM_MIX:-0.3333333333333333}" \
        --error_weight "${ERROR_MEMORY_WEIGHT:-1.0}" \
        --violation_weight "${ERROR_MEMORY_VIOLATION_WEIGHT:-1.0}" \
        --sampling_temperature "${ERROR_MEMORY_TEMPERATURE:-0.25}" \
        --seed "$SEED" \
        "${FREEZE_FLAGS[@]}"
    else
      COUNTERFACTUAL_FLAGS=()
      if [[ "$DATASET" == "wn18rr" ]]; then
        COUNTERFACTUAL_FLAGS+=(--strip_first_underscore)
      fi
      if [[ -n "$BANK_SOURCE_DIR" ]]; then
        require_file "$BANK_SOURCE_DIR/counterfactual_bank.jsonl"
        require_file "$BANK_SOURCE_DIR/metadata.json"
        require_file "$BANK_SOURCE_DIR/sampler_audit.jsonl"
        require_file "$BANK_SOURCE_DIR/relation_profiles.json"
        "$PYTHON" "$HNG/scripts/derive_structure_counterfactual_bank.py" \
          --source_dir "$BANK_SOURCE_DIR" \
          --output_dir "$SAMPLER_DIR" \
          --total_k "$LISTED_NEGATIVE_K" \
          "${FREEZE_FLAGS[@]}"
      else
        "$PYTHON" "$HNG/scripts/build_structure_counterfactual_bank.py" \
          --scores "$SCORES_DIR/candidate_scores.jsonl" \
          --raw_dir "$RAW_DIR" \
          --output_dir "$SAMPLER_DIR" \
          "${RELATION_TEXT_ARGS[@]}" \
          --total_k "$LISTED_NEGATIVE_K" \
          --uniform_mix "$COUNTERFACTUAL_UNIFORM_MIX" \
          --structure_weight "$COUNTERFACTUAL_STRUCTURE_WEIGHT" \
          --violation_weight "$COUNTERFACTUAL_VIOLATION_WEIGHT" \
          --temperature "$COUNTERFACTUAL_TEMPERATURE" \
          --seed "$SEED" \
          "${COUNTERFACTUAL_FLAGS[@]}" \
          "${FREEZE_FLAGS[@]}"
      fi
    fi
    ;;
  wire)
    start_stage_recording
    check_common_assets
    require_file "$MANIFEST"
    WIRE_OUTPUT="${WIRE_OUTPUT:-$RUN_DIR/wiring.json}"
    mkdir -p "$(dirname "$WIRE_OUTPUT")"
    "$PYTHON" "$HNG/scripts/verify_error_memory_wiring.py" \
      --task_file "$INPUT_FILE" \
      "${ELIGIBILITY_ARGS[@]}" \
      --manifest "$MANIFEST" \
      --raw_dir "$RAW_DIR" \
      "${RELATION_TEXT_ARGS[@]}" \
      --expected_listed "$EXPECTED_LISTED" \
      --listed_negative_k "$LISTED_NEGATIVE_K" \
      --negative_source "$NEGATIVE_SOURCE" \
      --output "$WIRE_OUTPUT"
    if [[ "$WIRE_OUTPUT" != "$RUN_DIR/wiring.json" ]]; then
      cp -f "$WIRE_OUTPUT" "$RUN_DIR/wiring.json"
    fi
    if [[ "$WIRE_OUTPUT" != "$SAMPLER_DIR/wiring.json" ]]; then
      mkdir -p "$SAMPLER_DIR"
      cp -f "$WIRE_OUTPUT" "$SAMPLER_DIR/wiring.json"
    fi
    ;;
  train)
    if [[ "${SKIP_TMUX:-0}" != "1" ]]; then
      TMUX_SESSION="${TMUX_SESSION:-$RUN_ID}"
      # shellcheck source=tmux_guard.sh
      source "$HNG/configs/tmux_guard.sh"
      tmux_guard_reexec "$0" "$@"
    fi
    if [[ -e "$RUN_DIR" && "${NO_RESUME:-0}" == "1" ]]; then
      echo "error: NO_RESUME=1 cannot reuse an existing RUN_DIR: $RUN_DIR" >&2
      exit 1
    fi
    # An interrupted listed run is intentionally restarted with the same
    # directory. train_listed_contrastive.py resumes its latest Trainer
    # checkpoint and rewrites only atomic metadata/prediction artifacts.
    mkdir -p "$RUN_DIR"
    start_stage_recording
    if [[ "${ALLOW_TRAIN:-0}" != "1" ]]; then
      echo "error: training is gated; set ALLOW_TRAIN=1 only after reviewing the preflight/wiring records" >&2
      exit 2
    fi
    check_train_assets
    write_environment_record
    "$PYTHON" "$HNG/scripts/verify_error_memory_wiring.py" \
      --task_file "$INPUT_FILE" \
      "${ELIGIBILITY_ARGS[@]}" \
      --manifest "$MANIFEST" \
      --raw_dir "$RAW_DIR" \
      "${RELATION_TEXT_ARGS[@]}" \
      --expected_listed "$EXPECTED_LISTED" \
      --listed_negative_k "$LISTED_NEGATIVE_K" \
      --negative_source "$NEGATIVE_SOURCE" \
      --output "$RUN_DIR/wiring.json" | tee "$RUN_DIR/wiring.log"

    "$PYTHON" - <<PY
import json
import sys
from pathlib import Path
sys.path.insert(0, "$HNG/src")
from hard_negative_grip.training_budget import refuse_underfit_listed_budget
from hard_negative_grip.task_file import is_grip_task_file, load_json_payload
payload = load_json_payload(Path("$INPUT_FILE"))
if not is_grip_task_file(payload):
    raise SystemExit("input task is not a GRIP task file")
n_qa = len(payload["qa_samples"])
if n_qa <= int("$EXPECTED_LISTED"):
    raise SystemExit(f"full paper task required; got {n_qa} QA rows")
budget = refuse_underfit_listed_budget(n_qa)
Path("$RUN_DIR/budget.json").write_text(json.dumps(budget, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(budget, ensure_ascii=False), flush=True)
PY

    SWANLAB_FLAGS=()
    if [[ -n "$SWANLAB_PROJECT" ]]; then
      SWANLAB_FLAGS+=(--swanlab_project "$SWANLAB_PROJECT" --swanlab_experiment_name "$SWANLAB_EXPERIMENT_NAME" --swanlab_group "$SWANLAB_GROUP" --swanlab_mode "$SWANLAB_MODE")
    fi
    TRAIN_FLAGS=()
    if [[ "${NO_RESUME:-0}" == "1" ]]; then TRAIN_FLAGS+=(--no_resume); fi
    if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then TRAIN_FLAGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"); fi
    if [[ "${SKIP_TRAIN:-0}" == "1" && -f "$RUN_DIR/listed/adapter/adapter_config.json" ]]; then TRAIN_FLAGS+=(--skip_train); fi
    if [[ "$DEFER_EVAL" == "1" ]]; then TRAIN_FLAGS+=(--defer_eval); fi
    NEGATIVE_TRAIN_FLAGS=()
    if [[ "$NEGATIVE_SOURCE" == "error_memory" ]]; then
      NEGATIVE_TRAIN_FLAGS+=(
        --listed_negative_source error_memory
        --negative_sampling_mode error_memory
        --error_memory_manifest "$MANIFEST"
        --error_memory_uniform_mix "${ERROR_MEMORY_UNIFORM_MIX:-0.3333333333333333}"
        --error_memory_weight "${ERROR_MEMORY_WEIGHT:-1.0}"
        --error_memory_violation_weight "${ERROR_MEMORY_VIOLATION_WEIGHT:-1.0}"
        --error_memory_temperature "${ERROR_MEMORY_TEMPERATURE:-0.25}"
      )
    else
      NEGATIVE_TRAIN_FLAGS+=(
        --listed_negative_source structure_counterfactual
        --negative_sampling_mode structure_counterfactual
        --counterfactual_manifest "$MANIFEST"
        --counterfactual_metadata "$COUNTERFACTUAL_METADATA"
        --counterfactual_uniform_mix "$COUNTERFACTUAL_UNIFORM_MIX"
        --counterfactual_structure_weight "$COUNTERFACTUAL_STRUCTURE_WEIGHT"
        --counterfactual_violation_weight "$COUNTERFACTUAL_VIOLATION_WEIGHT"
        --counterfactual_temperature "$COUNTERFACTUAL_TEMPERATURE"
      )
    fi
    cd "$CODE_DIR"
    # SwanLab 0.10.x treats SWANLAB_PROJECT as a nested JSON settings object.
    # The trainer receives the same values explicitly through SWANLAB_FLAGS, so
    # hide the shell exports from the Python process while retaining them for
    # tmux re-exec and artifact recording.
    env -u SWANLAB_PROJECT -u SWANLAB_EXPERIMENT_NAME -u SWANLAB_GROUP -u SWANLAB_MODE \
      "$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
      --input_file "$INPUT_FILE" \
      --eval_file "$EVAL_FILE" \
      --output_dir "$RUN_DIR" \
      --stage listed \
      "${ELIGIBILITY_ARGS[@]}" \
      --s1_adapter "$S1_ADAPTER" \
      --model_name "$MODEL_NAME" \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --dtype "$TORCH_DTYPE" \
      --lora_r "$TRAIN_LORA_R" \
      --lora_alpha "$TRAIN_LORA_ALPHA" \
      --target_modules down_proj up_proj gate_proj \
      --num_train_epochs 0 \
      --involve_qa_epochs "$INVOLVE_QA_EPOCHS" \
      --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --listed_negative_k "$LISTED_NEGATIVE_K" \
      --s1_gradient_checkpointing \
      --learning_rate "$TRAIN_LEARNING_RATE" \
      --weight_decay "$TRAIN_WEIGHT_DECAY" \
      --max_grad_norm "$TRAIN_MAX_GRAD_NORM" \
      --lambda_candidate "$LAMBDA_CANDIDATE" \
      --stop_after_step "$STOP_AFTER_STEP" \
      --temperature "$CONTRASTIVE_TEMPERATURE" \
      --gen_max_length 32 \
      --seed "$SEED" \
      --dataset "$DATASET" \
      --raw_dir "$RAW_DIR" \
      "${RELATION_TEXT_ARGS[@]}" \
      --save_steps "${SAVE_STEPS:-10}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
      "${NEGATIVE_TRAIN_FLAGS[@]}" \
      "${SWANLAB_FLAGS[@]}" \
      "${TRAIN_FLAGS[@]}" \
      2>&1 | tee "$RUN_DIR/run.log"
    ;;
esac
