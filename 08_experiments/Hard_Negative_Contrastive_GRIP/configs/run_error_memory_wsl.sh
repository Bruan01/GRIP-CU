#!/usr/bin/env bash
# GRIP-conditioned error-memory experiment runner for WSL.
#
# This script is intentionally staged. The default STEP=preflight performs
# read-only checks only. GPU work is explicit:
#   STEP=rollout  -> frozen-B1 free-generation errors on train QA
#   STEP=score    -> frozen-B1 candidate scores with train-only filtering
#   STEP=freeze   -> CPU-only error-memory bank construction
#   STEP=wire     -> CPU-only full-task manifest attachment check
#   STEP=train    -> Stage-2 GRIP + InfoNCE training, only with ALLOW_TRAIN=1
#
# Example (after the user has approved the corresponding expensive step):
#   STEP=rollout bash configs/run_error_memory_wsl.sh
#   STEP=score bash configs/run_error_memory_wsl.sh
#   STEP=freeze bash configs/run_error_memory_wsl.sh
#   STEP=wire bash configs/run_error_memory_wsl.sh
#   ALLOW_TRAIN=1 STEP=train SWANLAB_PROJECT=grip-negative-sampling \
#     bash configs/run_error_memory_wsl.sh
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
BASELINE_DIR="${GRIP_BASELINE_DIR:-$REPO_ROOT/13_base_method/grip-exp}"
CODE_DIR="${CODE_DIR:-$BASELINE_DIR}"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"

STEP="${STEP:-preflight}"
SEED="${SEED:-2026}"
MODEL_NAME="${MODEL_NAME:-qwen-7b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$CODE_DIR/model_cache}"
INPUT_FILE="${INPUT_FILE:-$HNG/data/grip_nell23k_tasks.json}"
EVAL_FILE="${EVAL_FILE:-$CODE_DIR/outputs/data/nell23k/processed_test.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
S1_ADAPTER="${S1_ADAPTER:-$CODE_DIR/artifacts/nell23k_paper_7b/lora_adapter}"
B1_ADAPTER="${B1_ADAPTER:-$CODE_DIR/artifacts/nell23k_paper_7b/lora_adapter}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HNG/results/runs/error_memory_artifacts}"
SCORES_DIR="${SCORES_DIR:-$ARTIFACT_ROOT/b1_train_scores}"
ROLLOUT_MANIFEST="${ROLLOUT_MANIFEST:-$ARTIFACT_ROOT/b1_train_rollout.jsonl}"
SAMPLER_DIR="${SAMPLER_DIR:-$ARTIFACT_ROOT/error_memory_bank}"
MANIFEST="${MANIFEST:-$SAMPLER_DIR/error_memory_k.jsonl}"
EXPECTED_LISTED="${EXPECTED_LISTED:-3253}"
RUN_ID="${RUN_ID:-error_memory_k9_full_$(date +%Y%m%d)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/$RUN_ID}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$RUN_ID}"
SWANLAB_GROUP="${SWANLAB_GROUP:-error-memory-k9}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"

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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    echo "s1_adapter=$S1_ADAPTER"
    echo "b1_adapter=$B1_ADAPTER"
    echo "scores_dir=$SCORES_DIR"
    echo "rollout_manifest=$ROLLOUT_MANIFEST"
    echo "sampler_dir=$SAMPLER_DIR"
    echo "seed=$SEED"
    echo "swanlab_project=$SWANLAB_PROJECT"
    echo "swanlab_group=$SWANLAB_GROUP"
    echo "swanlab_mode=$SWANLAB_MODE"
    date --iso-8601=seconds
    uname -a
    "$PYTHON" -c "import sys,torch; print('python', sys.version); print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
    for path in "$INPUT_FILE" "$EVAL_FILE" "$RAW_DIR/train.txt" "$S1_ADAPTER/adapter_model.safetensors" "$B1_ADAPTER/adapter_model.safetensors" "$SCORES_DIR/candidate_scores.jsonl" "$SCORES_DIR/metadata.json" "$ROLLOUT_MANIFEST" "$MANIFEST"; do
      if [[ -f "$path" ]]; then
        sha256sum "$path"
      fi
    done
  } > "$RUN_DIR/environment.txt"
}

check_common_assets() {
  require_file "$INPUT_FILE"
  require_file "$RAW_DIR/train.txt"
  require_dir "$HNG/src"
  require_dir "$BASELINE_DIR"
}

check_train_assets() {
  check_common_assets
  require_file "$EVAL_FILE"
  require_file "$S1_ADAPTER/adapter_config.json"
  require_file "$S1_ADAPTER/adapter_model.safetensors"
  require_file "$MANIFEST"
  check_cuda_and_model
}

case "$STEP" in
  preflight)
    check_common_assets
    echo "[preflight] read-only algorithm preflight passed"
    echo "[preflight] code=$HNG"
    echo "[preflight] next explicit steps: rollout -> score -> freeze -> wire -> train"
    echo "[preflight] no GPU work was started"
    ;;
  rollout)
    check_common_assets
    require_file "$B1_ADAPTER/adapter_config.json"
    require_file "$B1_ADAPTER/adapter_model.safetensors"
    check_cuda_and_model
    mkdir -p "$(dirname "$ROLLOUT_MANIFEST")"
    cd "$CODE_DIR"
    "$PYTHON" "$HNG/scripts/mine_rollout_hard_negatives.py" \
      --input_file "$INPUT_FILE" \
      --b1_adapter "$B1_ADAPTER" \
      --raw_dir "$RAW_DIR" \
      --output "$ROLLOUT_MANIFEST" \
      --model_name "$MODEL_NAME" \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --total_k 9 \
      --max_new_tokens "${MAX_NEW_TOKENS:-32}" \
      --seed "$SEED" \
      --filter_splits train \
      --progress_every "${PROGRESS_EVERY:-25}" \
      --resume
    ;;
  score)
    check_common_assets
    require_file "$B1_ADAPTER/adapter_config.json"
    require_file "$B1_ADAPTER/adapter_model.safetensors"
    check_cuda_and_model
    mkdir -p "$SCORES_DIR"
    cd "$CODE_DIR"
    "$PYTHON" "$HNG/scripts/score_relation_candidates_offline.py" \
      --task_file "$INPUT_FILE" \
      --raw_dir "$RAW_DIR" \
      --b1_adapter "$B1_ADAPTER" \
      --output_dir "$SCORES_DIR" \
      --model_name "$MODEL_NAME" \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --split train \
      --filter_splits train \
      --candidate_batch_size "${CANDIDATE_BATCH_SIZE:-8}" \
      --temperature "${SCORE_TEMPERATURE:-1.0}" \
      --seed "$SEED" \
      --progress_every "${PROGRESS_EVERY:-10}" \
      --resume
    ;;
  freeze)
    check_common_assets
    require_file "$SCORES_DIR/candidate_scores.jsonl"
    require_file "$SCORES_DIR/metadata.json"
    require_file "$ROLLOUT_MANIFEST"
    mkdir -p "$SAMPLER_DIR"
    FREEZE_FLAGS=()
    if [[ "${OVERWRITE_BANK:-0}" == "1" ]]; then
      FREEZE_FLAGS+=(--overwrite)
    fi
    "$PYTHON" "$HNG/scripts/mine_error_memory_negatives.py" \
      --scores "$SCORES_DIR/candidate_scores.jsonl" \
      --metadata "$SCORES_DIR/metadata.json" \
      --rollout_manifest "$ROLLOUT_MANIFEST" \
      --output_dir "$SAMPLER_DIR" \
      --raw_dir "$RAW_DIR" \
      --total_k 9 \
      --uniform_mix "${ERROR_MEMORY_UNIFORM_MIX:-0.3333333333333333}" \
      --error_weight "${ERROR_MEMORY_WEIGHT:-1.0}" \
      --violation_weight "${ERROR_MEMORY_VIOLATION_WEIGHT:-1.0}" \
      --sampling_temperature "${ERROR_MEMORY_TEMPERATURE:-0.25}" \
      --seed "$SEED" \
      "${FREEZE_FLAGS[@]}"
    ;;
  wire)
    check_common_assets
    require_file "$MANIFEST"
    "$PYTHON" "$HNG/scripts/verify_error_memory_wiring.py" \
      --task_file "$INPUT_FILE" \
      --manifest "$MANIFEST" \
      --raw_dir "$RAW_DIR" \
      --expected_listed "$EXPECTED_LISTED" \
      --output "${WIRE_OUTPUT:-$SAMPLER_DIR/wiring.json}"
    ;;
  train)
    if [[ "${ALLOW_TRAIN:-0}" != "1" ]]; then
      echo "error: training is gated; set ALLOW_TRAIN=1 only after reviewing the preflight/wiring records" >&2
      exit 2
    fi
    check_train_assets
    if [[ -e "$RUN_DIR" ]]; then
      echo "error: RUN_DIR already exists; choose a new RUN_ID/RUN_DIR instead of overwriting: $RUN_DIR" >&2
      exit 1
    fi
    if [[ "${SKIP_TMUX:-0}" != "1" ]]; then
      TMUX_SESSION="${TMUX_SESSION:-error-memory-k9-$(date +%Y%m%d)}"
      # shellcheck source=tmux_guard.sh
      source "$HNG/configs/tmux_guard.sh"
      tmux_guard_reexec "$0" "$@"
    fi
    write_environment_record
    "$PYTHON" "$HNG/scripts/verify_error_memory_wiring.py" \
      --task_file "$INPUT_FILE" \
      --manifest "$MANIFEST" \
      --raw_dir "$RAW_DIR" \
      --expected_listed "$EXPECTED_LISTED" \
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

    cd "$CODE_DIR"
    "$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
      --input_file "$INPUT_FILE" \
      --eval_file "$EVAL_FILE" \
      --output_dir "$RUN_DIR" \
      --stage listed \
      --s1_adapter "$S1_ADAPTER" \
      --model_name "$MODEL_NAME" \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --lora_r 4 \
      --lora_alpha 8 \
      --target_modules down_proj up_proj gate_proj \
      --num_train_epochs 1 \
      --involve_qa_epochs "${INVOLVE_QA_EPOCHS:-10}" \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-512}" \
      --s1_gradient_checkpointing \
      --learning_rate 1e-3 \
      --weight_decay 1e-4 \
      --max_grad_norm 1.0 \
      --lambda_candidate 1.0 \
      --temperature 1.0 \
      --gen_max_length 32 \
      --seed "$SEED" \
      --listed_negative_source error_memory \
      --negative_sampling_mode error_memory \
      --error_memory_manifest "$MANIFEST" \
      --error_memory_uniform_mix "${ERROR_MEMORY_UNIFORM_MIX:-0.3333333333333333}" \
      --error_memory_weight "${ERROR_MEMORY_WEIGHT:-1.0}" \
      --error_memory_violation_weight "${ERROR_MEMORY_VIOLATION_WEIGHT:-1.0}" \
      --error_memory_temperature "${ERROR_MEMORY_TEMPERATURE:-0.25}" \
      --raw_dir "$RAW_DIR" \
      --save_steps "${SAVE_STEPS:-10}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
      "${SWANLAB_FLAGS[@]}" \
      "${TRAIN_FLAGS[@]}" \
      2>&1 | tee "$RUN_DIR/run.log"
    ;;
esac
