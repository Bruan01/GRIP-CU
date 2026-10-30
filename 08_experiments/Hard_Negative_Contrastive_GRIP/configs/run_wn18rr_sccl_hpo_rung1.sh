#!/usr/bin/env bash
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
TRAIN_RUNNER="$HNG/configs/run_error_memory_wsl.sh"
HPO_ROOT="$HNG/results/hpo/wn18rr_sccl_seed2026"
DATA_ROOT="$REPO_ROOT/04_benchmark/data/wn18rr"
ARTIFACT_ROOT="$HNG/results/runs/wn18rr_sccl_3090_20261023/artifacts"
SAMPLER_DIR="$ARTIFACT_ROOT/structure_counterfactual_bank"
MODEL_CACHE_DIR="/mnt/e/GRIP-CU/model_cache"
BASELINE_EVAL_DIR="$HPO_ROOT/lambda1_valid"
BASELINE_SUMMARY="$BASELINE_EVAL_DIR/summary.json"
BASELINE_EVAL_PID="${BASELINE_EVAL_PID:-316576}"

mkdir -p "$HPO_ROOT"

run_lambda_rung1() {
  local lambda_value="$1"
  local lambda_slug="$2"
  local run_id="wn18rr_sccl_hpo_s2026_lc${lambda_slug}"
  local run_dir="$HNG/results/runs/$run_id"
  local eval_dir="$HPO_ROOT/lambda${lambda_slug}_rung1_valid500"
  local adapter="$run_dir/listed/adapter"
  local no_resume=0
  local skip_train=0

  if [[ -f "$eval_dir/summary.json" ]]; then
    echo "[rung1] lambda=$lambda_value already evaluated; skip"
    return
  fi
  if [[ ! -e "$run_dir" ]]; then
    no_resume=1
  fi
  if [[ -f "$adapter/adapter_config.json" && -f "$adapter/adapter_model.safetensors" ]]; then
    skip_train=1
  fi

  echo "[rung1] lambda=$lambda_value run_dir=$run_dir"
  ALLOW_TRAIN=1 \
    STEP=train \
    DATASET=wn18rr \
    NEGATIVE_SOURCE=structure_counterfactual \
    SEED=2026 \
    LAMBDA_CANDIDATE="$lambda_value" \
    STOP_AFTER_STEP=25 \
    DEFER_EVAL=1 \
    RUN_VARIANT="hpo_rung1_lc${lambda_slug}" \
    RUN_ID="$run_id" \
    RUN_DIR="$run_dir" \
    INPUT_FILE="$DATA_ROOT/tasks/grip_wn18rr_tasks.json" \
    EVAL_FILE="$DATA_ROOT/processed/processed_val.json" \
    RAW_DIR="$DATA_ROOT/raw" \
    MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    SAMPLER_DIR="$SAMPLER_DIR" \
    EXPECTED_LISTED=2709 \
    NO_RESUME="$no_resume" \
    SKIP_TRAIN="$skip_train" \
    SKIP_TMUX=1 \
    SAVE_STEPS=25 \
    SAVE_TOTAL_LIMIT=2 \
    INVOLVE_QA_EPOCHS=5 \
    TRAIN_LEARNING_RATE=1e-3 \
    CONTRASTIVE_TEMPERATURE=1.0 \
    TORCH_DTYPE=bfloat16 \
    bash "$TRAIN_RUNNER"

  mkdir -p "$eval_dir"
  "$PYTHON" "$HNG/scripts/evaluate_adapter.py" \
    --adapter "$adapter" \
    --eval_file "$HPO_ROOT/valid_screen_500.json" \
    --output_dir "$eval_dir" \
    --model_name qwen-7b \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --dtype bfloat16 \
    --gen_max_length 1000 \
    --progress_every 100 \
    --resume \
    --label "sccl_lambda${lambda_slug}_rung1_seed2026" \
    > "$eval_dir/evaluation.log" 2>&1
  echo "[rung1] lambda=$lambda_value complete"
}

# Finish the incumbent's full-valid score before launching any GPU training.
# If the attached evaluation was interrupted, resume its saved predictions.
if [[ ! -f "$BASELINE_SUMMARY" ]]; then
  while kill -0 "$BASELINE_EVAL_PID" 2>/dev/null; do
    sleep 15
  done
  if [[ ! -f "$BASELINE_SUMMARY" ]]; then
    "$PYTHON" "$HNG/scripts/evaluate_adapter.py" \
      --adapter "$HNG/results/runs/wn18rr_sccl_3090_20261023/train/listed/adapter" \
      --eval_file "$DATA_ROOT/processed/processed_val.json" \
      --output_dir "$BASELINE_EVAL_DIR" \
      --model_name qwen-7b \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --dtype bfloat16 \
      --gen_max_length 1000 \
      --progress_every 50 \
      --resume \
      --label sccl_lambda1_seed2026 \
      > "$HPO_ROOT/lambda1_valid_resume.log" 2>&1
  fi
fi

# The incumbent lambda=1.0 is already a full-budget run. Probe both sides of
# it on the first resource rung; later rungs promote only candidates supported
# by the paired validation results.
run_lambda_rung1 0.25 0p25
run_lambda_rung1 0.5 0p5
run_lambda_rung1 2.0 2p0

echo "[rung1] all candidate panels complete; inspect summaries before promotion"
