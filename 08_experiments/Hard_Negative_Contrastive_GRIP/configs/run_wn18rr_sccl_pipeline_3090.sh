#!/usr/bin/env bash
# Resumable end-to-end WN18RR paper-alignment pipeline.
#
# This supervisor is intentionally independent of the long baseline process:
# it waits for (or restarts) the baseline, then advances through score/freeze/
# wire/train/evaluate only when each durable artifact is present.
set -Eeuo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
BASELINE_DIR="${GRIP_BASELINE_DIR:-$REPO_ROOT/13_base_method/grip-exp}"
PYTHON="${PYTHON:-/home/kieran/miniconda3/envs/guardenv/bin/python}"
GENERIC_RUNNER="$HNG/configs/run_error_memory_wsl.sh"
BASELINE_SCRIPT="$BASELINE_DIR/shells/qwen_grip_inf/wn18rr_taskfile_3090.sh"
RUN_ROOT="${RUN_ROOT:-$HNG/results/runs/wn18rr_sccl_3090_20261023}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$RUN_ROOT/artifacts}"
TASK_FILE="$REPO_ROOT/04_benchmark/data/wn18rr/tasks/grip_wn18rr_tasks.json"
EVAL_FILE="$REPO_ROOT/04_benchmark/data/wn18rr/processed/processed_test.json"
RAW_DIR="$REPO_ROOT/04_benchmark/data/wn18rr/raw"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/mnt/e/GRIP-CU/model_cache}"
B1_ADAPTER="$BASELINE_DIR/outputs/paper_runs/wn18rr_taskfile_3090/stage2_adapter"
S1_ADAPTER="$BASELINE_DIR/outputs/paper_runs/wn18rr_taskfile_3090/stage1_adapter"
SCORES_DIR="$ARTIFACT_ROOT/b1_train_scores"
SAMPLER_DIR="$ARTIFACT_ROOT/structure_counterfactual_bank"
MANIFEST="$SAMPLER_DIR/counterfactual_bank.jsonl"
COUNTERFACTUAL_METADATA="$SAMPLER_DIR/metadata.json"
ENTITY_TEXT_FILE="$RAW_DIR/entity2text.txt"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1
export PYTHONPATH="$BASELINE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export GRIP_BASELINE_DIR="$BASELINE_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

adapter_ready() {
  [[ -f "$1/adapter_config.json" && -f "$1/adapter_model.safetensors" ]]
}

run_stage() {
  local step="$1"
  local run_dir="$2"
  shift 2
  echo "[pipeline] step=$step start=$(date --iso-8601=seconds)"
  env STEP="$step" DATASET=wn18rr NEGATIVE_SOURCE=structure_counterfactual \
    RUN_ID="wn18rr_sccl_3090_20261023_${step}" \
    RUN_DIR="$run_dir" ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    INPUT_FILE="$TASK_FILE" EVAL_FILE="$EVAL_FILE" RAW_DIR="$RAW_DIR" \
    ENTITY_TEXT_FILE="$ENTITY_TEXT_FILE" S1_ADAPTER="$S1_ADAPTER" \
    B1_ADAPTER="$B1_ADAPTER" MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
    EXPECTED_LISTED=2709 SKIP_TMUX=1 "$@" \
    bash "$GENERIC_RUNNER"
  echo "[pipeline] step=$step finished=$(date --iso-8601=seconds)"
}

wait_for_baseline() {
  while ! adapter_ready "$B1_ADAPTER"; do
    if pgrep -f "scripts/train_task_file_adapter.py.*wn18rr_taskfile_3090" >/dev/null; then
      echo "[pipeline] baseline is running; waiting 30s"
    else
      echo "[pipeline] baseline process is absent; restarting resumable baseline"
      bash "$BASELINE_SCRIPT"
    fi
    sleep 30
  done
  echo "[pipeline] baseline adapter ready: $B1_ADAPTER"
}

wait_for_baseline

if [[ ! -f "$SCORES_DIR/run_stats.json" ]]; then
  run_stage score "$RUN_ROOT/score" \
    CANDIDATE_BATCH_SIZE=16 PROGRESS_EVERY=25
else
  echo "[pipeline] score artifact already complete; reusing $SCORES_DIR/run_stats.json"
fi

if [[ ! -f "$MANIFEST" || ! -f "$COUNTERFACTUAL_METADATA" ]]; then
  run_stage freeze "$RUN_ROOT/freeze"
else
  echo "[pipeline] frozen counterfactual bank already exists; reusing $MANIFEST"
fi

if [[ ! -f "$RUN_ROOT/wire/wiring.json" ]]; then
  run_stage wire "$RUN_ROOT/wire"
else
  echo "[pipeline] wiring gate already passed; reusing $RUN_ROOT/wire/wiring.json"
fi

if ! adapter_ready "$RUN_ROOT/train/listed/adapter"; then
  run_stage train "$RUN_ROOT/train" \
    ALLOW_TRAIN=1 \
    SAVE_STEPS=10 SAVE_TOTAL_LIMIT=2 \
    TRAIN_LORA_R=24 TRAIN_LORA_ALPHA=48 INVOLVE_QA_EPOCHS=5 \
    PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=256
else
  echo "[pipeline] listed adapter already complete; reusing $RUN_ROOT/train/listed/adapter"
fi

for pair in \
  "b1:$B1_ADAPTER:$RUN_ROOT/eval/b1" \
  "listed:$RUN_ROOT/train/listed/adapter:$RUN_ROOT/eval/listed"; do
  IFS=: read -r label adapter output_dir <<<"$pair"
  if [[ ! -f "$output_dir/summary.json" ]]; then
    mkdir -p "$output_dir"
    cd "$BASELINE_DIR"
    "$PYTHON" "$HNG/scripts/evaluate_adapter.py" \
      --adapter "$adapter" \
      --eval_file "$EVAL_FILE" \
      --output_dir "$output_dir" \
      --model_name qwen-7b \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --dtype bfloat16 \
      --gen_max_length 1000 \
      --progress_every 25 \
      --resume \
      --label "$label"
  else
    echo "[pipeline] evaluation already complete: $output_dir/summary.json"
  fi
done

echo "[pipeline] WN18RR pipeline complete=$(date --iso-8601=seconds)"
