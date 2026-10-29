#!/usr/bin/env bash
# Adopt the running anchor, then run a fixed 27-epoch / 60-hour validation plan.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
export ROOT="${ROOT:-$HNG/results/runs/paper_v2_nell23k}"
export SEED="${SEED:-2026}"
export SOURCE_SEARCH_DIR="${SOURCE_SEARCH_DIR:-$ROOT/hpo_seed$SEED}"
export SEARCH_DIR="${SEARCH_DIR:-$ROOT/hpo_budget60_seed$SEED}"
export BUDGET_HOURS="${BUDGET_HOURS:-60}"
export TMUX_SESSION="${TMUX_SESSION:-nell-validation-budget60}"
# Initial adoption requires explicit process IDs and the original launch time.
# Restarts read those same immutable values from search_config.json.
if [[ -f "$SEARCH_DIR/search_config.json" ]]; then
  mapfile -t saved < <("$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["source_supervisor_pid"]); print(c["source_training_pid"]); print(c["started_at"])' "$SEARCH_DIR/search_config.json")
  export SOURCE_SUPERVISOR_PID="${SOURCE_SUPERVISOR_PID:-${saved[0]}}"
  export SOURCE_TRAINING_PID="${SOURCE_TRAINING_PID:-${saved[1]}}"
  export STARTED_AT="${STARTED_AT:-${saved[2]}}"
fi
: "${SOURCE_SUPERVISOR_PID:?Set the running old HPO supervisor PID for initial adoption}"
: "${SOURCE_TRAINING_PID:?Set the running anchor training PID for initial adoption}"
: "${STARTED_AT:?Set the original launch time, e.g. 2026-10-26T12:14:42Z}"
export SOURCE_SUPERVISOR_PID SOURCE_TRAINING_PID STARTED_AT
TASK_FILE="${TASK_FILE:-$ROOT/tasks/grip_nell23k_paper_tasks.json}"
EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
S1_ADAPTER="${S1_ADAPTER:-$ROOT/s1_adapter}"
CALIBRATED_MANIFEST="${CALIBRATED_MANIFEST:-$ROOT/sampler_seed$SEED/calibrated.jsonl}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
ARGS=(--input_file "$TASK_FILE" --eval_file "$EVAL_FILE" --s1_adapter "$S1_ADAPTER"
      --calibrated_manifest "$CALIBRATED_MANIFEST" --raw_dir "$RAW_DIR"
      --model_cache_dir "$CODE_DIR/model_cache" --search_dir "$SEARCH_DIR" --pipeline_root "$ROOT"
      --source_search_dir "$SOURCE_SEARCH_DIR" --source_supervisor_pid "$SOURCE_SUPERVISOR_PID"
      --source_training_pid "$SOURCE_TRAINING_PID" --started_at "$STARTED_AT"
      --budget_hours "$BUDGET_HOURS" --seed "$SEED")
RUNNER="$HNG/scripts/search_validation_budget.py"
if [[ -n "${RECOVERY_HOURS:-}" ]]; then
  RUNNER="$HNG/scripts/resume_validation_budget.py"
  ARGS+=(--recovery_hours "$RECOVERY_HOURS")
fi
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "$PYTHON" "$RUNNER" "${ARGS[@]}" --dry_run
fi
if [[ -z "${TMUX:-}" ]] && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "Session $TMUX_SESSION already exists; attach: tmux attach -t $TMUX_SESSION"
  exit 0
fi
# Source GPU process deliberately holds pipeline.lock until the adopted anchor exits.
# The new runner takes that lock before starting any further GPU process.
"$PYTHON" "$RUNNER" "${ARGS[@]}" --dry_run
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$SEARCH_DIR"
exec > >(tee -a "$SEARCH_DIR/run.log") 2>&1
echo "[budget] session=$TMUX_SESSION root=$SEARCH_DIR budget_hours=$BUDGET_HOURS"
"$PYTHON" "$RUNNER" "${ARGS[@]}"
