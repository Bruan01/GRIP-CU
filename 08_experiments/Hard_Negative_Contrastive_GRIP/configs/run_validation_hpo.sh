#!/usr/bin/env bash
# NELL23K validation search. Fixed directories make reruns resume the same study.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
ROOT="${ROOT:-$HNG/results/runs/paper_v2_nell23k}"
SEED="${SEED:-2026}"
SEARCH_DIR="${SEARCH_DIR:-$ROOT/hpo_seed$SEED}"
N_TRIALS="${N_TRIALS:-8}"
TMUX_SESSION="${TMUX_SESSION:-nell-validation-hpo}"
TASK_FILE="${TASK_FILE:-$ROOT/tasks/grip_nell23k_paper_tasks.json}"
EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
S1_ADAPTER="${S1_ADAPTER:-$ROOT/s1_adapter}"
CALIBRATED_MANIFEST="${CALIBRATED_MANIFEST:-$ROOT/sampler_seed$SEED/calibrated.jsonl}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
ARGS=(--input_file "$TASK_FILE" --eval_file "$EVAL_FILE" --s1_adapter "$S1_ADAPTER"
      --calibrated_manifest "$CALIBRATED_MANIFEST" --raw_dir "$RAW_DIR"
      --model_cache_dir "$CODE_DIR/model_cache" --search_dir "$SEARCH_DIR"
      --pipeline_root "$ROOT" --seed "$SEED" --n_trials "$N_TRIALS")
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "$PYTHON" "$HNG/scripts/search_validation_hparams.py" "${ARGS[@]}" --dry_run
fi
if [[ -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "Session $TMUX_SESSION already exists; attach: tmux attach -t $TMUX_SESSION"
  exit 0
fi
"$PYTHON" -c 'import optuna' || { echo "Install requirements-hpo.txt into $PYTHON first" >&2; exit 1; }
for path in "$TASK_FILE" "$EVAL_FILE" "$CALIBRATED_MANIFEST" "$RAW_DIR/train.txt" \
            "$S1_ADAPTER/adapter_config.json" "$S1_ADAPTER/adapter_model.safetensors" "$S1_ADAPTER/s1_optimizer.pt"; do
  [[ -f "$path" ]] || { echo "Missing prerequisite: $path; finish S1/score/freeze first" >&2; exit 1; }
done
# Check before detaching so a busy pipeline yields a visible error to the caller.
flock -n "$ROOT/pipeline.lock" true || { echo "Paper pipeline/HPO is active; no GPU task started" >&2; exit 1; }
echo "HPO run.log: $SEARCH_DIR/run.log"
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$SEARCH_DIR"
exec > >(tee -a "$SEARCH_DIR/run.log") 2>&1
echo "[hpo] session=${TMUX:-$TMUX_SESSION} total_budget=$N_TRIALS root=$SEARCH_DIR"
"$PYTHON" "$HNG/scripts/search_validation_hparams.py" "${ARGS[@]}"
