#!/usr/bin/env bash
# GRIP "no context" baseline row: base Qwen2.5-7B-Instruct, no LoRA, same test file
# and prompt as B1. Usage: DATASET=nell23k|fb15k237 bash configs/run_no_adapter_baseline.sh
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
DATASET="${DATASET:-nell23k}"
case "$DATASET" in
  nell23k) EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}" ;;
  fb15k237) EVAL_FILE="${EVAL_FILE:-$HNG/data/fb15k237/recurrent_relation_prediction_full.json}" ;;
  *) echo "error: DATASET must be nell23k or fb15k237" >&2; exit 2 ;;
esac
RUN_DIR="${RUN_DIR:-$HNG/results/runs/paper_no_adapter_$DATASET}"
TMUX_SESSION="${TMUX_SESSION:-paper-no-adapter-$DATASET}"
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" --no_adapter \
  --eval_file "$EVAL_FILE" --output_dir "$RUN_DIR" --splits test \
  --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache" \
  --decode generate --gen_max_length 32 --progress_every 200
