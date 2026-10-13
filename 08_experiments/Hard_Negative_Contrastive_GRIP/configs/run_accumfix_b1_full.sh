#!/usr/bin/env bash
# Retrain generation-only B1 with the repaired trainer and the same Stage-1 adapter.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
OLD_RUN="${OLD_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/20261013_accumfix_b1_full}"
TMUX_SESSION="${TMUX_SESSION:-accumfix-b1-20261013}"
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
if [[ ! -x "$PYTHON" ]]; then echo "error: missing Python: $PYTHON" >&2; exit 1; fi
if [[ ! -f "$OLD_RUN/s1_adapter/adapter_model.safetensors" ]]; then echo "error: missing Stage-1 adapter" >&2; exit 1; fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
printf '[b1] started %s\n' "$(date -u --iso-8601=seconds)"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
  --input_file "$HNG/grip_nell23k_tasks.json" \
  --eval_file "$HNG/data/nell23k/recurrent_relation_prediction.aligned.json" \
  --output_dir "$RUN_DIR" \
  --stage b1 \
  --s1_adapter "$OLD_RUN/s1_adapter" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --lora_r 4 --lora_alpha 8 \
  --target_modules down_proj up_proj gate_proj \
  --num_train_epochs 1 --involve_qa_epochs 10 \
  --s1_stop_loss_threshold 0.15 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 512 \
  --s1_gradient_checkpointing \
  --learning_rate 1e-3 --weight_decay 1e-4 --max_grad_norm 1.0 \
  --lambda_candidate 0 --temperature 1 --gen_max_length 32 --seed 2026 \
  --listed_negative_source train_graph \
  --raw_dir "$CODE_DIR/data/raw_datasets/nell23k" \
  --save_steps 10 --save_total_limit 2
printf '[b1] train complete %s\n' "$(date -u --iso-8601=seconds)"
SCALE=full DECODE=both NO_RESUME=0 RESUME_LAST=0 \
  EVAL_FILE="$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json" \
  LISTED_ADAPTER="$RUN_DIR/b1/adapter" ADAPTER_RUN="$RUN_DIR" \
  RUN_DIR="$RUN_DIR/full_decode" SKIP_RESULT_ARCHIVE=1 \
  bash "$HNG/configs/run_listed_only_decode.sh"
printf '[b1] full evaluation complete %s\n' "$(date -u --iso-8601=seconds)"
bash "$HNG/configs/archive_and_push_results.sh" "$RUN_DIR"
