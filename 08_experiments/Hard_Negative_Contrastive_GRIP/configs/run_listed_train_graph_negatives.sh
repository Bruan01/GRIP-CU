#!/usr/bin/env bash
# Homologous listed negatives: InfoNCE pool = official 198 train-graph relations,
# sampled with the same process.py permutation rule as eval 10-way.
#
# Does not change the loss. Reuses the 20260913 Stage-1 adapter and leaves B1
# untouched (lambda=0 never used the old 370-vocab negatives). Retrains listed
# only, then smoke-evaluates that adapter.
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
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}_qwen7b_train_graph_negatives}"

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
if [[ -e "$RUN_DIR" ]]; then
  echo "error: run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "old_run=$OLD_RUN"
  echo "s1_adapter=$S1_ADAPTER"
  echo "listed_negative_source=train_graph"
  echo "input_file=$INPUT_FILE"
  echo "eval_file=$EVAL_FILE"
  echo "raw_dir=$RAW_DIR"
  echo "note=retrain listed only; B1 is the 20260913 adapter"
  date --iso-8601=seconds
  "$PYTHON" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
} >> "$RUN_DIR/environment.txt" 2>&1

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

echo "[launch] listed train_graph negatives from $S1_ADAPTER" | tee -a "$RUN_DIR/run.log"
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
  --listed_negative_source train_graph \
  --raw_dir "$RAW_DIR" \
  2>&1 | tee -a "$RUN_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] listed finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/run.log"
echo "$RUN_DIR" > "$HNG/results/LAST_TRAIN_GRAPH_NEGATIVES_RUN.txt"
exit "$rc"
