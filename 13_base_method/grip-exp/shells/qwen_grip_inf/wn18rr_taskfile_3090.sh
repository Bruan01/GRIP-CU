#!/usr/bin/env bash
set -euo pipefail

# Full-task WN18RR path for this single-3090 machine.  The task JSON is the
# complete Qwen-7B-generated 8000/2000/6000 artifact; no pilot truncation is
# applied.  WN18RR has one graph, so a dataset-level adapter is equivalent to
# the original per-graph runner while avoiding repeated task construction.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
TASK_FILE="$ROOT/../../04_benchmark/data/wn18rr/tasks/grip_wn18rr_tasks.json"
OUTPUT_DIR="$ROOT/outputs/paper_runs/wn18rr_taskfile_3090"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/train_task_file_adapter.py \
    --task_file "$TASK_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name qwen-7b \
    --model_source local \
    --model_cache_dir /mnt/e/GRIP-CU/model_cache \
    --local_files_only \
    --paper_protocol wn18rr \
    --dtype bfloat16 \
    --tokenize_max_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --gradient_checkpointing \
    --stage2_checkpoint_interval 0
