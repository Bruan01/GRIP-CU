#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT/13_base_method/grip-exp${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export GRIP_VLLM_GPU_MEMORY_UTILIZATION=0.76

"$PYTHON" 04_benchmark/preprocessing/generate_grip_stage_tasks.py \
    --dataset codexm \
    --input-split train \
    --reference-split train \
    --input-file 04_benchmark/data/codexm/processed/processed_train_graph.json \
    --reference-file 04_benchmark/data/codexm/processed/processed_train_reference.json \
    --output-file 04_benchmark/data/codexm/tasks/grip_codexm_tasks.json \
    --task-cache-dir 04_benchmark/data/codexm/tasks/cache_vllm_qwen25_7b_b8_gpu76_20261026 \
    --seed 2026 \
    --model-name qwen-7b \
    --task-generator-model-name qwen-7b \
    --model-source local \
    --model-cache-dir /mnt/e/GRIP-CU/model_cache \
    --local-files-only \
    --task-generator-use-vllm \
    --task-generator-batch-size 8 \
    --task-generator-tokenize-max-length 10000 \
    --tokenize-max-length 4096 \
    --task-gen-max-length 1000 \
    --num-context-qa 8000 \
    --num-reason-qa 2000 \
    --num-summarization 6000 \
    --involve-qa-epochs 5 \
    --no-sample-node-attribute-task \
    --no-repharse-context-qa \
    --no-context-upsampling \
    --format-as-instruction
