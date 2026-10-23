#!/usr/bin/env bash
set -euo pipefail

# This is a hardware/protocol smoke path on the repository's 100-graph test
# split.  It is intentionally labelled local_split: it is not the paper's
# 400-graph CLEGR benchmark and must not be placed in the paper main table.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

"$PYTHON" scripts/mp_wrapper.py \
    --script scripts/run_grip.py \
    --num_process 1 \
    --dataset clegr_reasoning \
    --test_file "$ROOT/../../04_benchmark/data/clegr_reasoning/processed/processed_test.json" \
    --reference_file "$ROOT/../../04_benchmark/data/clegr_reasoning/processed/processed_train.json" \
    --num_test 100 \
    --output_file clegr_reasoning_qwen_paper_protocol_local_split.json \
    --do_eval \
    --metrics em f1 \
    --subprocess_args \
    --overwrite False \
    --task_generator_model_name qwen-7b \
    --task_generator_use_vllm True \
    --task_generator_batch_size 1 \
    --task_cache_dir /mnt/e/GRIP-CU/artifacts/grip_paper_tasks_3090/clegr_reasoning_local_split \
    --model_source local \
    --model_cache_dir /mnt/e/GRIP-CU/model_cache \
    --local_files_only True \
    --num_context_qa 20 \
    --num_reason_qa 160 \
    --num_summarization 20 \
    --sample_node_attribute_task True \
    --repharse_context_qa True \
    --context_upsampling True \
    --task_gen_max_length 1000 \
    --tokenize_max_length 4096 \
    --gen_max_length 1000 \
    --model_name qwen-7b \
    --quantization False \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules down_proj up_proj gate_proj \
    --gather_batches True \
    --num_train_epochs 50 \
    --involve_qa_epochs 50 \
    --s1_stop_loss_threshold 0.4 \
    --s2_stop_loss_threshold 0.4 \
    --s1_min_epoch 5 \
    --s2_min_epoch 5 \
    --continue_training False \
    --remove_unused_columns True \
    --report_to none \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-4 \
    --max_grad_norm 1.0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --lr_scheduler_type linear \
    --no_graph_context True \
    --use_subgraph False \
    --index_format False
