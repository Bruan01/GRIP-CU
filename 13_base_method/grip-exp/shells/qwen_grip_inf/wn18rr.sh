#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# Disable the optional FlashInfer sampler because this WSL image has no nvcc;
# vLLM's prebuilt attention path remains available for task generation.  The
# paper release uses batches of four for vLLM task generation; this fits the
# 24 GB RTX 3090 with the local Qwen2.5-7B generator.
"$PYTHON" scripts/mp_wrapper.py \
    --script scripts/run_grip.py \
    --num_process 1 \
    --dataset wn18rr \
    --output_file wn18rr_qwen_paper_3090.json \
    --do_eval \
    --metrics em f1 hit \
    --subprocess_args \
    --overwrite True \
    --task_generator_model_name qwen-7b \
    --task_generator_use_vllm True \
    --task_generator_batch_size 4 \
    --task_cache_dir /mnt/e/GRIP-CU/artifacts/grip_paper_tasks_3090/wn18rr \
    --model_source local \
    --model_cache_dir /mnt/e/GRIP-CU/model_cache \
    --local_files_only True \
    --num_context_qa 8000 \
    --num_reason_qa 2000 \
    --num_summarization 6000 \
    --sample_node_attribute_task False \
    --repharse_context_qa False \
    --context_upsampling False \
    --task_gen_max_length 1000 \
    --tokenize_max_length 4096 \
    --gen_max_length 1000 \
    --model_name qwen-7b \
    --quantization False \
    --lora_r 24 \
    --lora_alpha 48 \
    --target_modules down_proj up_proj gate_proj \
    --gather_batches False \
    --num_train_epochs 1 \
    --involve_qa_epochs 5 \
    --s1_stop_loss_threshold 0.15 \
    --s2_stop_loss_threshold 0.15 \
    --s1_min_epoch 1 \
    --s2_min_epoch 1 \
    --continue_training True \
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
    --index_format False \
    --format_as_instruction True

