#!/usr/bin/env bash
# Paper-aligned NELL23K GRIP reproduction for one 24 GB GPU.
# Effective batch = 1 * 512 = 512, matching Table 13 while avoiding a 4x4096-token peak.
set -euo pipefail

export PYTHONPATH=.
python scripts/mp_wrapper.py \
  --script scripts/run_grip.py \
  --num_process 1 \
  --dataset nell23k \
  --output_file nell23k_qwen_paper_3090.json \
  --do_eval \
  --metrics em f1 hit \
  --subprocess_args \
  --overwrite True \
  --model_name qwen-7b \
  --model_source local \
  --model_cache_dir model_cache \
  --local_files_only True \
  --task_generator_model_name qwen-7b \
  --task_generator_use_vllm False \
  --task_generator_batch_size 4 \
  --task_cache_dir artifacts/task_cache/nell23k_paper_qwen \
  --num_context_qa 8000 \
  --num_reason_qa 2000 \
  --num_summarization 6000 \
  --sample_node_attribute_task False \
  --repharse_context_qa False \
  --context_upsampling False \
  --task_gen_max_length 1000 \
  --tokenize_max_length 4096 \
  --gen_max_length 1000 \
  --quantization False \
  --lora_r 4 \
  --lora_alpha 8 \
  --target_modules down_proj up_proj gate_proj \
  --gather_batches False \
  --num_train_epochs 1 \
  --involve_qa_epochs 10 \
  --s1_stop_loss_threshold 0.15 \
  --s2_stop_loss_threshold 0.15 \
  --s1_min_epoch 1 \
  --s2_min_epoch 1 \
  --continue_training False \
  --remove_unused_columns True \
  --report_to none \
  --overwrite_output_dir True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 512 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --log_level info \
  --logging_strategy steps \
  --logging_steps 1 \
  --save_strategy no \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  --lr_scheduler_type linear \
  --no_graph_context True \
  --use_subgraph False \
  --index_format False
