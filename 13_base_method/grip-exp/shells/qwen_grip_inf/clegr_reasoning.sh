#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# The packaged benchmark has a 300/100/100 development split.  The original
# GRIP paper evaluates CLEGR on the remaining 400 graphs after a 20% train
# reference split, so materialize that split before launching the per-graph
# runner.  The split manifest is kept beside the files for provenance.
CLEGR_DATA_DIR="$ROOT/../../04_benchmark/data/clegr_reasoning"
CLEGR_PAPER_SPLIT_DIR="$CLEGR_DATA_DIR/paper_split"
if [[ ! -f "$CLEGR_PAPER_SPLIT_DIR/processed_train.json" || \
      ! -f "$CLEGR_PAPER_SPLIT_DIR/processed_test.json" || \
      ! -f "$CLEGR_PAPER_SPLIT_DIR/manifest.json" ]]; then
    "$PYTHON" scripts/build_clegr_paper_split.py \
        --data_dir "$CLEGR_DATA_DIR/processed" \
        --output_dir "$CLEGR_PAPER_SPLIT_DIR" \
        --seed 2026
fi

# The paper release uses vLLM batches of four.  This fits the 24 GB RTX 3090
# with the Qwen2.5-7B task generator and is faster than serial generation
# while preserving the generated-task protocol.  CLEGR's paper metric is
# evaluated afterward by scripts/judge_clegr_siliconflow.py, not by the local
# EM/F1 diagnostics.
"$PYTHON" scripts/mp_wrapper.py \
    --script scripts/run_grip.py \
    --num_process 1 \
    --dataset clegr_reasoning \
    --test_file "$CLEGR_PAPER_SPLIT_DIR/processed_test.json" \
    --reference_file "$CLEGR_PAPER_SPLIT_DIR/processed_train.json" \
    --output_file clegr_reasoning_qwen_paper_3090.json \
    --num_test 400 \
    --subprocess_args \
    --overwrite False \
    --task_generator_model_name qwen-7b \
    --task_generator_use_vllm True \
    --task_generator_batch_size 4 \
    --task_cache_dir /mnt/e/GRIP-CU/artifacts/grip_paper_tasks_3090/clegr_reasoning \
    --model_source local \
    --model_cache_dir /mnt/e/GRIP-CU/model_cache \
    --local_files_only True \
    --num_context_qa 20 \
    --num_reason_qa 160 \
    --num_summarization 20 \
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

