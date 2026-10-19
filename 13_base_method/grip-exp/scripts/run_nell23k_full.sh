#!/usr/bin/env bash
# Linux equivalent of run_nell23k_full.ps1.
# Run the paper-aligned Qwen2.5-7B / NELL23K reproduction on one GPU.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        PYTHON_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
    else
        PYTHON_EXECUTABLE="python3"
    fi
else
    PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"
fi
DRY_RUN=0
RESUME_OUTPUTS=0

usage() {
    cat <<'USAGE'
Usage: scripts/run_nell23k_full.sh [options]

Options:
  --resume-outputs         Keep/resume an existing prediction output file.
  --dry-run                Print the command without starting Python.
  --python-executable PATH Python interpreter (default: $PYTHON_EXECUTABLE).
  -h, --help               Show this help message.

The script can safely be rerun after interruption. Cached task-generation
entries are reused by scripts/run_grip.py.
USAGE
}

while (($#)); do
    case "$1" in
        --resume-outputs) RESUME_OUTPUTS=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --python-executable)
            (($# >= 2)) || { echo "error: --python-executable requires a value" >&2; exit 2; }
            PYTHON_EXECUTABLE="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd -- "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

model_dir="$PROJECT_ROOT/model_cache/Qwen--Qwen2.5-7B-Instruct"
data_file="$PROJECT_ROOT/outputs/data/nell23k/processed_test.json"
task_cache="$PROJECT_ROOT/artifacts/task_cache/nell23k_paper_qwen"
log_dir="$PROJECT_ROOT/artifacts/logs"
output_file="$PROJECT_ROOT/outputs/grip_inf/nell23k/nell23k_qwen_paper_3090.json"
mkdir -p "$log_dir" "$task_cache"

if [[ ! -f "$data_file" ]]; then
    echo "Missing processed NELL23K data: $data_file" >&2
    echo "Run: $PYTHON_EXECUTABLE scripts/process_raw_data.py --datasets nell23k --output_dir outputs/data --seed 2026" >&2
    exit 1
fi
if [[ ! -f "$model_dir/config.json" || ! -f "$model_dir/model-00004-of-00004.safetensors" ]]; then
    echo "Missing complete Qwen2.5-7B ModelScope cache: $model_dir" >&2
    echo "Run: $PYTHON_EXECUTABLE scripts/download_model.py --model_name qwen-7b --model_cache_dir model_cache" >&2
    exit 1
fi

if ((RESUME_OUTPUTS)); then
    overwrite=False
else
    overwrite=True
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$log_dir/nell23k_full_$timestamp.log"

args=(
    scripts/mp_wrapper.py
    --script scripts/run_grip.py
    --num_process 1
    --dataset nell23k
    --output_file nell23k_qwen_paper_3090.json
    --do_eval
    --metrics em f1 hit
    --subprocess_args
    --overwrite "$overwrite"
    --model_name qwen-7b
    --model_source local
    --model_cache_dir model_cache
    --local_files_only True
    --task_generator_model_name qwen-7b
    --task_generator_use_vllm False
    --task_generator_batch_size 4
    --task_cache_dir artifacts/task_cache/nell23k_paper_qwen
    --num_context_qa 8000
    --num_reason_qa 2000
    --num_summarization 6000
    --sample_node_attribute_task False
    --repharse_context_qa False
    --context_upsampling False
    --task_gen_max_length 1000
    --tokenize_max_length 4096
    --gen_max_length 1000
    --quantization False
    --lora_r 4
    --lora_alpha 8
    --target_modules down_proj up_proj gate_proj
    --gather_batches False
    --num_train_epochs 1
    --involve_qa_epochs 10
    --s1_stop_loss_threshold 0.15
    --s2_stop_loss_threshold 0.15
    --s1_min_epoch 1
    --s2_min_epoch 1
    --continue_training False
    --remove_unused_columns True
    --report_to none
    --overwrite_output_dir True
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 512
    --learning_rate 1e-3
    --weight_decay 1e-4
    --adam_beta1 0.9
    --adam_beta2 0.98
    --adam_epsilon 1e-8
    --max_grad_norm 1.0
    --log_level info
    --logging_strategy steps
    --logging_steps 1
    --save_strategy no
    --bf16 True
    --tf32 False
    --gradient_checkpointing False
    --lr_scheduler_type linear
    --no_graph_context True
    --use_subgraph False
    --index_format False
)

printf 'Project root: %s\n' "$PROJECT_ROOT"
printf 'Task cache:   %s\n' "$task_cache"
printf 'Log file:     %s\n' "$log_file"
printf 'Output file:  %s\n' "$output_file"
printf 'Output resume mode: %s (overwrite=%s)\n' "$([[ $RESUME_OUTPUTS -eq 1 ]] && echo true || echo false)" "$overwrite"
printf 'Progress display: task-generation stages show [current/total] prompts; Trainer and inference show their own tqdm bars.\n'
printf '\nCommand: %q' "$PYTHON_EXECUTABLE"
printf ' %q' "${args[@]}"
printf '\n'

if ((DRY_RUN)); then
    echo 'Dry run only; no model process was started.'
    exit 0
fi

"$PYTHON_EXECUTABLE" "${args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
if ((status != 0)); then
    echo "NELL23K reproduction exited with code $status. Re-run this script to reuse task-cache entries. Log: $log_file" >&2
    exit "$status"
fi

if [[ ! -f "$output_file" ]]; then
    echo "Run completed without the expected prediction file: $output_file" >&2
    exit 1
fi
printf '\nCompleted successfully. Metrics are near the end of: %s\nPredictions: %s\n' "$log_file" "$output_file"
