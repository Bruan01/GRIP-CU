#!/usr/bin/env bash
# H5 compute-matched B1 + one extra training seed per dataset, then paired stats.
#
# Queue (one GPU, sequential; rerun the same command to resume):
#   1. NELL23K B1 with 37 QA epochs (compute-matched to Random-K), full decode.
#      Random-K processes 3.64x B1's Stage-2 tokens on NELL (CPU count of
#      generation + 10-way candidate rows) and ran 3.74x B1's wall clock;
#      37 epochs >= both, so the control errs in B1's favour.
#   2. NELL23K seed 2027: B1 + Random-K (fresh seed-2027 Random-K manifest), full decode.
#   3. FB15K-237 seed 2027: B1 + Random-K (negatives redrawn with seed 2027), full decode.
#   4. CPU: scripts/paired_significance.py over every finished pair.
#
# Seeds vary Stage-2 data order and negative draws only. Stage 1 is the shared
# adapter of each dataset (20260913 for NELL, 20261015 for FB) and is not retrained.
#
# Resume: trainer checkpoints every 10 steps (keep 2) under each train dir,
# decode appends per question. Disk is tight (~21 GB free on 2026-10-18), so a
# variant's trainer_* checkpoints are deleted once its final adapter exists.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
STATS_PYTHON="${STATS_PYTHON:-/opt/anaconda3/bin/python}"   # has matplotlib for the figure
TMUX_SESSION="${TMUX_SESSION:-h5-seeds-20261018}"
QUEUE_DIR="${QUEUE_DIR:-$HNG/results/runs/20261018_h5_seed_queue}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
CM_EPOCHS="${CM_EPOCHS:-37}"
STEPS="${STEPS:-cm nell_seed fb_seed stats}"

RUNS="$HNG/results/runs"
NELL_S1="$RUNS/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/s1_adapter"
NELL_TASK="$HNG/grip_nell23k_tasks.json"
NELL_RAW="$CODE_DIR/data/raw_datasets/nell23k"
NELL_SMOKE="$HNG/data/nell23k/recurrent_relation_prediction.aligned.json"
NELL_FULL="$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json"
NELL_RK_2027="$RUNS/20261018_shared_pool_samplers_seed2027/random_k.jsonl"
FB_RUN="$RUNS/20261015_fb15k237_qwen7b_b1_randomk"
FB_S1="$FB_RUN/s1_adapter"
FB_TASK="$FB_RUN/tasks/grip_fb15k237_tasks.json"
FB_RAW="$HNG/data/raw_datasets/fb15k237"
FB_SMOKE="$HNG/data/fb15k237/recurrent_relation_prediction_smoke.json"
FB_FULL="$HNG/data/fb15k237/recurrent_relation_prediction_full.json"

CM_RUN="$RUNS/20261018_nell_b1_compute_matched_${CM_EPOCHS}ep"
NELL_SEED_RUN="$RUNS/20261018_nell_seed2027"
FB_SEED_RUN="$RUNS/20261018_fb15k237_seed2027"

for path in "$PYTHON" "$NELL_S1/adapter_model.safetensors" "$NELL_TASK" "$NELL_RAW/train.txt" \
  "$NELL_SMOKE" "$NELL_FULL" "$NELL_RK_2027" "$FB_S1/adapter_model.safetensors" "$FB_TASK" \
  "$FB_RAW/rel2text.json" "$FB_SMOKE" "$FB_FULL"; do
  [[ -e "$path" ]] || { echo "error: missing $path" >&2; exit 1; }
done

source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$QUEUE_DIR"
exec 9>"$QUEUE_DIR/queue.lock"
flock -n 9 || { echo "error: queue already running for $QUEUE_DIR" >&2; exit 1; }
exec > >(tee -a "$QUEUE_DIR/run.log") 2>&1
trap 'rc=$?; echo "[queue] exit=$rc $(date -u --iso-8601=seconds); rerun the same command to resume"' EXIT

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() { echo "[queue] $* $(date -u --iso-8601=seconds)"; }

check_disk() {
  local free_gb
  free_gb=$(df -BG --output=avail "$RUNS" | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    echo "error: only ${free_gb}G free under $RUNS (< ${MIN_FREE_GB}G); free space and rerun" >&2
    exit 1
  fi
  log "disk free ${free_gb}G"
}

adapter_done() { [[ -f "$1/adapter_config.json" && -f "$1/adapter_model.safetensors" ]]; }

prune_checkpoints() {  # <train_dir> <variant>; only after the final adapter exists
  local dir="$1" variant="$2"
  if adapter_done "$dir/$variant/adapter" && [[ -d "$dir/trainer_$variant" ]]; then
    log "prune $dir/trainer_$variant (final adapter saved)"
    rm -rf -- "$dir/trainer_$variant"
  fi
}

common_args=(
  --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache"
  --lora_r 4 --lora_alpha 8 --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1 --s1_stop_loss_threshold 0.15
  --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --s1_gradient_checkpointing
  --learning_rate 1e-3 --weight_decay 1e-4 --max_grad_norm 1.0
  --temperature 1.0 --gen_max_length 32 --memory_size 0
  --save_steps 10 --save_total_limit 2
)

train() {  # <train_dir> <variant b1|listed> <extra args...>
  local dir="$1" variant="$2"; shift 2
  if adapter_done "$dir/$variant/adapter"; then
    log "skip train $dir/$variant (adapter exists)"
  else
    check_disk
    log "train $variant -> $dir"
    "$PYTHON" "$HNG/scripts/train_listed_contrastive.py" --output_dir "$dir" --stage "$variant" \
      "${common_args[@]}" "$@"
  fi
  prune_checkpoints "$dir" "$variant"
}

decode() {  # <adapter_dir> <eval_file> <output_dir>
  log "decode $1 -> $3"
  "$PYTHON" "$HNG/scripts/eval_listed_decode.py" --adapter_dir "$1" --eval_file "$2" \
    --output_dir "$3" --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache" \
    --decode both --gen_max_length 32 --progress_every 200
}

nell_args=(--input_file "$NELL_TASK" --eval_file "$NELL_SMOKE" --raw_dir "$NELL_RAW")
fb_args=(--input_file "$FB_TASK" --eval_file "$FB_SMOKE" --raw_dir "$FB_RAW"
         --relation_text_file "$FB_RAW/rel2text.json")

log "started steps=[$STEPS] session=$TMUX_SESSION"
for step in $STEPS; do
  case "$step" in
    cm)
      train "$CM_RUN" b1 "${nell_args[@]}" --s1_adapter "$NELL_S1" --seed 2026 \
        --involve_qa_epochs "$CM_EPOCHS" --lambda_candidate 0 --listed_negative_source train_graph
      decode "$CM_RUN/b1/adapter" "$NELL_FULL" "$CM_RUN/full_decode/b1"
      ;;
    nell_seed)
      train "$NELL_SEED_RUN/train_b1" b1 "${nell_args[@]}" --s1_adapter "$NELL_S1" --seed 2027 \
        --involve_qa_epochs 10 --lambda_candidate 0 --listed_negative_source train_graph
      train "$NELL_SEED_RUN/train_listed" listed "${nell_args[@]}" --s1_adapter "$NELL_S1" \
        --seed 2027 --involve_qa_epochs 10 --lambda_candidate 1.0 \
        --listed_negative_source score_hard --score_hard_manifest "$NELL_RK_2027"
      decode "$NELL_SEED_RUN/train_b1/b1/adapter" "$NELL_FULL" "$NELL_SEED_RUN/full/b1"
      decode "$NELL_SEED_RUN/train_listed/listed/adapter" "$NELL_FULL" "$NELL_SEED_RUN/full/listed"
      ;;
    fb_seed)
      train "$FB_SEED_RUN/train_b1" b1 "${fb_args[@]}" --s1_adapter "$FB_S1" --seed 2027 \
        --involve_qa_epochs 10 --lambda_candidate 0 --listed_negative_source train_graph
      train "$FB_SEED_RUN/train_listed" listed "${fb_args[@]}" --s1_adapter "$FB_S1" --seed 2027 \
        --involve_qa_epochs 10 --lambda_candidate 1.0 --listed_negative_source train_graph \
        --listed_negative_k 9 --listed_negative_seed 2027
      decode "$FB_SEED_RUN/train_b1/b1/adapter" "$FB_FULL" "$FB_SEED_RUN/full/b1"
      decode "$FB_SEED_RUN/train_listed/listed/adapter" "$FB_FULL" "$FB_SEED_RUN/full/listed"
      ;;
    stats)
      log "paired stats"
      "$STATS_PYTHON" "$HNG/scripts/paired_significance.py"
      ;;
    *) echo "error: unknown step $step" >&2; exit 1 ;;
  esac
  log "step $step complete"
done
log "all steps complete; see $HNG/results/paired_stats/paired_stats.md"
