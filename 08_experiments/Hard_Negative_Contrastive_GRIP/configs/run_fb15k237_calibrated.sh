#!/usr/bin/env bash
# FB15K-237 Calibrated (seed 2026), queued behind the H5 seed queue on the one GPU.
#
# Mirrors the NELL Calibrated pipeline:
#   1. score: FB B1 adapter scores every matchable relation QA against all 233
#      train relation texts, train-only false-negative filter (resumable).
#   2. freeze: truncated Calibrated manifest, same knobs as NELL 20261005
#      (k_uniform 6, k_soft 3, soft_pool_k 9, rho 0.33, lambda 1.0/0.25/0.5).
#   3. wire: CPU check that every matchable QA receives its frozen negatives.
#   4. train: listed from the shared FB Stage-1 adapter, seed 2026, 10 QA epochs.
#   5. decode: full FB eval (appends per question), then paired stats.
#
# Waits for $H5_QUEUE_DIR/queue.lock, then holds it, so the two queues never
# share the GPU. Rerun the same command to resume. When the final adapter
# exists, only the newest trainer checkpoint is kept, hardlinked to the adapter.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
STATS_PYTHON="${STATS_PYTHON:-/opt/anaconda3/bin/python}"
TMUX_SESSION="${TMUX_SESSION:-fb-calibrated-20261019}"
RUNS="$HNG/results/runs"
H5_QUEUE_DIR="${H5_QUEUE_DIR:-$RUNS/20261018_h5_seed_queue}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
STEPS="${STEPS:-score freeze wire train decode stats}"

FB_RUN="$RUNS/20261015_fb15k237_qwen7b_b1_randomk"
FB_S1="$FB_RUN/s1_adapter"
FB_B1="$FB_RUN/b1/adapter"
FB_TASK="$FB_RUN/tasks/grip_fb15k237_tasks.json"
FB_RAW="$HNG/data/raw_datasets/fb15k237"
FB_REL="$FB_RAW/rel2text.json"
FB_SMOKE="$HNG/data/fb15k237/recurrent_relation_prediction_smoke.json"
FB_FULL="$HNG/data/fb15k237/recurrent_relation_prediction_full.json"

SCORE_DIR="${SCORE_DIR:-$RUNS/20261019_fb15k237_offline_confusion_train_filter}"
SAMPLER_DIR="${SAMPLER_DIR:-$RUNS/20261019_fb15k237_truncated_calibrated_sampler}"
RUN_DIR="${RUN_DIR:-$RUNS/20261019_fb15k237_calibrated}"
MANIFEST="$SAMPLER_DIR/calibrated.jsonl"
EXPECTED_LISTED="${EXPECTED_LISTED:-2939}"   # matchable FB relation QA (CPU count)

for path in "$PYTHON" "$FB_S1/adapter_model.safetensors" "$FB_B1/adapter_model.safetensors" \
  "$FB_TASK" "$FB_RAW/train.txt" "$FB_RAW/entity2wikidata.json" "$FB_REL" "$FB_SMOKE" "$FB_FULL"; do
  [[ -e "$path" ]] || { echo "error: missing $path" >&2; exit 1; }
done

source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR" "$H5_QUEUE_DIR"
exec 8>"$RUN_DIR/pipeline.lock"
flock -n 8 || { echo "error: pipeline already running for $RUN_DIR" >&2; exit 1; }
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'rc=$?; echo "[fb-cal] exit=$rc $(date -u --iso-8601=seconds); rerun the same command to resume"' EXIT

log() { echo "[fb-cal] $* $(date -u --iso-8601=seconds)"; }

# Do not truncate the H5 lock file; open for append and block until it is free.
exec 9>>"$H5_QUEUE_DIR/queue.lock"
log "waiting for GPU queue lock $H5_QUEUE_DIR/queue.lock"
flock 9
log "acquired GPU queue lock; steps=[$STEPS] session=$TMUX_SESSION"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

check_disk() {
  local free_gb
  free_gb=$(df -BG --output=avail "$RUNS" | tail -1 | tr -dc '0-9')
  (( free_gb >= MIN_FREE_GB )) || { echo "error: only ${free_gb}G free (< ${MIN_FREE_GB}G)" >&2; exit 1; }
  log "disk free ${free_gb}G"
}

adapter_done() { [[ -f "$1/adapter_config.json" && -f "$1/adapter_model.safetensors" ]]; }

keep_last_checkpoint() {  # <trainer_dir> <adapter_dir>; keep newest, hardlink weights
  local trainer="$1" adapter="$2" last ckpt
  [[ -d "$trainer" ]] || return 0
  last=$(find "$trainer" -maxdepth 1 -type d -name 'checkpoint-*' | sort -t- -k2 -n | tail -1)
  [[ -n "$last" ]] || return 0
  for ckpt in "$trainer"/checkpoint-*; do
    [[ "$ckpt" == "$last" ]] || { log "drop older $ckpt"; rm -rf -- "$ckpt"; }
  done
  if cmp -s "$adapter/adapter_model.safetensors" "$last/adapter_model.safetensors"; then
    ln -f "$adapter/adapter_model.safetensors" "$last/adapter_model.safetensors.tmp"
    mv -f "$last/adapter_model.safetensors.tmp" "$last/adapter_model.safetensors"
    log "kept $last (weights hardlinked to final adapter)"
  fi
}

for step in $STEPS; do
  case "$step" in
    score)
      if [[ -f "$SCORE_DIR/run_stats.json" ]]; then
        log "skip score ($SCORE_DIR/run_stats.json exists)"
      else
        check_disk; mkdir -p "$SCORE_DIR"
        log "score FB B1 x 233 relations -> $SCORE_DIR"
        "$PYTHON" "$HNG/scripts/score_relation_candidates_offline.py" \
          --task_file "$FB_TASK" --raw_dir "$FB_RAW" --relation_text_file "$FB_REL" \
          --b1_adapter "$FB_B1" --output_dir "$SCORE_DIR" \
          --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache" \
          --limit 0 --candidate_batch_size 8 --temperature 1.0 \
          --filter_splits train --progress_every 20 --resume
      fi
      ;;
    freeze)
      if [[ -f "$MANIFEST" ]]; then
        log "skip freeze ($MANIFEST exists)"
      else
        log "freeze truncated Calibrated -> $SAMPLER_DIR"
        "$PYTHON" "$HNG/scripts/freeze_shared_pool_samplers.py" \
          --scores "$SCORE_DIR/candidate_scores.jsonl" --metadata "$SCORE_DIR/metadata.json" \
          --output_dir "$SAMPLER_DIR" --k_fixed 9 --k_min 1 --k_max 20 \
          --k_uniform 6 --k_soft 3 --soft_pool_k 9 --rho 0.33 \
          --lambda_0 1.0 --lambda_min 0.25 --lambda_beta 0.5 --seed 2026 --variants calibrated
      fi
      ;;
    wire)
      log "wire Calibrated manifest (expect $EXPECTED_LISTED listed QA)"
      "$PYTHON" "$HNG/scripts/verify_shared_pool_sampler_wiring.py" \
        --task_file "$FB_TASK" --sampler_dir "$SAMPLER_DIR" --raw_dir "$FB_RAW" \
        --relation_text_file "$FB_REL" --expected_listed "$EXPECTED_LISTED" \
        --variants calibrated --output "$RUN_DIR/wiring.json" > "$RUN_DIR/wiring.log"
      ;;
    train)
      if adapter_done "$RUN_DIR/listed/adapter"; then
        log "skip train (adapter exists)"
      else
        check_disk
        log "train Calibrated -> $RUN_DIR"
        "$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
          --input_file "$FB_TASK" --eval_file "$FB_SMOKE" --raw_dir "$FB_RAW" \
          --relation_text_file "$FB_REL" --output_dir "$RUN_DIR" --stage listed \
          --s1_adapter "$FB_S1" --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache" \
          --lora_r 4 --lora_alpha 8 --target_modules down_proj up_proj gate_proj \
          --num_train_epochs 1 --involve_qa_epochs 10 --s1_stop_loss_threshold 0.15 \
          --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --s1_gradient_checkpointing \
          --learning_rate 1e-3 --weight_decay 1e-4 --max_grad_norm 1.0 \
          --lambda_candidate 1.0 --temperature 1.0 --gen_max_length 32 --memory_size 0 --seed 2026 \
          --listed_negative_source score_hard --score_hard_manifest "$MANIFEST" \
          --save_steps 10 --save_total_limit 2
      fi
      adapter_done "$RUN_DIR/listed/adapter" && keep_last_checkpoint "$RUN_DIR/trainer_listed" "$RUN_DIR/listed/adapter"
      ;;
    decode)
      log "decode Calibrated full FB eval"
      "$PYTHON" "$HNG/scripts/eval_listed_decode.py" --adapter_dir "$RUN_DIR/listed/adapter" \
        --eval_file "$FB_FULL" --output_dir "$RUN_DIR/full/listed" --model_name qwen-7b \
        --model_cache_dir "$CODE_DIR/model_cache" --decode both --gen_max_length 32 --progress_every 200
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
