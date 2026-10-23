#!/usr/bin/env bash
# Strict-paper pipeline (GRIP KDD'26 Sec. 4 + Table 13): B1 -> Calibrated / Soft-Mix.
# Not run yet: review before launching. One GPU; rerun the same command to resume.
#
#   DATASET=nell23k|fb15k237  SEEDS="2026 2027"  STEPS="tasks s1 b1 score freeze wire train"
#
# Layout under $ROOT (default results/runs/paper_$DATASET):
#   tasks/          paper task file (Ns/Nr/Nc = 6000/2000/8000), generated once (seed 2026)
#   s1_adapter/     S1 (plain-text context + summary), trained once (seed $S1_SEED)
#   b1_seed$S/      S2 B1 per seed (answer-only L_QA, early stop 0.15, min 1 epoch)
#   confusion/      offline confusion DB from b1_seed$SCORE_SEED, train-only false-negative filter
#   sampler_seed$S/ calibrated.jsonl (frozen with --seed $S) + soft_mix.jsonl (fixed lambda 1)
#   calibrated_seed$S/, soft_mix_seed$S/   S2 with InfoNCE over the frozen negatives
#
# Every S2 arm uses the same Table 13 settings and stops on generation_loss (L_QA
# window mean), so InfoNCE never changes the stop epoch. Headline = test.official_em
# in <arm>/{b1,listed}/summary.json. Random-K is intentionally not part of this run.
# Optional step "decode": extra closed-set (10-way) decode on test; not a paper metric.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
DATASET="${DATASET:?set DATASET=nell23k or fb15k237}"
SEEDS="${SEEDS:-2026 2027}"
S1_SEED="${S1_SEED:-2026}"
SCORE_SEED="${SCORE_SEED:-2026}"
STEPS="${STEPS:-tasks s1 b1 score freeze wire train}"
ROOT="${ROOT:-$HNG/results/runs/paper_$DATASET}"
TMUX_SESSION="${TMUX_SESSION:-paper-$DATASET}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

case "$DATASET" in
  nell23k)
    RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
    # All NELL23K test triples in the raw file (4,944; paper reports 4,952, see docs).
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
    TRAIN_REF="$ROOT/data/processed_train.json"
    LORA=(--lora_r 4 --lora_alpha 8)
    S2_EPOCHS=10
    REL_TEXT=()
    ;;
  fb15k237)
    DATA_DIR="${DATA_DIR:-$HNG/data/fb15k237}"
    RAW_DIR="${RAW_DIR:-$HNG/data/raw_datasets/fb15k237}"
    # The 10,000 random FB15K-237 test questions.
    EVAL_FILE="${EVAL_FILE:-$DATA_DIR/recurrent_relation_prediction_full.json}"
    TRAIN_REF="$DATA_DIR/processed_train.json"
    LORA=(--lora_r 24 --lora_alpha 48)
    S2_EPOCHS=5
    REL_TEXT=(--relation_text_file "$RAW_DIR/rel2text.json")
    ;;
  *) echo "error: unknown DATASET=$DATASET" >&2; exit 1 ;;
esac
TASK_FILE="$ROOT/tasks/grip_${DATASET}_paper_tasks.json"
S1_ADAPTER="$ROOT/s1_adapter"
SCORE_DIR="$ROOT/confusion"

source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
[[ -x "$PYTHON" ]] || { echo "error: missing Python: $PYTHON" >&2; exit 1; }
mkdir -p "$ROOT"
exec 9>"$ROOT/pipeline.lock"
flock -n 9 || { echo "error: pipeline already running: $ROOT" >&2; exit 1; }
exec > >(tee -a "$ROOT/run.log") 2>&1
trap 'rc=$?; echo "[paper-$DATASET] exit=$rc $(date --iso-8601=seconds); rerun the same command to resume"' EXIT
log() { echo "[paper-$DATASET] $* $(date --iso-8601=seconds)"; }

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

check_disk() {
  local free_gb
  free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  (( free_gb >= MIN_FREE_GB )) || { echo "error: only ${free_gb}G free (< ${MIN_FREE_GB}G)" >&2; exit 1; }
}
adapter_done() { [[ -f "$1/adapter_config.json" && -f "$1/adapter_model.safetensors" ]]; }

# Table 13 (Qwen2.5-7B) settings shared by S1 and every S2 arm. Weight decay is not
# in the paper (project choice: 0); Table 13's 1e-4 is the Adam epsilon.
COMMON=(
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py"
  --input_file "$TASK_FILE" --eval_file "$EVAL_FILE" --eval_splits test
  --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache"
  --raw_dir "$RAW_DIR" "${REL_TEXT[@]}"
  "${LORA[@]}" --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1 --s1_stop_loss_threshold 0.15
  --involve_qa_epochs "$S2_EPOCHS" --s2_stop_loss_threshold 0.15 --s2_min_epoch 1
  --s2_stop_loss_key generation_loss --qa_loss_mask answer_only
  --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --s1_gradient_checkpointing
  --learning_rate 1e-3 --weight_decay 0 --adam_epsilon 1e-4 --max_grad_norm 1.0
  --gen_max_length 32 --save_steps 10 --save_total_limit 2
)

log "started steps=[$STEPS] seeds=[$SEEDS] root=$ROOT"
for step in $STEPS; do
  case "$step" in
    tasks)
      mkdir -p "$ROOT/tasks"
      if [[ "$DATASET" == nell23k ]]; then
        # Official process.py writes no train QA file; build the train-only graph + k-shot reference.
        "$PYTHON" "$HNG/scripts/build_nell23k_train_ref.py" --raw_dir "$RAW_DIR" --output_file "$TRAIN_REF" --seed 2026
      else
        "$PYTHON" "$HNG/scripts/prepare_fb15k237.py" --raw_dir "$RAW_DIR" --out_dir "$DATA_DIR"
      fi
      # Reuses an existing task file. Node+edge context QA, rooted 3-hop x 3 subgraphs,
      # random reasoning type, plain-text S1 samples, every train triple recited.
      "$PYTHON" "$CODE_DIR/scripts/generate_nell23k_tasks.py" \
        --input-file "$TRAIN_REF" --ref-file "$TRAIN_REF" --require-train-ref \
        --task-file "$TASK_FILE" --task-cache-dir "$ROOT/tasks/cache" \
        --seed 2026 --model-name qwen-7b --model-source local \
        --model-cache-dir "$CODE_DIR/model_cache" --local-files-only \
        --task-generator-model-name qwen-7b --no-task-generator-use-vllm \
        --task-generator-batch-size 1 --task-gen-max-length 1000 \
        --num-context-qa 8000 --num-reason-qa 2000 --num-summarization 6000 \
        --sample-node-attribute-task --no-repharse-context-qa --no-context-upsampling \
        --no-format-as-instruction --uniform-sampling --rooted-reasoning-subgraph \
        --random-reasoning-type --keep-parallel-edges --train-entity-nodes-only \
        --involve-qa-epochs "$S2_EPOCHS"
      ;;
    s1)
      check_disk
      "${COMMON[@]}" --stage s1 --output_dir "$ROOT" --seed "$S1_SEED" \
        --lambda_candidate 0 --listed_negative_source train_graph
      ;;
    b1)
      for seed in $SEEDS; do
        check_disk
        log "B1 seed=$seed"
        "${COMMON[@]}" --stage b1 --output_dir "$ROOT/b1_seed$seed" --s1_adapter "$S1_ADAPTER" \
          --seed "$seed" --lambda_candidate 0 --listed_negative_source train_graph
      done
      ;;
    score)
      # One confusion DB per dataset (~6 h): seed-$SCORE_SEED B1 scores every matchable
      # relation QA against all train relations; false negatives filtered on train only.
      if [[ -f "$SCORE_DIR/run_stats.json" ]]; then
        log "skip score ($SCORE_DIR/run_stats.json exists)"
      else
        check_disk; mkdir -p "$SCORE_DIR"
        "$PYTHON" "$HNG/scripts/score_relation_candidates_offline.py" \
          --task_file "$TASK_FILE" --raw_dir "$RAW_DIR" "${REL_TEXT[@]}" \
          --b1_adapter "$ROOT/b1_seed$SCORE_SEED/b1/adapter" --output_dir "$SCORE_DIR" \
          --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache" \
          --limit 0 --candidate_batch_size 8 --temperature 1.0 \
          --filter_splits train --progress_every 20 --resume
      fi
      ;;
    freeze)
      for seed in $SEEDS; do
        sampler="$ROOT/sampler_seed$seed"
        if [[ -f "$sampler/calibrated.jsonl" ]]; then
          log "skip freeze calibrated seed=$seed (exists)"
        else
          "$PYTHON" "$HNG/scripts/freeze_shared_pool_samplers.py" \
            --scores "$SCORE_DIR/candidate_scores.jsonl" --metadata "$SCORE_DIR/metadata.json" \
            --output_dir "$sampler" --k_fixed 9 --k_min 1 --k_max 20 \
            --k_uniform 6 --k_soft 3 --soft_pool_k 9 --rho 0.33 \
            --lambda_0 1.0 --lambda_min 0.25 --lambda_beta 0.5 --seed "$seed" --variants calibrated
        fi
        # Soft-Mix = the same frozen negatives with a fixed lambda of 1 (isolates the lambda schedule).
        if [[ -f "$sampler/soft_mix.jsonl" ]]; then
          log "skip freeze soft_mix seed=$seed (exists)"
        else
          "$PYTHON" "$HNG/scripts/freeze_fixed_lambda_manifest.py" --source "$sampler/calibrated.jsonl" \
            --output "$sampler/soft_mix.jsonl" --lambda_candidate 1 --variant soft_mix
        fi
      done
      ;;
    wire)
      expected=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["qa_count"])' "$SCORE_DIR/run_stats.json")
      for seed in $SEEDS; do
        log "wire seed=$seed (expect $expected listed QA)"
        "$PYTHON" "$HNG/scripts/verify_shared_pool_sampler_wiring.py" \
          --task_file "$TASK_FILE" --sampler_dir "$ROOT/sampler_seed$seed" --raw_dir "$RAW_DIR" \
          "${REL_TEXT[@]}" --expected_listed "$expected" --variants calibrated soft_mix \
          --output "$ROOT/sampler_seed$seed/wiring.json" > "$ROOT/sampler_seed$seed/wiring.log"
      done
      ;;
    train)
      for seed in $SEEDS; do
        for arm in calibrated soft_mix; do
          check_disk
          log "train $arm seed=$seed"
          "${COMMON[@]}" --stage listed --output_dir "$ROOT/${arm}_seed$seed" --s1_adapter "$S1_ADAPTER" \
            --seed "$seed" --lambda_candidate 1.0 --temperature 1.0 --memory_size 0 \
            --listed_negative_source score_hard \
            --score_hard_manifest "$ROOT/sampler_seed$seed/$arm.jsonl"
        done
      done
      ;;
    decode)
      for seed in $SEEDS; do
        for pair in "b1_seed$seed/b1" "calibrated_seed$seed/listed" "soft_mix_seed$seed/listed"; do
          adapter_done "$ROOT/$pair/adapter" || { log "skip decode $pair (no adapter)"; continue; }
          "$PYTHON" "$HNG/scripts/eval_listed_decode.py" --adapter_dir "$ROOT/$pair/adapter" \
            --eval_file "$EVAL_FILE" --splits test --output_dir "$ROOT/$pair/closed_set" \
            --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache" \
            --decode closed_set --gen_max_length 32 --progress_every 200
        done
      done
      ;;
    *) echo "error: unknown step $step" >&2; exit 1 ;;
  esac
  log "step $step complete"
done
log "all steps complete; headline = <arm>/{b1,listed}/summary.json test.official_em"
