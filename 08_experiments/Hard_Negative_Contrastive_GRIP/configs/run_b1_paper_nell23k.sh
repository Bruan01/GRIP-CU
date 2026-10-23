#!/usr/bin/env bash
# Strict-paper GRIP B1 on NELL23K (GRIP KDD'26 Sec. 4 + Table 13). Not run yet:
# review before launching. Pipeline: train-only graph/k-shot reference ->
# paper task generation (Qwen2.5-7B) -> S1 (plain-text context+summary) ->
# S2 B1 (answer-only L_QA, early stop 0.15, max 10 epochs) -> test-only EM.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SEED="${SEED:-2026}"
export RUN_DIR="${RUN_DIR:-$HNG/results/runs/paper_b1_nell23k_seed$SEED}"
TMUX_SESSION="${TMUX_SESSION:-paper-b1-nell23k-$SEED}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
# Test split = all NELL23K test triples present in the raw file (see docs for 4,944 vs paper 4,952).
EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
TRAIN_REF="$RUN_DIR/data/processed_train.json"
TASK_FILE="$RUN_DIR/tasks/grip_nell23k_paper_tasks.json"
TASK_CACHE_DIR="$RUN_DIR/tasks/cache"
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
[[ -x "$PYTHON" ]] || { echo "error: missing Python: $PYTHON" >&2; exit 1; }
mkdir -p "$RUN_DIR/data" "$RUN_DIR/tasks"
exec 9>"$RUN_DIR/pipeline.lock"
flock -n 9 || { echo "Pipeline already running: $RUN_DIR" >&2; exit 1; }
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[paper-b1] NELL23K seed=$SEED started $(date --iso-8601=seconds)"

# 1. Train-only graph + k-shot reference (official process.py has no train QA file).
"$PYTHON" "$HNG/scripts/build_nell23k_train_ref.py" --raw_dir "$RAW_DIR" --output_file "$TRAIN_REF" --seed 2026

# 2. Paper task generation: Ns=6000, Nr=2000, Nc=8000; node+edge context QA;
#    rooted 3-hop x 3 subgraphs; random reasoning type; plain-text S1 samples.
"$PYTHON" "$CODE_DIR/scripts/generate_nell23k_tasks.py" \
  --input-file "$TRAIN_REF" --ref-file "$TRAIN_REF" --require-train-ref \
  --task-file "$TASK_FILE" --task-cache-dir "$TASK_CACHE_DIR" \
  --seed 2026 --model-name qwen-7b --model-source local \
  --model-cache-dir "$CODE_DIR/model_cache" --local-files-only \
  --task-generator-model-name qwen-7b --no-task-generator-use-vllm \
  --task-generator-batch-size 1 --task-gen-max-length 1000 \
  --num-context-qa 8000 --num-reason-qa 2000 --num-summarization 6000 \
  --sample-node-attribute-task --no-repharse-context-qa --no-context-upsampling \
  --no-format-as-instruction --uniform-sampling --rooted-reasoning-subgraph \
  --random-reasoning-type --keep-parallel-edges --train-entity-nodes-only \
  --involve-qa-epochs 10

# 3-4. S1 then S2 B1 with Table 13 NELL23K/Qwen settings.
COMMON=(
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py"
  --input_file "$TASK_FILE" --eval_file "$EVAL_FILE" --eval_splits test
  --output_dir "$RUN_DIR" --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache"
  --raw_dir "$RAW_DIR" --listed_negative_source train_graph
  --lora_r 4 --lora_alpha 8 --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1 --s1_stop_loss_threshold 0.15
  --involve_qa_epochs 10 --s2_stop_loss_threshold 0.15 --s2_min_epoch 1
  --qa_loss_mask answer_only
  --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --s1_gradient_checkpointing
  --learning_rate 1e-3 --weight_decay 0 --adam_epsilon 1e-4 --max_grad_norm 1.0
  --lambda_candidate 0 --gen_max_length 32 --seed "$SEED"
  --save_steps 10 --save_total_limit 2
)
"${COMMON[@]}" --stage s1
"${COMMON[@]}" --stage b1
echo "[paper-b1] completed $(date --iso-8601=seconds); headline = b1/summary.json test.official_em"
