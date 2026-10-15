#!/usr/bin/env bash
# FB15K-237: train-only task generation -> shared S1 -> B1/Random-K -> full decode.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
export RUN_DIR="${RUN_DIR:-$HNG/results/runs/20261015_fb15k237_qwen7b_b1_randomk}"
export TMUX_SESSION="${TMUX_SESSION:-fb15k237-b1-randomk}"
export DATA_DIR="${DATA_DIR:-$HNG/data/fb15k237}"
export RAW_DIR="${RAW_DIR:-$HNG/data/raw_datasets/fb15k237}"
export TASK_FILE="$RUN_DIR/tasks/grip_fb15k237_tasks.json"
export TASK_CACHE_DIR="$RUN_DIR/tasks/cache"
export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$RUN_DIR/tasks"
# Also protect one RUN_DIR against launches under different tmux session names.
exec 9>"$RUN_DIR/pipeline.lock"
flock -n 9 || { echo "Pipeline already running: $RUN_DIR" >&2; exit 1; }
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'rc=$?; echo "[pipeline] exit=$rc time=$(date --iso-8601=seconds)"; exit "$rc"' EXIT
cd "$CODE_DIR"
echo "[pipeline] FB15K-237 started $(date --iso-8601=seconds)"
echo "run_dir=$RUN_DIR tmux_session=$TMUX_SESSION"
"$PYTHON" "$HNG/scripts/prepare_fb15k237.py" --raw_dir "$RAW_DIR" --out_dir "$DATA_DIR"
# Freeze generation/training settings and source hashes before spending GPU time.
"$PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
raw = Path(os.environ['RAW_DIR'])
plan = {
    'dataset': 'fb15k237', 'seed': 2026, 'model': 'qwen-7b',
    'num_context_qa': 8000, 'num_reason_qa': 2000, 'num_summarization': 6000,
    'reference_split': 'train', 's1_epochs': 1, 's2_epochs': 10,
    'negative_source': 'train_graph', 'negative_k': 9, 'lambda_candidate': 1.0,
    'batch': 1, 'accum': 512, 'save_steps': 10, 'save_total_limit': 2,
    'raw_sha256': {name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
                   for name in ['train.txt', 'valid.txt', 'test.txt', 'rel2text.json', 'entity2wikidata.json']},
}
p = Path(os.environ['RUN_DIR']) / 'experiment_plan.json'
if p.exists() and json.loads(p.read_text()) != plan:
    raise SystemExit('Existing experiment plan differs; use a separate RUN_DIR.')
p.write_text(json.dumps(plan, indent=2) + '\n')
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
print('[preflight] GPU:', torch.cuda.get_device_name(0), 'memory:', torch.cuda.mem_get_info())
x = torch.ones((8, 8), device='cuda'); print('[preflight] CUDA matmul:', (x @ x)[0, 0].item())
PY
"$PYTHON" "$CODE_DIR/scripts/generate_nell23k_tasks.py" \
  --input-file "$DATA_DIR/processed_train.json" \
  --ref-file "$DATA_DIR/processed_train.json" \
  --task-file "$TASK_FILE" --task-cache-dir "$TASK_CACHE_DIR" \
  --seed 2026 --model-name qwen-7b --model-source local \
  --model-cache-dir "$CODE_DIR/model_cache" --local-files-only \
  --task-generator-model-name qwen-7b --no-task-generator-use-vllm \
  --task-generator-batch-size 1 --task-gen-max-length 1000 \
  --num-context-qa 8000 --num-reason-qa 2000 --num-summarization 6000 \
  --no-sample-node-attribute-task --no-repharse-context-qa \
  --no-context-upsampling --format-as-instruction --involve-qa-epochs 10
# CPU preflight: relation QA must receive nine train-vocabulary negatives.
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.task_file import build_qa_assets_from_task_texts
raw = Path(os.environ['RAW_DIR'])
payload = json.loads(Path(os.environ['TASK_FILE']).read_text())
order = load_train_relation_order(raw, relation_text_file=raw / 'rel2text.json')
_, metas = build_qa_assets_from_task_texts(payload['qa_samples'], seed=2026, relation_order=order)
listed = [m for m in metas if m['listed_relations']]
assert payload['context_samples'] and listed, 'No context or no matched relation QA'
assert all(len(m['listed_relations']) == 9 and m['matched_train_relation'] not in m['listed_relations'] for m in listed)
print('[preflight] context=', len(payload['context_samples']), 'QA=', len(metas), 'contrastive QA=', len(listed), 'relations=', len(order))
PY
COMMON=(
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py"
  --input_file "$TASK_FILE" --eval_file "$DATA_DIR/recurrent_relation_prediction_smoke.json"
  --output_dir "$RUN_DIR" --model_name qwen-7b --model_cache_dir "$CODE_DIR/model_cache"
  --raw_dir "$RAW_DIR" --relation_text_file "$RAW_DIR/rel2text.json"
  --lora_r 4 --lora_alpha 8 --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1 --involve_qa_epochs 10 --s1_stop_loss_threshold 0.15
  --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --s1_gradient_checkpointing
  --learning_rate 1e-3 --weight_decay 1e-4 --max_grad_norm 1.0
  --listed_negative_source train_graph --listed_negative_k 9 --lambda_candidate 1.0
  --memory_size 0 --temperature 1.0 --gen_max_length 32 --seed 2026
  --save_steps 10 --save_total_limit 2
)
echo '[pipeline] shared FB S1, then B1 and Random-K sequentially'
"${COMMON[@]}" --stage all
# Both full decoders append per question and reuse completed predictions.
for variant in b1 listed; do
  "$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
    --adapter_dir "$RUN_DIR/$variant/adapter" \
    --eval_file "$DATA_DIR/recurrent_relation_prediction_full.json" \
    --output_dir "$RUN_DIR/full/$variant" --model_name qwen-7b \
    --model_cache_dir "$CODE_DIR/model_cache" --decode both --gen_max_length 32 --progress_every 50
done
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --eval_file "$DATA_DIR/recurrent_relation_prediction_full.json" \
  --output_dir "$RUN_DIR/full" --compare --eval_name fb15k237-full --decode both
echo "[pipeline] completed $(date --iso-8601=seconds)"
