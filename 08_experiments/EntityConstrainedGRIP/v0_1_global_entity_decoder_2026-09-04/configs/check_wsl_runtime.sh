#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
PYTHON="${PYTHON:-/home/kieran/miniconda3/envs/guardenv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
cd "$ROOT"
CHECKPOINT_ARGS=("$@")
nvidia-smi
"$PYTHON" -c 'import torch, transformers; assert torch.cuda.is_available(); print("CUDA", torch.cuda.get_device_name(0), "torch", torch.__version__, "transformers", transformers.__version__)'
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON" scripts/build_entity_vocabulary.py
"$PYTHON" scripts/validate_setup.py --require-checkpoints "${CHECKPOINT_ARGS[@]}"
"$PYTHON" scripts/runtime_self_test.py --checkpoint "${RUNTIME_SELF_TEST_CHECKPOINT:-direct_answer_only_seed43}" --model-name-or-path "$MODEL_NAME_OR_PATH"
printf 'WSL_PREFLIGHT_PASSED repo=%s\n' "$REPO"
