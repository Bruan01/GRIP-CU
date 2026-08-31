#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

"$PYTHON_EXECUTABLE" - <<'PY'
import torch, transformers
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the registered WSL experiment")
print(f"gpu={torch.cuda.get_device_name(0)}")
capability = torch.cuda.get_device_capability(0)
print(f"compute_capability={capability[0]}.{capability[1]}")
PY
"$PYTHON_EXECUTABLE" scripts/runtime_self_test.py
"$PYTHON_EXECUTABLE" scripts/validate_setup.py
printf 'WSL runtime preflight passed\n'
