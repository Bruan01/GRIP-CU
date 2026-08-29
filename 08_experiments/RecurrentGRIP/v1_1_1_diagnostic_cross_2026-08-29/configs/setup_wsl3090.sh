#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-}"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "error: this setup script is intended for WSL2" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is unavailable inside WSL2" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install uv in WSL2 before running this script" >&2
  exit 1
fi

cd "$CODE_DIR"
uv venv --python "$PYTHON_VERSION" .venv
if [[ -n "$PYTORCH_CUDA_INDEX" ]]; then
  uv pip install --python .venv/bin/python \
    "torch==$TORCH_VERSION" \
    --index-url "$PYTORCH_CUDA_INDEX" \
    --extra-index-url https://pypi.org/simple
else
  uv pip install --python .venv/bin/python "torch==$TORCH_VERSION"
fi
uv pip install --python .venv/bin/python \
  "transformers==4.56.1" \
  "peft==0.17.1" \
  "accelerate==1.10.1" \
  "datasets==4.0.0" \
  "numpy==2.2.6" \
  "scipy==1.16.1" \
  "torch-geometric==2.6.1" \
  "openai==1.107.0" \
  "anthropic==0.66.0" \
  "tenacity==9.1.2" \
  "tiktoken==0.11.0" \
  "pytest>=8,<9" \
  tqdm

.venv/bin/python "$VERSION_DIR/configs/verify_wsl3090.py"
echo "WSL2 RTX 3090 environment ready: $CODE_DIR/.venv"
