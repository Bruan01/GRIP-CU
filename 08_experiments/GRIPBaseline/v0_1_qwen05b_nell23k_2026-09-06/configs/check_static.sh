#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
GRIP_ROOT="$REPO_ROOT/13_base_method/grip-exp"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

bash -n "$GRIP_ROOT/scripts/run_nell23k_qwen05b.sh"
bash -n "$REPO_ROOT/08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/configs/run_qwen05b_nell23k_wsl.sh"
"$PYTHON_EXECUTABLE" -m py_compile \
    "$GRIP_ROOT/constants.py" \
    "$GRIP_ROOT/models/ft_models/hf.py" \
    "$GRIP_ROOT/scripts/run_grip.py" \
    "$GRIP_ROOT/scripts/mp_wrapper.py"

"$PYTHON_EXECUTABLE" - "$GRIP_ROOT/constants.py" <<'PY'
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
found = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in {"HF_DECODER_ONLY_LLMS", "MODELSCOPE_DECODER_ONLY_LLMS"}:
            found[node.targets[0].id] = ast.literal_eval(node.value)
for name in ("HF_DECODER_ONLY_LLMS", "MODELSCOPE_DECODER_ONLY_LLMS"):
    assert found[name]["qwen-0.5b"] == "Qwen/Qwen2.5-0.5B-Instruct"
print("QWEN05B_MAPPING_READY")
PY

DRY_RUN_OUTPUT="$(mktemp)"
trap 'rm -f "$DRY_RUN_OUTPUT"' EXIT
(
    cd "$GRIP_ROOT"
    RUN_ID=static_qwen05b_dry_run \
    "$GRIP_ROOT/scripts/run_nell23k_qwen05b.sh" --dry-run >"$DRY_RUN_OUTPUT"
)
grep -q -- '--model_name qwen-0.5b' "$DRY_RUN_OUTPUT"
grep -q -- '--task_generator_model_name qwen-7b' "$DRY_RUN_OUTPUT"
grep -q -- '--no_graph_context True' "$DRY_RUN_OUTPUT"
grep -q -- '--use_subgraph False' "$DRY_RUN_OUTPUT"
grep -q -- '--index_format False' "$DRY_RUN_OUTPUT"

echo 'GRIP_QWEN05B_STATIC_CHECKS_PASSED'
