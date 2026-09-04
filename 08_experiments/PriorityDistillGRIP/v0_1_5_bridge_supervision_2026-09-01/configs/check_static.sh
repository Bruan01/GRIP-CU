#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_EXECUTABLE" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_EXECUTABLE" scripts/validate_setup.py --config configs/direct_answer_only.json --output artifacts/direct_answer_only_setup_audit.json
"$PYTHON_EXECUTABLE" scripts/validate_setup.py --config configs/graph_free_trace.json --output artifacts/graph_free_trace_setup_audit.json
"$PYTHON_EXECUTABLE" scripts/validate_setup.py --config configs/candidate_index_anti_copy_bridge.json --output artifacts/candidate_index_anti_copy_bridge_setup_audit.json
"$PYTHON_EXECUTABLE" scripts/runtime_self_test.py
"$PYTHON_EXECUTABLE" -m py_compile scripts/*.py priority_distill/*.py tests/*.py
bash -n configs/*.sh
"$PYTHON_EXECUTABLE" - <<'PY'
from pathlib import Path
from priority_distill.config import load_config
expected = {
    'direct_answer_only.json': 'direct_answer_only',
    'graph_free_trace.json': 'graph_free_trace',
    'candidate_index_anti_copy_bridge.json': 'candidate_index_anti_copy_bridge',
}
for name, protocol in expected.items():
    config = load_config(Path('configs') / name)
    assert config['protocol'] == protocol
print('v0.1.5 bridge configs validated')
PY
printf 'PriorityDistill-GRIP v0.1.5 bridge-supervision static checks passed\n'
