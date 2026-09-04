#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_EXECUTABLE" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_EXECUTABLE" scripts/validate_setup.py --config configs/direct_answer_only.json --output artifacts/direct_answer_only_setup_audit.json
"$PYTHON_EXECUTABLE" scripts/validate_setup.py --config configs/candidate_selection_anti_copy_replay.json --output artifacts/anti_copy_replay_setup_audit.json
"$PYTHON_EXECUTABLE" scripts/runtime_self_test.py
"$PYTHON_EXECUTABLE" -m py_compile scripts/*.py priority_distill/*.py tests/*.py
bash -n configs/*.sh
"$PYTHON_EXECUTABLE" - <<'PY'
from pathlib import Path
from priority_distill.config import load_config
expected = {
    'direct_answer_only.json': 'direct_answer_only',
    'oracle_two_stage.json': 'oracle_two_stage',
    'candidate_selection_two_stage.json': 'candidate_selection_two_stage',
    'candidate_selection_anti_copy.json': 'candidate_selection_anti_copy',
    'candidate_selection_anti_copy_replay.json': 'candidate_selection_anti_copy_replay',
}
for name, protocol in expected.items():
    config = load_config(Path('configs') / name)
    assert config['protocol'] == protocol
for name, protocol in {'candidate_index_two_stage.json':'candidate_index_two_stage', 'candidate_index_anti_copy.json':'candidate_index_anti_copy', 'candidate_index_anti_copy_replay.json':'candidate_index_anti_copy_replay'}.items():
    config = load_config(Path('configs') / name)
    assert config['protocol'] == protocol
print('v0.1.4 configs validated')
PY
printf 'PriorityDistill-GRIP v0.1.4 explicit-selection static checks passed\n'
