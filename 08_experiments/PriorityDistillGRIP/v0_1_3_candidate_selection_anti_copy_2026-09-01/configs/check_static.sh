#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/validate_setup.py --config configs/direct_answer_only.json --output artifacts/direct_answer_only_setup_audit.json
python3 scripts/validate_setup.py --config configs/candidate_selection_anti_copy_replay.json --output artifacts/anti_copy_replay_setup_audit.json
python3 scripts/runtime_self_test.py
python3 -m py_compile scripts/*.py priority_distill/*.py tests/*.py
bash -n configs/*.sh
python3 - <<'PY'
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
print('v0.1.3 configs validated')
PY
printf 'PriorityDistill-GRIP v0.1.3 anti-copy static checks passed\n'
