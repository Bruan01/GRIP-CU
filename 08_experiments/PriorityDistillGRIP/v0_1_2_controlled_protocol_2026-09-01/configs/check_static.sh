#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/validate_setup.py
python3 -m py_compile scripts/*.py priority_distill/*.py tests/*.py
bash -n configs/*.sh
python3 - <<'PY'
from pathlib import Path
from priority_distill.config import load_config
for name, protocol in [('direct_answer_only.json','direct_answer_only'), ('oracle_two_stage.json','oracle_two_stage')]:
    config = load_config(Path('configs') / name)
    assert config['protocol'] == protocol
print('controlled configs validated')
PY
printf 'PriorityDistill-GRIP v0.1.2 controlled static checks passed\n'
