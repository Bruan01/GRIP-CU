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
config = load_config(Path('configs/oracle_stage2_diagnostic.json'))
assert config['diagnostic']['conditions'] == ['graph_free', 'oracle_evidence']
assert config['training']['stage2_epochs'] >= 2
print('diagnostic config validated')
PY
printf 'PriorityDistill-GRIP v0.1.1 diagnostic static checks passed\n'
