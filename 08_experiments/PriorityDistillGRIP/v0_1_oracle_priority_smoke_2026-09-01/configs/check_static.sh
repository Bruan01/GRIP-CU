#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/build_supervision.py
python3 scripts/validate_setup.py
python3 -m py_compile scripts/*.py priority_distill/*.py tests/*.py
bash -n configs/*.sh
dry_run_file="$(mktemp)"
trap 'rm -f "$dry_run_file"' EXIT
python3 scripts/run_suite.py --dry-run --run-id static_dry_run > "$dry_run_file"
test "$(grep -c '^COMMAND ' "$dry_run_file")" -eq 6
if rg -n '/Users/mac|13_base_method/grip-exp/' scripts priority_distill tests configs/*.json 2>/dev/null; then
    echo 'error: runtime source contains a machine-specific path or direct base-repository dependency' >&2
    exit 1
fi
printf 'PriorityDistill-GRIP v0.1 static checks passed\n'
