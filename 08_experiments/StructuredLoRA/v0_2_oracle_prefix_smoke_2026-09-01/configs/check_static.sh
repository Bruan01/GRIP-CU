#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/validate_setup.py
python3 -m py_compile scripts/*.py structured_lora/*.py tests/*.py
dry_run_file="$(mktemp)"
trap 'rm -f "$dry_run_file"' EXIT
python3 scripts/run_suite.py --dry-run --run-id static_dry_run > "$dry_run_file"
test "$(grep -c '^COMMAND ' "$dry_run_file")" -eq 9
if rg -n '/Users/mac|13_base_method/grip-exp/' scripts structured_lora tests 2>/dev/null; then
    echo 'error: runtime source contains a machine-specific path or imports the base repository' >&2
    exit 1
fi
printf 'StructuredLoRA v0.2 static checks passed\n'
