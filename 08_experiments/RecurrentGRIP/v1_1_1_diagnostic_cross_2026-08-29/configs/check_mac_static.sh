#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"

cd "$CODE_DIR"
python3 - <<'PY'
import ast
from pathlib import Path

roots = [Path("arguments"), Path("data"), Path("evaluation"), Path("grip"), Path("scripts"), Path("tests")]
paths = [Path("recurrent_context_sampling.py"), Path("recurrent_cross_audit.py")]
paths.extend(Path("../configs").glob("*.py"))
for root in roots:
    paths.extend(root.rglob("*.py"))
count = 0
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count += 1
print(f"AST syntax check: PASS ({count} Python files)")
PY
bash -n ../configs/*.sh
for config in ../configs/*.json; do
  python3 -m json.tool "$config" >/dev/null
done
python3 -m unittest \
  tests.test_prepare_recurrent_nell23k \
  tests.test_context_sampler \
  tests.test_recurrent_args \
  tests.test_recurrent_metrics \
  tests.test_recurrent_cross_audit \
  tests.test_export_diagnostic_artifacts \
  -v
echo "macOS static checks: PASS"
