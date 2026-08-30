#!/usr/bin/env bash
set -euo pipefail
VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
CONDA_ENV="${CONDA_ENV:-guardenv}"
cd "$CODE_DIR"
conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import ast
from pathlib import Path
roots = [Path("arguments"), Path("data"), Path("evaluation"), Path("grip"), Path("scripts"), Path("tests"), Path("candidate_energy")]
paths = [Path("recurrent_context_sampling.py"), Path("recurrent_cross_audit.py")]
paths.extend(Path("../configs").glob("*.py"))
for root in roots: paths.extend(root.rglob("*.py"))
for path in paths: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"AST syntax check: PASS ({len(paths)} Python files)")
PY
bash -n ../configs/*.sh
conda run --no-capture-output -n "$CONDA_ENV" python -m unittest \
  tests.test_prepare_recurrent_nell23k tests.test_context_sampler tests.test_recurrent_args \
  tests.test_recurrent_metrics tests.test_recurrent_cross_audit tests.test_export_diagnostic_artifacts \
  tests.test_candidate_energy -v
echo "static checks: PASS"
