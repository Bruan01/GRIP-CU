#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"

cd "$CODE_DIR"
python3 - <<'PY'
import ast
from pathlib import Path

roots = [Path("arguments"), Path("data"), Path("evaluation"), Path("grip"), Path("scripts"), Path("tests")]
count = 0
for root in roots:
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += 1
print(f"AST syntax check: PASS ({count} Python files)")
PY
bash -n ../configs/*.sh
python3 -m json.tool ../configs/pilot_qwen05b.json >/dev/null
echo "macOS static checks: PASS"
