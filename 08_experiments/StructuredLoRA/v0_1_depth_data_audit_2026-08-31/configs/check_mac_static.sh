#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m compileall -q scripts structured_lora_audit tests
python3 -m unittest discover -s tests -v
python3 scripts/run_depth_data_audit.py --config configs/depth_audit_nell23k.json
python3 - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
summary = json.loads((root / "artifacts/reports/summary.json").read_text())
assert summary["label_artifact"]["rows"] == sum(summary["dataset"]["split_sizes"].values())
assert summary["exact_hop"]["total_tasks"] == 1024
assert summary["decision"]["decision"] in {"GO_ORACLE_PREFIX", "REVISE_DEPTH_AXIS"}
assert (root / "artifacts/reports/REPORT.md").is_file()
print("StructuredLoRA v0.1 static and artifact checks passed")
PY
