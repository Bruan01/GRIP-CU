#!/usr/bin/env bash
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
PYTHON="/home/kieran/miniconda3/envs/guardenv/bin/python"
TRAIN_RUNNER="$HNG/configs/run_error_memory_wsl.sh"
HPO_ROOT="$HNG/results/hpo/wn18rr_sccl_seed2026"
DATA_ROOT="$REPO_ROOT/04_benchmark/data/wn18rr"
ARTIFACT_ROOT="$HNG/results/runs/wn18rr_sccl_3090_20261023/artifacts"
SAMPLER_DIR="$ARTIFACT_ROOT/structure_counterfactual_bank"
MODEL_CACHE_DIR="/mnt/e/GRIP-CU/model_cache"
RUNG1_SESSION="wn18rr_sccl_hpo_s2026_rung1"

mkdir -p "$HPO_ROOT"

echo "[promote] waiting for rung1 session=$RUNG1_SESSION"
while tmux has-session -t "$RUNG1_SESSION" 2>/dev/null; do
  sleep 30
done

for slug in 0p25 0p5 2p0; do
  summary="$HPO_ROOT/lambda${slug}_rung1_valid500/summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "[promote] missing rung1 result: $summary" >&2
    exit 2
  fi
done

"$PYTHON" - "$HPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [
    ("0p25", 0.25),
    ("0p5", 0.5),
    ("2p0", 2.0),
]
ranked = []
for slug, value in candidates:
    path = root / f"lambda{slug}_rung1_valid500" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    metric = summary.get("valid") or summary.get("all")
    if not metric or "em" not in metric or "count" not in metric:
        raise SystemExit(f"missing valid EM/count in {path}")
    ranked.append({"slug": slug, "lambda": value, "em": float(metric["em"]), "count": int(metric["count"])})
ranked.sort(key=lambda row: (-row["em"], row["lambda"]))
selected = ranked[:2]
(root / "rung1_ranking.json").write_text(
    json.dumps({"metric": "valid500_em", "ranking": ranked, "promoted": selected}, indent=2) + "\n",
    encoding="utf-8",
)
(root / "rung1_promoted.tsv").write_text(
    "".join(f"{row['slug']}\t{row['lambda']}\t{row['em']:.8f}\n" for row in selected),
    encoding="utf-8",
)
print("[promote] rung1 ranking: " + ", ".join(f"lambda={r['lambda']} EM={r['em']:.4f}" for r in ranked))
print("[promote] advancing: " + ", ".join(f"lambda={r['lambda']}" for r in selected))
PY

latest_checkpoint() {
  local trainer_dir="$1"
  find "$trainer_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -V | tail -n 1 | sed "s#^#$trainer_dir/#"
}

run_to_step() {
  local slug="$1"
  local lambda_value="$2"
  local target_step="$3"
  local stage_tag="$4"
  local run_id="wn18rr_sccl_hpo_s2026_lc${slug}"
  local run_dir="$HNG/results/runs/$run_id"
  local trainer_dir="$run_dir/trainer_listed"
  local checkpoint

  checkpoint="$(latest_checkpoint "$trainer_dir")"
  if [[ -z "$checkpoint" || ! -f "$checkpoint/trainer_state.json" ]]; then
    echo "[promote] no resumable Trainer checkpoint under $trainer_dir" >&2
    exit 3
  fi
  echo "[promote] lambda=$lambda_value target_step=$target_step resume=$checkpoint"
  ALLOW_TRAIN=1 \
    STEP=train \
    DATASET=wn18rr \
    NEGATIVE_SOURCE=structure_counterfactual \
    SEED=2026 \
    LAMBDA_CANDIDATE="$lambda_value" \
    STOP_AFTER_STEP="$target_step" \
    DEFER_EVAL=1 \
    RUN_VARIANT="hpo_${stage_tag}_lc${slug}" \
    RUN_ID="$run_id" \
    RUN_DIR="$run_dir" \
    INPUT_FILE="$DATA_ROOT/tasks/grip_wn18rr_tasks.json" \
    EVAL_FILE="$DATA_ROOT/processed/processed_val.json" \
    RAW_DIR="$DATA_ROOT/raw" \
    MODEL_CACHE_DIR="$MODEL_CACHE_DIR" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    SAMPLER_DIR="$SAMPLER_DIR" \
    EXPECTED_LISTED=2709 \
    NO_RESUME=0 \
    SKIP_TRAIN=0 \
    RESUME_FROM_CHECKPOINT="$checkpoint" \
    SKIP_TMUX=1 \
    SAVE_STEPS=25 \
    SAVE_TOTAL_LIMIT=2 \
    INVOLVE_QA_EPOCHS=5 \
    TRAIN_LEARNING_RATE=1e-3 \
    CONTRASTIVE_TEMPERATURE=1.0 \
    TORCH_DTYPE=bfloat16 \
    bash "$TRAIN_RUNNER"
}

evaluate_full_valid() {
  local slug="$1"
  local lambda_value="$2"
  local rung="$3"
  local run_dir="$HNG/results/runs/wn18rr_sccl_hpo_s2026_lc${slug}"
  local eval_dir="$HPO_ROOT/lambda${slug}_${rung}_fullvalid"
  if [[ -f "$eval_dir/summary.json" ]]; then
    echo "[promote] lambda=$lambda_value $rung full-valid already evaluated"
    return
  fi
  mkdir -p "$eval_dir"
  echo "[promote] lambda=$lambda_value $rung full-valid evaluation"
  "$PYTHON" "$HNG/scripts/evaluate_adapter.py" \
    --adapter "$run_dir/listed/adapter" \
    --eval_file "$DATA_ROOT/processed/processed_val.json" \
    --output_dir "$eval_dir" \
    --model_name qwen-7b \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --dtype bfloat16 \
    --gen_max_length 1000 \
    --progress_every 50 \
    --resume \
    --label "sccl_lambda${lambda_value}_${rung}_seed2026" \
    > "$eval_dir/evaluation.log" 2>&1
}

while IFS=$'\t' read -r slug lambda_value screen_em; do
  [[ -n "$slug" ]] || continue
  run_to_step "$slug" "$lambda_value" 50 rung2
  evaluate_full_valid "$slug" "$lambda_value" rung2
done < "$HPO_ROOT/rung1_promoted.tsv"

"$PYTHON" - "$HPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
promoted = []
for line in (root / "rung1_promoted.tsv").read_text(encoding="utf-8").splitlines():
    slug, value, _ = line.split("\t")
    summary = json.loads((root / f"lambda{slug}_rung2_fullvalid" / "summary.json").read_text(encoding="utf-8"))
    metric = summary.get("valid") or summary.get("all")
    promoted.append({"slug": slug, "lambda": float(value), "em": float(metric["em"]), "count": int(metric["count"])})
promoted.sort(key=lambda row: (-row["em"], row["lambda"]))
baseline_path = root / "lambda1_valid" / "summary.json"
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
baseline_metric = baseline.get("valid") or baseline.get("all")
winner = promoted[0]
decision = {
    "selection_metric": "full_valid_em",
    "promoted_rung1": promoted,
    "winner": winner,
    "lambda1_baseline_em": float(baseline_metric["em"]),
    "baseline_count": int(baseline_metric["count"]),
}
(root / "rung2_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
(root / "rung2_winner.tsv").write_text(f"{winner['slug']}\t{winner['lambda']}\t{winner['em']:.8f}\n", encoding="utf-8")
print(f"[promote] rung2 winner lambda={winner['lambda']} EM={winner['em']:.4f}; incumbent={baseline_metric['em']:.4f}")
PY

IFS=$'\t' read -r winner_slug winner_lambda winner_em < "$HPO_ROOT/rung2_winner.tsv"
run_to_step "$winner_slug" "$winner_lambda" 125 rung3
evaluate_full_valid "$winner_slug" "$winner_lambda" rung3

"$PYTHON" - "$HPO_ROOT" "$DATA_ROOT" "$HNG" "$MODEL_CACHE_DIR" "$winner_slug" "$winner_lambda" <<'PY'
import json
import sys
from pathlib import Path

root, data_root, hng, cache, slug, value = map(Path, sys.argv[1:])
root = Path(sys.argv[1]); data_root = Path(sys.argv[2]); hng = Path(sys.argv[3]); cache = Path(sys.argv[4])
slug = sys.argv[5]; value = float(sys.argv[6])
valid_path = root / f"lambda{slug}_rung3_fullvalid" / "summary.json"
baseline_path = root / "lambda1_valid" / "summary.json"
valid = json.loads(valid_path.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
valid_metric = valid.get("valid") or valid.get("all")
baseline_metric = baseline.get("valid") or baseline.get("all")
beats = float(valid_metric["em"]) > float(baseline_metric["em"])
outcome = {
    "winner_lambda": value,
    "winner_valid_em": float(valid_metric["em"]),
    "winner_valid_count": int(valid_metric["count"]),
    "lambda1_valid_em": float(baseline_metric["em"]),
    "lambda1_valid_count": int(baseline_metric["count"]),
    "beats_incumbent_on_valid": beats,
    "test_evaluated": False,
}
if beats:
    import subprocess
    eval_dir = root / f"lambda{slug}_final_test"
    eval_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(hng / "scripts" / "evaluate_adapter.py"),
        "--adapter", str(hng / "results" / "runs" / f"wn18rr_sccl_hpo_s2026_lc{slug}" / "listed" / "adapter"),
        "--eval_file", str(data_root / "processed" / "processed_test.json"),
        "--output_dir", str(eval_dir),
        "--model_name", "qwen-7b",
        "--model_cache_dir", str(cache),
        "--dtype", "bfloat16",
        "--gen_max_length", "1000",
        "--progress_every", "50",
        "--resume",
        "--label", f"sccl_lambda{value}_final_test_seed2026",
    ]
    with (eval_dir / "evaluation.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    test = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    test_metric = test.get("test") or test.get("all")
    outcome["test_evaluated"] = True
    outcome["test_em"] = float(test_metric["em"])
    outcome["test_count"] = int(test_metric["count"])
(root / "search_outcome.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
print(json.dumps(outcome, indent=2))
PY

echo "[promote] promotion search complete"
