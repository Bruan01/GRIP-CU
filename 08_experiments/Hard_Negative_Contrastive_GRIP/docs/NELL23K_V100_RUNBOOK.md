# NELL23K Structure-Counterfactual-9 on V100

This run keeps the paper-comparable budget: the complete NELL23K task file,
the frozen paper Stage-1/B1 adapter, K=9 negatives, seed 2026, one listed
training epoch with 10 QA epochs, effective batch 512, and the original
generation plus candidate InfoNCE objective. The V100 profile selects FP16
explicitly because Volta is an FP16-first architecture.

## Stages

Run the stages in this order on the GPU host:

```bash
PYTHON=/home/vipuser/miniconda3/bin/python \
MODEL_CACHE_DIR=/root/GRIP-CU/13_base_method/grip-exp/model_cache \
STEP=preflight bash configs/run_nell23k_v100.sh

PYTHON=/home/vipuser/miniconda3/bin/python STEP=score \
  bash configs/run_nell23k_v100.sh
STEP=freeze bash configs/run_nell23k_v100.sh
STEP=wire bash configs/run_nell23k_v100.sh
ALLOW_TRAIN=1 STEP=train SWANLAB_MODE=offline \
  bash configs/run_nell23k_v100.sh
```

Before the full score stage, benchmark candidate batch sizes 1/8/16/32 and
use the largest stable setting. This only affects offline B1 scoring; listed
training remains microbatch 1 with the protected effective batch.

## Evaluation and selection

Evaluate the frozen paper adapter and the innovation adapter on validation and
test with `scripts/evaluate_adapter.py`. Selection is validation-only; test
must not decide which adapter is retained. The selected adapter is copied once
to the run's `best_adapter/` directory and accompanied by `selection.json`.

Each split directory contains `predictions.jsonl`, `summary.json`, and
`metadata.json`. The top-level run record should also include the exact
environment, run configuration, budget, wiring validation, comparison, and
the stdout/stderr inference log.

The WSL runner additionally writes `stage_status.json`, `stage_history.jsonl`,
`resource_usage.json`, and `artifact_manifest.json` for every stage. The
manifest is the authoritative completion gate: it lists expected artifacts,
missing required files, file sizes/mtimes, and SHA256 hashes at stage finish.
For score, the required set is `candidate_scores.jsonl`, `qa_summary.jsonl`,
`metadata.json`, and `run_stats.json`; for freeze it is the bank, audit,
relation profiles, and metadata; for wire it is top-level `wiring.json`; and
for train it includes the budget, adapter/config, predictions, summary,
metadata, and `run.log`.

## Storage policy

Commit to Git: scripts, configs, tests, and this runbook. Keep out of Git:
model/data files, candidate score dumps, negative banks, checkpoints, LoRA
weights, predictions, and logs. Store those under the remote run directory;
retain only one selected LoRA adapter after validation. If the experiment
summary is small enough for source control, add it as a separate text/JSON
record after the run; never add the weight file or raw score dump.
