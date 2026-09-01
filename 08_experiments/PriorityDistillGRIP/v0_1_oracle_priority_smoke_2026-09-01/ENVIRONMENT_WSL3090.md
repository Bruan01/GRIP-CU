# WSL2 + RTX 3090 Environment

## Recommended setup

```bash
cd 08_experiments/PriorityDistillGRIP/v0_1_oracle_priority_smoke_2026-09-01
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-wsl.txt
```

If PyTorch must be installed from a CUDA-specific wheel index, install the WSL/CUDA-compatible torch build first, then install the remaining requirements.

## Preflight

```bash
bash configs/check_wsl_runtime.sh
```

This checks:

- torch and transformers imports;
- CUDA availability;
- GPU name and compute capability;
- toy LoRA forward/backward and adapter serialization;
- data SHA and split sizes;
- supervision leakage invariants.

## Smoke

```bash
RUN_ID=wsl3090_priority_distill_smoke_01 bash configs/run_wsl_smoke.sh
```

## Resume

Use the same `RUN_ID`. Completed method/seed folders are skipped; partial folders are replaced.

```bash
RUN_ID=wsl3090_priority_distill_smoke_01 bash configs/run_wsl_smoke.sh
```

## Full seeds

Run only after inspecting the one-seed report:

```bash
RUN_ID=wsl3090_priority_distill_full_01 bash configs/run_wsl_full_seeds.sh
```

## Expected outputs

```text
results/runs/<RUN_ID>/
├── answer_only/seed_42/
├── more_qa_equal_token/seed_42/
├── random_path_equal_token/seed_42/
├── all_paths_equal_token/seed_42/
├── oracle_priority_equal_token/seed_42/
├── suite_summary.json
├── suite_metrics.csv
└── REPORT.md
```
