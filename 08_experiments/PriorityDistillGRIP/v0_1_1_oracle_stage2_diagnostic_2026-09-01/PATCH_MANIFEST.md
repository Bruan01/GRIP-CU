# Patch Manifest — v0.1.1 Oracle Stage-2 Diagnostic

This is a new immutable experiment snapshot. v0.1 is not modified.

## Added

- `configs/oracle_stage2_diagnostic.json`: one-seed oracle-only, eight Stage-2 epochs, constant-LR default;
- `configs/run_wsl_diagnostic.sh`: conda/WSL-friendly launcher with the known local Qwen snapshot;
- `scripts/run_diagnostic.py`: training entry point;
- `scripts/summarize_diagnostic.py`: checkpoint curve report;
- `priority_distill/experiment.py`: per-stage checkpointing and dual-condition evaluation;
- `priority_distill/supervision.py`: gold-path diagnostic prompt builder;
- documentation and setup audit.

## Weight policy

`checkpoints/**/adapter_model.pt` is local-only and ignored by Git. JSON/JSONL/Markdown metrics and predictions are eligible for review and commit after an experiment run.
