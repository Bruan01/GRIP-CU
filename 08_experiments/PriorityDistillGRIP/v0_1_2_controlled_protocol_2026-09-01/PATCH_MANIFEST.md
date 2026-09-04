# Patch Manifest — v0.1.2 Controlled Protocol

This is an independent protocol revision based on v0.1.1 core training utilities.

- `priority_distill/controlled.py`: direct and oracle protocols, matched token-budget segments, validation-only selection workflow support.
- `configs/direct_answer_only.json`: direct answer-only protocol.
- `configs/oracle_two_stage.json`: oracle two-stage protocol.
- `configs/run_controlled_wsl.sh`: WSL + `guardenv` runner.
- `scripts/run_controlled.py`: controlled experiment entry point.
- `scripts/summarize_controlled.py`: validation checkpoint selection and held-out test report.
- `tests/test_controlled.py`: protocol/config invariants.

Checkpoint adapter weights stay ignored by Git. Reports, logs, metrics, predictions, and audits are publishable artifacts.
