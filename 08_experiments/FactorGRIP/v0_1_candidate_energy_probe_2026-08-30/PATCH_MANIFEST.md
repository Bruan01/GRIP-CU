# Patch Manifest — FactorGRIP v0.1 Candidate-Energy Probe

## Baseline

- Source snapshot: `08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/`
- Parent commit at fork: `2ee0c86`
- Base method touched: **no** (`13_base_method/grip-exp/` is unchanged)
- Run scope: NELL23K only, Qwen2.5-0.5B, inference-only, evaluation recurrence `K=1`.

## Added in this version

- `candidate_energy/scoring.py`: batched candidate sequence log-likelihood and length normalization.
- `candidate_energy/constrained.py`: trie-constrained candidate generation.
- `candidate_energy/probe.py`: correct / none / wrong-depth adapter controls and decoder orchestration.
- `candidate_energy/artifacts.py`: immutable `RUN_ID` and resume/overwrite guards plus provenance.
- `scripts/run_candidate_energy_probe.py`: WSL runner; no training and no target-derived prompt.
- `scripts/analyze_candidate_energy.py`: accuracy, paired McNemar, bootstrap CI, and Go/Stop gate.
- `tests/test_candidate_energy.py`: pure-Python/tensor unit tests for ranking, normalization, leakage, permutation, trie, and guards.
- `configs/run_nell23k_candidate_probe_wsl.sh`: conda `guardenv` execution wrapper.

## Controls unavailable

A shuffled adapter is unavailable because the input is one graph and there is no second independently trained graph adapter. The available negative control is `wrong_depth` (the `train_k2` adapter evaluated at `K=1`); the result summary must not treat that as a shuffled control.
