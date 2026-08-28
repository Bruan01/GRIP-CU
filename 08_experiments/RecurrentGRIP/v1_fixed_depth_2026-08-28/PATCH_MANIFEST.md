# Patch Manifest

## Modified from Original GRIP

- `grip-exp/arguments/__init__.py`
- `grip-exp/constants.py`
- `grip-exp/data/raw_datasets/clegr/process.py`
- `grip-exp/grip/training/__init__.py`

## Added

- `grip-exp/arguments/recurrent_args.py`
- `grip-exp/grip/recurrent/`
- `grip-exp/grip/tasks/recurrent_tasks/`
- `grip-exp/grip/training/recurrent_trainer.py`
- `grip-exp/evaluation/recurrent_metrics.py`
- `grip-exp/scripts/prepare_recurrent_clegr.py`
- `grip-exp/scripts/run_recurrent_grip.py`
- `grip-exp/scripts/run_recurrent_pilot.py`
- `grip-exp/scripts/analyze_recurrent_results.py`
- `grip-exp/tests/`
- `grip-exp/docs/RECURRENT_GRIP.md`
- `ENVIRONMENT_WSL3090.md`
- `configs/check_mac_static.sh`
- `configs/setup_wsl3090.sh`
- `configs/verify_wsl3090.py`
- `configs/test_wsl3090.sh`
- `configs/prepare_clegr.sh`
- `configs/run_qwen05b_smoke_wsl.sh`
- `configs/run_qwen05b_pilot.sh`
- `configs/analyze_pilot.sh`
- `configs/pilot_qwen05b.json`

## Runtime-specific changes inside the snapshot

- `arguments/recurrent_args.py` adds `evaluation_device` and `require_cuda`;
- `scripts/run_recurrent_grip.py` explicitly places the reloaded evaluation model on CUDA/CPU/MPS;
- WSL launchers require CUDA and use the version-local `.venv`;
- smoke/pilot launchers create immutable per-run subdirectories and apply GNU `timeout`;
- no platform environment is copied into the version registry.

## Intentionally Unchanged

- `13_base_method/grip-exp/`；
- Original `scripts/run_grip.py`；
- Original `grip/training/train.py`；
- Original default LoRA construction in `models/utils.py`。
