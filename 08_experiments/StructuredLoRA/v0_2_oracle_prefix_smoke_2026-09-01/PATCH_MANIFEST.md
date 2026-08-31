# Patch Manifest — StructuredLoRA v0.2

## Added experiment code

```text
structured_lora/config.py
structured_lora/experiment.py
structured_lora/io_utils.py
structured_lora/metrics.py
structured_lora/modules.py
structured_lora/records.py
structured_lora/routing.py
structured_lora/runtime_data.py
structured_lora/suite.py
scripts/run_experiment.py
scripts/run_suite.py
scripts/summarize_suite.py
scripts/validate_setup.py
scripts/runtime_self_test.py
```

## Added registered configuration and launchers

```text
configs/oracle_prefix_smoke.json
configs/check_static.sh
configs/check_wsl_runtime.sh
configs/run_wsl_smoke.sh
configs/run_wsl_full_seeds.sh
requirements-wsl.txt
```

## Added tests

```text
tests/test_config.py
tests/test_metrics.py
tests/test_records.py
tests/test_routing.py
tests/test_suite.py
```

## Added documentation and output contract

```text
README.md
PROJECT_MEMORY.md
SOURCE_BASELINE.md
PATCH_MANIFEST.md
NEXT_STEP_WSL_PROMPT.md
results/README.md
results/.gitignore
artifacts/setup_audit.json
```

## Explicitly untouched

```text
13_base_method/grip-exp/
08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/
```
