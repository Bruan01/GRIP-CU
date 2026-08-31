# Source Baseline

## Parent research repository

```text
repository: GRIP-CU
branch at implementation: wsl/nell23k-smoke-20260829
parent commit: 97372c44ad3a73f7f82651cebaa0ed0db647bec0
```

## Original GRIP repository

```text
path: 13_base_method/grip-exp/
commit: 2835b440bfd2c4de36f0380ae19bc1c22e6cb459
status at implementation: clean
```

The original GRIP repository is read-only for this experiment. Its selected LoRA target modules are mirrored in v0.2 configuration, but no source file is imported at runtime and no original file is modified.

## Data source

```text
08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/artifacts/exact_hop/
```

The static setup audit records SHA256 for all three exact-hop splits and fails if expected row counts change.
