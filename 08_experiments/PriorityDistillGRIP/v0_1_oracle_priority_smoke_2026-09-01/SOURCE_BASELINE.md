# Source Baseline

## Repository snapshot

- Repository: `GRIP-CU`
- Remote: `https://github.com/Bruan01/GRIP-CU.git`
- Branch at creation: `wsl/nell23k-smoke-20260829`
- Parent commit at creation: `87281bc`
- Creation date: `2026-09-01`

## Reused scientific assets

Read-only data source:

```text
08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/artifacts/exact_hop/
```

Runtime design reference:

```text
08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/
```

The v0.2 StructuredLoRA WSL run established that Qwen2.5-0.5B-Instruct with rank-8 LoRA targeting `down_proj`, `up_proj`, and `gate_proj` runs on RTX 3090. PriorityDistill copies and simplifies the standard monolithic LoRA logic into its own package.

## Isolation policy

- No file under `13_base_method/grip-exp/` was modified.
- Runtime code does not import Original GRIP or StructuredLoRA.
- Every future behavior/config change must create a new version directory.
- Existing StructuredLoRA, RecurrentGRIP, and FactorGRIP evidence remains immutable history.
