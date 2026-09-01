# Patch Manifest

## New experiment files

```text
08_experiments/PriorityDistillGRIP/
├── README.md
└── v0_1_oracle_priority_smoke_2026-09-01/
    ├── README.md
    ├── SOURCE_BASELINE.md
    ├── PATCH_MANIFEST.md
    ├── PROJECT_MEMORY.md
    ├── NEXT_STEP_WSL_PROMPT.md
    ├── ENVIRONMENT_WSL3090.md
    ├── EXPERIMENT_PLAN.md
    ├── EXPERIMENT_TRACKER.md
    ├── requirements-wsl.txt
    ├── configs/
    ├── priority_distill/
    ├── scripts/
    ├── tests/
    ├── artifacts/
    └── results/
```

## Repository-level index updates

- `README.md`: current primary thread changed from StructuredLoRA to PriorityDistill-GRIP.
- `08_experiments/README.md`: new versioned experiment registered.
- `TODO_PRIORITY_DISTILL.md`: root WSL execution TODO and decision boundary.
- `.gitignore`: allow PriorityDistill metrics, predictions, and reports while keeping adapters and logs ignored.

## Explicit non-changes

- `13_base_method/grip-exp/`: unchanged.
- Existing experiment version directories: unchanged.
- Existing result files: unchanged.
- Existing untracked `FactorGRIP/v0_1_1_candidate_energy_audit_2026-08-30/` and `13_base_method/grip-exp.zip`: not touched.
