# Results directory

Each invocation creates an immutable run folder:

```text
results/runs/<RUN_ID>/<METHOD>/seed_<SEED>/
```

A method/seed folder contains:

- `run_summary.json`
- `adapter_model.pt`
- `adapter_config.json`
- `predictions_validation.jsonl`
- `predictions_test.jsonl`
- `environment.json`
- `data_audit.json`

The run root additionally receives:

- `suite_summary.json`
- `suite_metrics.csv`
- `REPORT.md`

Do not overwrite a completed run. Use a new `RUN_ID`; `--overwrite` exists only for a failed local smoke.
