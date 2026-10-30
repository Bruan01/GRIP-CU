# 2026-10-29 weight retention and cleanup

The user requested removal of unused weights while preserving later paper work.
This is a one-off cleanup, separate from the running FB training controller.

- `plan.json`: reviewed paths, byte sizes, reasons, protected directories and evidence hashes.
- `deleted_files.jsonl`: per-file deletion ledger. Initial entries have full SHA256; later retired weights have an explicitly labelled first/last-MiB content sample, not a full-file SHA256.
- `retained_adapters.json`: full SHA256 of the retained final paper/candidate adapters.
- `result.json`: completion and post-cleanup validation; absent until validation succeeds.
- `cleanup.log`, `cleanup_weights.py`: actual execution and exact implementation.

Removed: binary weights and optimizer/RNG state from the archived PRUNED NELL
trials; intermediate checkpoints of completed NELL and WN full-budget runs;
and the older WN lambda=0.5 checkpoint, retaining its newer resumable step 50.
Training arguments, tokenizer files, JSON checkpoint state and all other
provenance remain. `WEIGHTS_REMOVED_20261029.json` marks retired directories;
those directories are provenance records and must not be treated as resumable
checkpoints after cleanup.

Retained: NELL B1 and the paper's SC-CL final model; the full-budget HPO model;
step-8 best candidate and resumable checkpoint; step-12 and high-lambda
follow-up models/checkpoints; WN Stage-1/B1, the selected lambda=0.25 final
model, and the latest resumable partial candidates. Current FB, incomplete
CoDEx, base-model caches and datasets are excluded from deletion. Ambiguous
legacy/pilot models are left in place.

Predictions, score dumps, configurations, trial databases, analysis files and
manuscript sources are preserved. Final paper/candidate model hashes are
checked before and after cleanup. Paper retention is also recorded under
`11_paper/artifacts/weight_retention_20261029.json` on successful completion.
