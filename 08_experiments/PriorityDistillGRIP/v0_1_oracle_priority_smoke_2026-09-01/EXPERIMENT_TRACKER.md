# Experiment Tracker

| Run ID | Milestone | Variant | Seed | Status | Decision role |
|---|---|---|---:|---|---|
| STATIC-20260901 | M0 | data/supervision/static audit | — | DONE | WSL readiness |
| PD-S42-A | M1 | answer_only | 42 | TODO | baseline |
| PD-S42-Q | M1 | more_qa_equal_token | 42 | TODO | token control |
| PD-S42-R | M1 | random_path_equal_token | 42 | TODO | irrelevant-path control |
| PD-S42-ALL | M1 | all_paths_equal_token | 42 | TODO | no-priority control |
| PD-S42-O | M1 | oracle_priority_equal_token | 42 | TODO | oracle upper bound |
| PD-FULL | M3 | all five methods | 42/43/44 | BLOCKED_BY_M1 | final gate |
| PD-V02 | M4 | learned prioritizer | — | BLOCKED_BY_M3 | next method version |
