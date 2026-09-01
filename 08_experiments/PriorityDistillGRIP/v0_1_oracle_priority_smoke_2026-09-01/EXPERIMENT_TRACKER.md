# Experiment Tracker

| Run ID | Milestone | Variant | Seed | Status | Decision role |
|---|---|---|---:|---|---|
| STATIC-20260901 | M0 | data/supervision/static audit | — | DONE | WSL readiness |
| PD-S42-A | M1 | answer_only | 42 | DONE — 0.3141 test; preliminary stop | baseline |
| PD-S42-Q | M1 | more_qa_equal_token | 42 | DONE — 0.3526 test; preliminary stop | token control |
| PD-S42-R | M1 | random_path_equal_token | 42 | DONE — 0.3397 test; preliminary stop | irrelevant-path control |
| PD-S42-ALL | M1 | all_paths_equal_token | 42 | DONE — 0.3397 test; preliminary stop | no-priority control |
| PD-S42-O | M1 | oracle_priority_equal_token | 42 | DONE — 0.3141 test; preliminary stop | oracle upper bound |
| PD-FULL | M3 | all five methods | 42/43/44 | NOT RUN — blocked by PRELIMINARY_STOP | final gate |
| PD-V02 | M4 | learned prioritizer | — | STOPPED — oracle smoke failed | next method version |

## M1 smoke result

- Run: `wsl3090_priority_distill_smoke_01`
- Decision: `PRELIMINARY_STOP`
- Equal-token audit: PASS (`0.02558%` relative gap); Stage-1 truncation: PASS (`0`).
- Mechanism gates: FAIL (oracle tied answer-only at `0.3141`, below More-QA `0.3526`, random/all-path `0.3397`, and lower on deep 3/4).
- Full seeds and v0.2 learned prioritizer are not authorized.
