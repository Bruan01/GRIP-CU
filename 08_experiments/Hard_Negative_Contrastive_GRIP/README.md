# Hard-Negative Contrastive GRIP

Research project for adding structure-aware contrastive supervision to GRIP (In-Parameter Graph Reasoning through Fine-Tuning Large Language Models).

## Research question

Can hard negatives that preserve local relation, entity, or path structure improve GRIP's ability to distinguish the correct answer from plausible alternatives, while retaining the original generation objective and graph-specific LoRA adapter behavior?

## Working title

**Contrastive In-Parameter Graph Reasoning with Structure-Aware Hard Negatives**

This is a working title, not a novelty claim. The literature review in `RESEARCH_REVIEW.md` records the closest prior work and the boundary of the proposed contribution.

## Project layout

- `RESEARCH_REVIEW.md`: literature search, overlap analysis, and research gap.
- `EXPERIMENT_PLAN.md`: first falsifiable experiment design.
- `IMPLEMENTATION.md`: integration contract with the existing GRIP/RecurrentGRIP snapshot.
- `experiment_template.md`: per-run record template.
- `configs/`: frozen pilot configurations and planned sweeps.
- `src/hard_negative_grip/`: dependency-light candidate generation and contrastive losses.
- `tests/`: unit tests for deterministic sampling, leakage prevention, and losses.
- `results/`: H2 gate scores. The 2026-09-09 storage-adapter rerun is the current verdict (`results/h2_gate_verdict.md`).

## Initial status

Design, candidate audit, and a zero-training H2 gate are in place. There is still no contrastive training run. H2 was re-scored on 2026-09-09 with the MLP storage adapter from `quick01_storage_quick` (`scripts/score_h2_gate.py --recipe storage`). `path_local` is not harder than uniform; `tail_range` shows only a weak signal.

Val/test 10-way lists are aligned to the official GRIP processor. The 2026-09-09 aligned H2 check passed: `listed_relation` is harder than uniform on 159/160 adapter questions and 160/160 base-model questions. That is a prompt-list effect, not a novelty result. Do not start path/tail_range B2–B10 training. A method result would require GRIP + listed contrastive training vs original GRIP on generation EM.

## Quick checks

From the project directory:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
python scripts/align_official_nell23k_lists.py --output_file data/nell23k/recurrent_relation_prediction.aligned.json --report_file data/nell23k/official_alignment_report_smoke.json
PYTHONPATH=src python scripts/audit_candidates.py data/nell23k/recurrent_relation_prediction.aligned.json --output_file results_nell23k_audit_aligned.json
```

Re-run the aligned listed-vs-uniform H2 gate from the RecurrentGRIP snapshot `grip-exp` directory:

```bash
PYTHONPATH=. .venv/bin/python ../../Hard_Negative_Contrastive_GRIP/scripts/score_h2_gate.py --recipe storage --control correct --output ../../Hard_Negative_Contrastive_GRIP/results/h2_gate_results_storage_aligned.json
PYTHONPATH=. .venv/bin/python ../../Hard_Negative_Contrastive_GRIP/scripts/score_h2_gate.py --recipe storage --control none --output ../../Hard_Negative_Contrastive_GRIP/results/h2_gate_results_storage_aligned_noadapter.json
```

## Existing code reused

The implementation is designed to sit beside, rather than modify, the existing snapshot:

`GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp`

Relevant interfaces are documented in `IMPLEMENTATION.md`.
