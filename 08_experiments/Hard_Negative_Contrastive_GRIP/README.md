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
- `results/`: reserved for run artifacts; no results are claimed yet.

## Initial status

This directory contains the first design and a small, testable implementation of the objective. It does not yet contain a completed GPU training result. The first execution target is a NELL23K smoke run using the cached Qwen2.5-0.5B setup in the existing RecurrentGRIP experiment snapshot.

## Quick checks

From the project directory:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

## Existing code reused

The implementation is designed to sit beside, rather than modify, the existing snapshot:

`GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp`

Relevant interfaces are documented in `IMPLEMENTATION.md`.
