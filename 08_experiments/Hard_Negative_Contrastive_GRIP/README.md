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
- `src/hard_negative_grip/`: candidate generation, contrastive losses, and the listed Stage-2 trainer.
- `scripts/train_listed_contrastive.py`: shared Stage 1, then original GRIP vs listed contrast.
- `tests/`: unit tests for deterministic sampling, leakage prevention, losses, and listed training pieces.
- `results/`: H2 gate scores. The 2026-09-09 storage-adapter rerun is the current verdict (`results/h2_gate_verdict.md`).

## Initial status

Design, candidate audit, and a zero-training H2 gate are in place. The first training comparison is original GRIP vs GRIP + listed 10-way contrastive: shared Stage 1 graph storage, then two Stage 2 forks. H2 was re-scored on 2026-09-09 with the MLP storage adapter from `quick01_storage_quick` (`scripts/score_h2_gate.py --recipe storage`). `path_local` is not harder than uniform; `tail_range` shows only a weak signal.

Val/test 10-way lists are aligned to the official GRIP processor. The 2026-09-09 aligned H2 check passed: `listed_relation` is harder than uniform on 159/160 adapter questions and 160/160 base-model questions. That is a prompt-list effect, not a novelty result. Do not start path/tail_range B2–B10 training. The method gate is generation EM of GRIP + listed contrast vs original GRIP.

## Listed vs original GRIP training

Shared Stage 1 uses the `quick01_storage_quick` recipe (Qwen2.5-0.5B, MLP LoRA r=4/alpha=8, full `down/up/gate_proj`). Stage 2 then forks from that adapter:

- `b1`: generation loss only (original GRIP)
- `listed`: generation + InfoNCE over the prompt's 9 distractors

Both Stage 2 arms use the full QA epoch budget (no S2 early stop). Primary metric is greedy generation exact match.

```bash
# smoke: 64 train / 32 val / 64 test, aligned official 10-way
SCALE=smoke bash configs/run_listed_vs_b1.sh

# same recipe on the 512/128/512 pilot split
SCALE=pilot bash configs/run_listed_vs_b1.sh
```

Or call the trainer directly from the RecurrentGRIP `grip-exp` directory after Stage 1:

```bash
PYTHONPATH=.:../../Hard_Negative_Contrastive_GRIP/src \
  .venv/bin/python ../../Hard_Negative_Contrastive_GRIP/scripts/train_listed_contrastive.py \
  --input_file ../../Hard_Negative_Contrastive_GRIP/data/nell23k/recurrent_relation_prediction.aligned.json \
  --output_dir ../../Hard_Negative_Contrastive_GRIP/results/runs/listed_vs_b1_smoke \
  --stage all
```

`--stage` can be `s1`, then `b1` / `listed` in parallel with `--s1_adapter`, then `compare`.

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
