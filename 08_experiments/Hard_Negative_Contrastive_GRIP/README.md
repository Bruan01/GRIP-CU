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
- `汇报_进展与实验结果.md`: briefing, including Qwen2.5-7B official full val/test decode.
- `REPORT.md`: internal status and H2 verdict; 7B numbers summarized in §4.
- `results/`: H2 gates, 0.5B smoke, 7B training adapters, and full/pilot decode summaries. H2 write-up: `results/h2_gate_verdict.md`.

## Initial status

H2: `path_local` is not harder than uniform; `tail_range` is a weak signal. Do not train B2–B10 on those families. Official 10-way alignment passed listed-vs-uniform as a hygiene check (159/160), not a novelty result.

Method gate: Qwen2.5-7B listed contrast vs original GRIP on official NELL23K val/test (9895). Generation test EM 84.91% → 89.12%; closed-set 86.93% → 92.05%. The 0.5B 96-question smoke still fails (43.8% vs 51.0%). Compute-matched H5 is not done. Details: `汇报_进展与实验结果.md`.

Negative sampling: the official 10-way distractors are uniform draws over the 198 train relations. Homologous uniform negatives (experiment H) beat Stage-1 cosine-pool negatives (experiment I, +0.82 pp on the full 9895), and an offline audit shows why — the cosine top-40 pool captures less of the true distractor mass than uniform sampling does. Partial PCA whitening (experiment I2) repairs the degenerate embedding geometry (mean pairwise cosine 0.8776 → −0.0049, effective negatives 39.1/40 → 15.5) but cannot raise the coverage ceiling of a restricted pool. Details: `汇报_进展与实验结果.md` §4.9–4.10.

## Listed vs original GRIP training

Shared Stage 1 uses the `quick01_storage_quick` recipe (MLP LoRA r=4/alpha=8, full `down/up/gate_proj`). Default model is Qwen2.5-0.5B on the aligned relation-prediction split. `run_listed_vs_b1_qwen7b.sh` switches to Qwen2.5-7B, trains on `grip_nell23k_tasks.json` (paper context + summarization + generated QA), and keeps val/test EM on the aligned 10-way split. The 7B recipe on one 24GB 3090 is `batch=1`, `accum=512`. Stage 2 then forks from that adapter:

- `b1`: generation loss only (original GRIP)
- `listed`: generation + InfoNCE over 9 distractors. Default pool is the official 198 train-graph relations, sampled with the `process.py` rule. `LISTED_NEGATIVE_SOURCE=embed_sim` keeps that vocabulary but prefers Stage-1 cosine neighbors. `LISTED_NEGATIVE_SOURCE=qa_vocab` restores the older 370-relation QA-gold pool used by the 20260913 run.

Both Stage 2 arms use the full QA epoch budget (no S2 early stop). Primary metric is greedy generation exact match.

Server jobs auto-enter a detached tmux session and write mid-run HuggingFace checkpoints (`--save_steps 10`, keep 2). Re-running the same `RUN_DIR` resumes from `trainer_*/checkpoint-*`. `SKIP_TMUX=1` is only for short debug. `FORCE_NEW=1` or `--no_resume` starts a directory/stage from scratch.

```bash
# smoke: 64 train / 32 val / 64 test, aligned official 10-way
SCALE=smoke bash configs/run_listed_vs_b1.sh

# same recipe on the 512/128/512 pilot split
SCALE=pilot bash configs/run_listed_vs_b1.sh

# same listed-vs-B1 fork on Qwen2.5-7B, trained on grip_nell23k_tasks.json
SCALE=smoke bash configs/run_listed_vs_b1_qwen7b.sh

# retrain listed only, homologous 198 train-graph negatives, reuse 20260913 Stage 1
bash configs/run_listed_train_graph_negatives.sh

# retrain listed only, same 198 vocab, negatives prefer Stage-1 cosine neighbors
bash configs/run_listed_embed_negatives.sh

# same as above but the relation table is first passed through a linear transform
# (default partial PCA whitening); pool/temperature/seed stay unchanged
bash configs/run_listed_whiten_negatives.sh

# offline (no GPU): which transform, if any, makes the negatives useful
python scripts/whiten_relation_embeddings.py --mode pca_whiten --k 197 --alpha 0.5
python scripts/audit_relation_geometry.py --markdown results/relation_geometry_audit.md

# freeze the existing 3253×198 B1 score table as an immutable confusion DB
# (offline; do not rescore). Re-mine with LIMIT=0 only if that JSONL is incomplete.
bash configs/run_freeze_confusion_db.sh

# small-scale live dump of the same scorer: 100 relation QA × 198 relations
# (tmux). Raise LIMIT up to 1000; do not use this for the full set.
LIMIT=100 bash configs/run_offline_score_small.sh

# listed-only larger-slice decode; reuse frozen 20260915 B1 predictions
SCALE=pilot bash configs/run_listed_only_decode.sh

# queue that decode behind another GPU job
WAIT_FOR_PATTERN=dpo-full-20260919_000258 \
  SCALE=pilot TMUX_SESSION=expH-pilot-decode \
  RUN_DIR=results/runs/20260919_qwen7b_pilot_listed_only \
  bash configs/run_listed_only_decode.sh

# after that run finishes, rewrite listed-vs-frozen-B1 comparison.json
LISTED_RUN=results/runs/20260918_qwen7b_train_graph_negatives \
  bash configs/compare_listed_to_frozen_b1.sh

# resume the last listed retrain (same RUN_DIR, latest checkpoint)
RESUME_LAST=1 bash configs/run_listed_train_graph_negatives.sh

# resume Stage 2 from a finished Stage-1 adapter (single GPU, sequential)
RUN_DIR=results/runs/<run> RESUME_S1_ADAPTER=$RUN_DIR/s1_adapter \
  SCALE=smoke bash configs/run_listed_vs_b1_qwen7b.sh
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

Full official NELL23K val/test decode on saved 7B adapters (no retraining):

```bash
bash configs/prepare_full_nell23k_eval.sh
bash configs/run_full_decode_eval.sh
```

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
