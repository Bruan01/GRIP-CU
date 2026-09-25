# Dataset

- Task file: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json`
- Scores: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_full/candidate_scores.jsonl`
- QA summaries: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_full/qa_summary.jsonl`
- Frozen DB (20260921 mining table, not this dump): `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_confusion_db_frozen_b1/confusion_db.jsonl`
- Live 20-QA rescore: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_full/audit/live_20qa_rescore.json`
- Split label: `train`
- Stage-2 QA samples: 12014
- Matchable relation QA expected: 3253
- Unmatched relation-like golds skipped: 44
- Non-relation QA skipped: 8717
- Scored QA: 3253
- Candidate rows: 644094
- Vocab size: 198
- Gold rows: 3253
- Valid negatives: 640711
- Alternative true: 130
- Duplicate QA: 0
- Empty relations: 0
- NaN: 0
- Inf: 0
- token_length <= 0: 0
- Entity pair missing: 76
- Completeness: PASS (0 failures)

# Model / Checkpoint

- Checkpoint: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter`
- Exists: True
- Is B1 adapter: True
- Is listed adapter: False
- B1 run_metadata: `{"stage": "s2", "variant": "b1", "model_name": "qwen-7b", "lambda_candidate": 0.0, "qa_samples": 12014, "candidate_forwards": 0, "last_generation_loss": 0.13537873327732086, "last_candidate_loss": 0.0, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 512, "gradient_checkpointing": false, "seconds": 13194.5, "s1_adapter": "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/s1_adapter", "created_at_utc": "2026-09-14T06:11:57.726180+00:00"}`
- Listed adapter (not used for scoring): `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/listed/adapter`
- Tokenizer: `Qwen2TokenizerFast`
- dtype: `bfloat16`
- candidate_batch_size: `8`
- scoring_version: `offline_confusion_v1`
- GPU occupancy at audit time: `{"available": false, "error": "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 595.91", "torch_cuda_available": true, "torch_error": null, "live_rescore_possible": true}`

This scorer is the **B1 generation-only Stage-2 adapter**, not the listed InfoNCE adapter and not a paper RecurrentGRIP snapshot.

# Candidate Scoring Definition

- `score(A|Q)` = length-normalized continuation mean log-probability
- Prefix ends at the last `<answer>` tag, inclusive
- Continuation is the raw relation string only
- EOS and `</answer>` are excluded
- Temperature is **not** applied in the scorer; pairwise_confusion uses T=1.0
- Training path: `ListedContrastiveTrainer._score_candidate_rows` → `score_candidate_rows`
- Offline path: `score_relations` → `score_candidates`
- Shared implementation: `score_packed_candidate_rows`

# Relation Vocabulary Construction

- Source split: **train.txt only**
- Construction: `process.py` insertion order (`unique_rel = list(set())`)
- Size: 198
- Gold alignment key: `matched_train_relation`
- Observed matched golds: 193 / 198

# False-negative Filtering

- Filter source: union of `train.txt`, `valid.txt`, `test.txt` via `known_pair_relations()`
- Gold is always invalid (`invalid_reason=gold_relation`)
- Other known pair relations are scored but marked `is_valid_negative=false` (`alternative_true_relation`)
- Sampled alternative_true: 100; present in KG: 100; missing: 0
- Sampled valid negatives: 500; known-true leaked into valid pool: 0
- KG limitation: known_pair_relations unions train/valid/test. Facts absent from all three splits cannot be filtered. QA with a missing entity pair skip pair-level filtering entirely.

# Data Leakage Analysis

- Relation vocab split: train.txt insertion order (process.py unique_rel)
- True-relation filtering splits: train.txt, valid.txt, test.txt
- Confusion scorer checkpoint: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter`
- Checkpoint training data: `{"task_file": "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json", "qa_samples": 12014, "context_samples": 80820, "b1_objective": "generation loss only (lambda_candidate=0)", "listed_objective": "generation + InfoNCE (lambda_candidate=1)", "b1_saw_scored_qa": true, "b1_saw_eval_relation_prediction_labels": false}`
- Offline mining QA: `{"source": "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json", "split_label": "train", "n_matchable_relation_qa": 3253, "n_unmatched_relation_qa": 44, "n_non_relation_qa": 8717, "n_scored": 3253}`
- Pair overlap: `{"entity_pair_missing": 76, "qa_pair_in_train": 2979, "qa_pair_in_valid": 33, "qa_pair_in_test": 35, "alternative_true_from_train": 65, "alternative_true_from_valid": 31, "alternative_true_from_test": 34}`

## A. Used as test-set confusion analysis

Dump scores Stage-2 train QA only. It is not a test-set confusion table. Do not treat these ranks as test EM / test 10-way analysis.

## B. Used as a future training sampler

- **RED:** known_pair_relations() unions train.txt, valid.txt, and test.txt. A future train sampler that drops alternative_true candidates therefore uses test KG structure, not just train negatives.
- **RED:** 35 scored QA entity pairs also appear in test.txt; 34 alternative_true labels come from test triples; 31 come from valid triples.
- **RED:** B1 generation training already saw the scored Stage-2 questions. These scores are in-distribution teacher values, not a held-out val set.

B1 **did** see the scored Stage-2 questions during generation-only training. That is in-distribution teacher scoring, not test-label leakage. It **is** leakage if these scores are treated as a held-out validation of B1.

Any future sampler that consumes `is_valid_negative` from this dump **uses test KG structure** unless the filter is rebuilt from train-only triples.

# Numerical Consistency

- All candidate rows recomputed for `score_gap` and `pairwise_confusion`: failures=0
- All QA `negative_mass` sums: failures=0
- Random 100 row subsample: failures=0
- All QA ranks recomputed: failures=0
- Random 100 QA rank subsample: failures=0

# Reproducibility

- Training vs offline 20-QA: n_qa=20 max_abs=0 mean_abs=0 rank_mismatches=0
- Dump vs training 20-QA: n_qa=20 max_abs=0 mean_abs=0 rank_mismatches=0
- Dump vs offline 20-QA: n_qa=20 max_abs=0 mean_abs=0 rank_mismatches=0
- Training run1 vs run2: n_qa=20 max_abs=0 mean_abs=0 rank_mismatches=0
- Offline run1 vs run2: n_qa=20 max_abs=0 mean_abs=0 rank_mismatches=0
- Dump vs historical compare20 (same dump, earlier 20-QA rerun): n_qa=20 max_abs=0 mean_abs=0 rank_mismatches=0
- Dump vs frozen 20260921 mining table: n_qa=3253 max_abs=0.21875 mean_abs=0.000479507 rank_mismatches=7931
- Frozen-table note: results/runs/20260923_confusion_db_frozen_b1 comes from the 20260921 score-hard mining table, not this production dump. Score drift there is expected and is not an identity check for this dataset.
- GPU nondeterminism: bf16 packed attention can move scores by one ulp when candidate_batch_size changes (documented for 16/32 vs 8). Same batch size should match within 1e-5.
- Live 7B 20-QA rescore status: PASS

# Known Limitations

- Open-world KG: facts outside train/valid/test cannot be filtered.
- 76 QA have no parsed entity pair, so alternative-true filtering is skipped for those rows.
- 44 unmatched relation-like golds are outside the 198 train-graph names and are not in this dump.
- Analysis artifacts under `analysis/` may have been built from the frozen 20260921 table; `confusion_vocab/` from this dump. Do not mix them.
- NVML `nvidia-smi` can fail with a driver/library mismatch even when `torch.cuda.is_available()` is true.

# Final Status

**REVISE**

Blocking issues: test_structure_in_train_filter
All issues: test_structure_in_train_filter, frozen_20260921_table_is_not_this_dump
