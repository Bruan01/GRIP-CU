# Implementation contract

## Baseline boundary

The branch is based on `origin/main` and is a sibling of
`origin/recurrentgrip-nell23k-20260904`. The runtime baseline is the original
GRIP implementation in `13_base_method/grip-exp`. No recursive executor,
recursive depth, or recursive task schema is part of this experiment.

## Training boundary

Only the candidate relation learning stage is changed:

```text
original GRIP generation loss
    + lambda_candidate * InfoNCE(gold relation, 9 sampled relations)
```

Stage 1 graph-context training and the B1 generation-only arm remain the
control path. `ListedContrastiveTrainer` computes the same continuation score
for the positive relation and the nine candidate relations, then applies
cross-entropy over that decision set.

## ErrorMemory-9 boundary sampler

The sampler is a frozen, train-only preprocessing step:

1. B1 greedily generates answers for training relation questions.
2. A gold-conditioned memory counts in-vocabulary B1 errors.
3. The B1 score dump supplies the candidate score violation.
4. Known true relations for the same entity pair are removed using train-only
   triples.
5. Nine negatives are sampled without replacement from the mixture of a
   uniform floor and the ErrorMemory/violation softmax.
6. The JSONL bank is validated and then consumed by Stage 2.

The bank stores the sampler parameters and provenance. Validation and test
predictions never enter the bank.

## File-level contract

- `task_file.py`: original GRIP task JSON and standard processed graph record.
- `official_lists.py`: train relation vocabulary and paper candidate protocol.
- `error_memory_sampling.py`: ErrorMemory construction and frozen K=9 bank.
- `listed_training.py`: candidate InfoNCE and unchanged generation objective.
- `mine_rollout_hard_negatives.py`: B1 train-only free-generation errors.
- `score_relation_candidates_offline.py`: B1 train-only full-vocabulary scores.
- `mine_error_memory_negatives.py`: CPU-only bank freezer.
- `verify_error_memory_wiring.py`: CPU-only leakage, ID, vocabulary, and K checks.
- `train_listed_contrastive.py`: original GRIP model loading, Stage 2 training,
  standard graph evaluation, and SwanLab callbacks.
- `configs/run_error_memory_wsl.sh`: preflight, rollout, score, freeze, wire,
  and explicitly gated train stages.

## Required assets

The WSL runner accepts explicit paths for all machine-local assets. At minimum
it needs:

- `GRIP_BASELINE_DIR`: initialized `13_base_method/grip-exp`;
- `INPUT_FILE`: full original GRIP task JSON with `context_samples` and
  `qa_samples`;
- `EVAL_FILE`: standard processed graph record with `graph`, `questions`, and
  `answers`;
- NELL23K `train.txt`, `valid.txt`, and `test.txt` under `RAW_DIR`;
- frozen S1/B1 adapters and the local model cache before GPU stages.

The default runner step is read-only `preflight`; no training starts unless
`ALLOW_TRAIN=1 STEP=train` is explicitly set.
