# Implementation Contract

## Existing GRIP snapshot

The first integration target is:

`GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp`

The snapshot already provides:

- `scripts/prepare_recurrent_nell23k.py` for train/validation/test question records;
- `grip/tasks/recurrent_tasks/task_dataset.py` for chat-formatted QA strings;
- `grip/training/train.py` for the two-stage Hugging Face `Trainer` flow;
- `grip/recurrent/model.py` for graph-specific LoRA construction;
- `scripts/run_recurrent_grip.py` for graph adapter save/load and `correct`, `none`, and multi-graph `shuffled` controls;
- `evaluation/recurrent_metrics.py` for exact match, hop, and adapter summaries.

The new project should be imported by `PYTHONPATH=src` and should not edit the baseline snapshot until the standalone audit is complete.

## Integration stages

### Stage A: data and candidate audit

Rewrite val/test 10-way lists with `scripts/align_official_nell23k_lists.py` before generating negatives, so `listed_relation` is taken from the same closed set used at evaluation.

For each training QA triple, call `generate_hard_negatives` with:

- the graph triples used for context;
- all train, validation, and test triples as `all_known_triples`;
- the graph entity and relation vocabularies;
- the question's 10-way list as `listed_relations`;
- a fixed `num_per_kind` and `path_hops`.

`listed_relation` uses the full 9 distractors. Persist candidate provenance alongside each question. Reject or flag a query if it has no candidate in a required family. Do not silently fill a missing family with a random candidate.

### Stage B: answer continuation scoring

For an identical question prefix, tokenize:

- the positive answer continuation;
- each negative answer continuation.

Use teacher-forced token log-probabilities from the same GRIP adapter. Divide each answer log-probability sum by the number of answer tokens. Pass one positive score and a matrix of negative scores to `candidate_infonce_loss` or `margin_ranking_loss`.

The score API should be kept separate from `model.generate`; generation remains the primary evaluation behavior.

### Stage C: custom training loss

Extend the existing trainer/model in a new experiment copy. The custom forward path should return:

`loss = generation_loss + lambda_candidate * candidate_loss + lambda_adapter * adapter_loss`

The existing generation labels and data collator must remain valid. Candidate loss should be disabled with `lambda_candidate=0` for the B1 baseline and adapter loss with `lambda_adapter=0` for B8.

Candidate scoring may require additional model forwards. Record the number of candidate forwards and wall-clock cost. Avoid accidentally training separate adapters for positive and negative candidates; all candidate scores in one query use the same graph adapter.

### Stage D: adapter contrast

For the same prompt, compute the positive continuation score under:

- the correct graph adapter;
- the disabled adapter;
- the shuffled graph adapter when at least two graph adapters exist.

Use the correct adapter score as the positive and the other view scores as negatives in `adapter_contrastive_loss`. The none/shuffled scores are model-view negatives, not factual answer negatives.

### Stage E: evaluation

For every query and candidate family, save:

- raw generated response and parsed answer;
- exact-match correctness;
- normalized continuation scores;
- candidate rank/MRR/Hit@K;
- adapter condition and adapter score margin;
- candidate provenance and structural distance;
- latency and peak memory.

Aggregate by split, candidate family, hop/distance, K, and adapter condition.
