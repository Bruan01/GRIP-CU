# Project Memory — PriorityDistill-GRIP v0.1.3

## Environment

- WSL2 + RTX 3090 24GB
- conda environment: `guardenv`
- local model snapshot: `/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- model weights/checkpoints remain ignored; metrics, predictions, logs and reports may be committed.

## Purpose

Test whether candidate-path supervision transfers to graph-free answering after removing the direct terminal-to-answer copy shortcut.

## Key implementation decisions

- Full candidate selection uses one gold and three train-only same-depth/different-answer distractors, without a gold label and with deterministic random order.
- Anti-copy selection replaces every path terminal by `<MASKED_TERMINAL>` and supplies the four possible endpoints in an unassociated randomized list.
- Stage 2 replay uses 80% answer-only and 20% masked-evidence input-token budget.
- Stage-2 learning rate is `3e-4`, lower than v0.1.2's `1e-3`.
- Stage 1 and Stage 2 segments record input tokens, supervised answer tokens, examples, and optimizer steps.
- Graph-free evaluation never includes candidate evidence. Checkpoint selection uses validation only.

## Interpretation boundary

A positive full-path result can still be terminal copying. A positive terminal-masked graph-free result is stronger evidence that the model learned an association/matching signal. A replay gain indicates that abrupt removal of evidence caused forgetting. No learned prioritizer should be implemented from a single full-path gain.

## Implementation status — 2026-09-01

- Fixed the replay protocol validation name to `candidate_selection_anti_copy_replay`; the old `candidate_selection_replay` name must not be used.
- Replaced the copied v0.1.2 suite-gate unit tests with v0.1.3 protocol validation tests. The old aggregate gate is not applicable to this controlled-protocol experiment.
- Static validation passed in `guardenv`: 24 unit tests, setup audits for direct and anti-copy-replay, runtime self-test, Python compilation, shell syntax, and all five config validations.
- All five GPU runs completed on WSL2/RTX 3090 using `guardenv`; validation-selected metrics are recorded below. Checkpoint weights remain local/ignored.

## Completed GPU runs — 2026-09-01

Validation-selected held-out graph-free accuracy:

| protocol | run id | selected checkpoint | validation | test |
|---|---|---:|---:|---:|
| `direct_answer_only` | `wsl3090_v013_direct_20260901_01` | `stage2_epoch4` | 0.3224 | 0.3462 |
| `oracle_two_stage` | `wsl3090_v013_oracle_20260901_01` | `stage2_epoch6` | 0.3289 | 0.3141 |
| `candidate_selection_two_stage` | `wsl3090_v013_candidate_20260901_01` | `stage2_epoch6` | 0.3487 | 0.2885 |
| `candidate_selection_anti_copy` | `wsl3090_v013_anticopy_20260901_01` | `stage2_epoch4` | 0.3158 | 0.2756 |
| `candidate_selection_anti_copy_replay` | `wsl3090_v013_anticopy_replay_20260901_01` | `stage2_epoch7` | 0.3289 | 0.2885 |

Interpretation of this single-seed mechanism run:

- The anti-copy candidate protocol did not beat the direct answer-only baseline (0.2756 vs 0.3462 test), so there is no evidence here that candidate selection transferred into stronger graph-free answering.
- Replay recovered a small amount relative to anti-copy (0.2885 vs 0.2756, +0.0128 absolute), suggesting that the Stage-2 evidence-to-no-evidence switch may cause some forgetting, but replay did not close the gap to direct training.
- The ordinary candidate protocol was also below direct (0.2885 vs 0.3462), so any earlier apparent candidate benefit should not be treated as proof of a learned prioritizer.
- Stage-1 graph-free accuracy is expected to be 0.0 because Stage 1 only trains with evidence; the meaningful Stage-1 check is oracle-evidence accuracy. Stage-2 graph-free accuracy rises gradually, while oracle-evidence accuracy generally falls, showing a transfer/retention trade-off rather than a simple optimization failure.
- Do not claim a definitive seed-independent conclusion from these numbers. The current conclusion is only that this controlled run failed to demonstrate a robust anti-copy candidate-selection benefit.
