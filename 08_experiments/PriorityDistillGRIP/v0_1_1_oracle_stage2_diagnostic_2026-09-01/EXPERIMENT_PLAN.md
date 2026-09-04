# Oracle Stage-2 Diagnostic Plan

## Hypothesis

The v0.1 oracle result may be caused by either:

1. insufficient Stage-2 answer-only adaptation; or
2. Stage-1 evidence-conditioned copying of the visible terminal answer.

## Controlled design

- one method: `oracle_priority_equal_token`;
- one seed: 42;
- same strict NELL23K exact-hop train/validation/test splits;
- Stage 1 keeps the v0.1 equal-input-token reference;
- Stage 2 uses answer-only rows, one epoch at a time;
- Stage 2 default: 8 epochs, constant learning rate after warm-up;
- evaluate before training, after Stage 1, and after every Stage-2 epoch;
- evaluate both graph-free and oracle-evidence prompts;
- save adapter-only checkpoints locally; metrics/predictions are commit-eligible.

## Decision boundary

This is a diagnosis, not a new mechanism claim. Do not run full seeds or implement a learned prioritizer based only on this run. The next protocol revision depends on the curve:

- graph-free recovery: revise Stage-2 budget/scheduler and then re-run a controlled comparison;
- no recovery with strong oracle-evidence accuracy: revise evidence serialization / anti-copy construction;
- neither condition learns: inspect optimizer/data/model setup before any mechanism conclusion.
