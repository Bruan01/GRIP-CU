# Experiment Plan — v0.1.3 Candidate Selection and Anti-Copy

## Question

Does training with multiple candidate paths improve later graph-free answering, and does any improvement survive removal of the direct terminal-to-answer copy shortcut?

## Required protocols

1. `direct_answer_only`
2. `oracle_two_stage`
3. `candidate_selection_two_stage`
4. `candidate_selection_anti_copy`
5. `candidate_selection_anti_copy_replay`

The first two provide continuity with v0.1.2. The third isolates candidate selection with visible terminals. The fourth hides every terminal and supplies an unassociated endpoint pool. The fifth adds 20% masked-evidence replay during Stage 2.

## Decision rules

- If full-path candidate selection improves but terminal-masked selection does not, the gain is likely terminal copying.
- If terminal-masked selection improves graph-free validation/test, that is evidence for a stronger matching/internalization signal.
- If replay improves over no replay, Stage-1/Stage-2 distribution shift is a major issue.
- Do not implement a learned prioritizer based on one positive full-path result alone.

## Invariants

- train-only same-depth/different-answer distractors;
- no gold-path label in prompts;
- no path evidence at graph-free test time;
- validation-only checkpoint selection;
- token budget and actual optimizer statistics recorded.
