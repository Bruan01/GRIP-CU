# Experiment Plan — PriorityDistill-GRIP v0.1.6 Joint Trace + Answer

## Question

Can graph-free joint supervision of intermediate trace tokens and a separately supervised final entity improve compositional answering without exposing graph evidence at evaluation time?

## Protocol

- Stage 1 and Stage 2 use `graph_free_trace_answer_joint`.
- Target format masks the terminal inside the trace and places the final entity in a separate `Answer:` segment.
- Graph-free validation/test prompts contain no path or graph evidence.
- Tokenized input budgets, supervised tokens and optimizer steps are recorded.
- Checkpoint selection is validation-only.

## Required diagnostics

1. Graph-free answer, trace and joint exact match.
2. Train-vs-held-out audit.
3. Gold trace with masked terminal.
4. Gold trace with visible terminal as an oracle-only sanity check.
5. Teacher-forced candidate ranking to separate answer preference from free-generation failure.
6. Corrected generation-budget audit with enough tokens for the longest target.

## Decision rule

Do not claim knowledge internalization from loss reduction alone. A promising follow-up must improve held-out graph-free answer/joint metrics over the matched direct baseline and should remain positive on a seen-composition versus novel-composition split. If free generation remains poor but candidate ranking is strong, prioritize candidate-set scoring/constrained decoding before adding a learned router.
