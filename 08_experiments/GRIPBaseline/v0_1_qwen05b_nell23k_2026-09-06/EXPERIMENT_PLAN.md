# Experiment plan

## Research question

What is the performance of the original GRIP training/inference protocol when
the fine-tuned backbone is reduced from Qwen2.5-7B to Qwen2.5-0.5B on the
same NELL23K protocol?

## Hypothesis

The 0.5B model should provide a reproducible lower-capacity reference for
interpreting later GRIP extensions. The result is expected to be lower than
the official 7B result; the useful comparison is against other methods that
also use Qwen2.5-0.5B and the same data/protocol, not a claim of scale-invariant
performance.

## Protocol

1. Use the repository's current `outputs/data/nell23k/processed_test.json`.
2. Use the original GRIP task generation and LoRA training code.
3. Use Qwen2.5-7B as task generator by default and Qwen2.5-0.5B as the
   fine-tuned/inference backbone.
4. Keep inference graph-free (`no_graph_context=True`, no subgraph/index).
5. Evaluate the complete processed test split for the formal baseline.
6. Report exact-match (`em`) as the primary metric and retain `f1` and `hit`
   as secondary diagnostics.
7. Save the processed test data, task cache, predictions, adapter, logs,
   config/runtime events, model path, and SHA256 values.

## Interpretation gate

- A smoke run is only a pipeline check, not a paper baseline.
- The formal baseline requires all test rows and no test labels in task
  generation or prompts.
- Record the actual dataset row count and SHA256 because local NELL23K file
  counts can differ across releases.
- Compare later E03/E09 results with matching model size, training budget,
  seed, and graph-access boundary.
