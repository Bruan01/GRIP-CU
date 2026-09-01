# PriorityDistill-GRIP v0.1 Smoke Results Analysis

- **Run ID**: `wsl3090_priority_distill_smoke_01`
- **Run date**: 2026-09-01 (Asia/Shanghai)
- **Runtime**: WSL2 Ubuntu + NVIDIA GeForce RTX 3090 24GB
- **Environment**: conda `guardenv`, Python 3.10.19, PyTorch 2.12.0+cu130, Transformers 4.57.3, PEFT 0.17.1
- **Backbone**: local Hugging Face snapshot of `Qwen/Qwen2.5-0.5B-Instruct`
- **Decision**: `PRELIMINARY_STOP`

## Result table

| Method | Test accuracy | Deep 3/4 | d1 | d2 | d3 | d4 | Stage-1 input tokens | Truncated examples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `answer_only` | 0.3141 (49/156) | 0.4872 | 0.1538 | 0.1282 | 0.5385 | 0.4359 | 0 | 0 |
| `more_qa_equal_token` | **0.3526 (55/156)** | **0.5128** | 0.1795 | 0.2051 | **0.6154** | 0.4103 | 199366 | 0 |
| `random_path_equal_token` | 0.3397 (53/156) | 0.4872 | 0.1538 | **0.2308** | 0.4615 | **0.5128** | 199337 | 0 |
| `all_paths_equal_token` | 0.3397 (53/156) | 0.4744 | **0.1795** | **0.2308** | 0.4872 | 0.4615 | 199315 | 0 |
| `oracle_priority_equal_token` | 0.3141 (49/156) | 0.4487 | 0.1538 | 0.2051 | 0.4615 | 0.4359 | 199321 | 0 |

## Gate audit

The equal-token and prompt-integrity checks passed:

- Stage-1 token relative gap: `0.0002558` (0.02558%), below the 1% limit.
- Stage-1 prompt truncation: `0/0` truncated examples for all path methods.
- All five methods completed with finite validation/test metrics.
- Every `run_summary.json` records `inference_graph_access: false`.
- All runs executed with CUDA available on the RTX 3090; peak recorded allocation remained below 9 GiB.

The mechanism gates failed:

- Oracle minus `answer_only`: `0.0000`, required `>= 0.0200`.
- Oracle minus equal-token More-QA: `-0.0385`, required `>= 0.0100`.
- Oracle minus random path: `-0.0256`, required `> 0`.
- Oracle minus all paths: `-0.0256`, required `> 0`.
- Oracle deep 3/4 minus `answer_only`: `-0.0385`, required `> 0`.

## Interpretation

This one-seed smoke does **not** support the claim that a perfect gold path is a useful training-time teacher under the registered equal-token design. The strongest result is the equal-token More-QA control, while the oracle path is tied with answer-only and below every path-bearing control on test accuracy. The oracle also loses on deep 3/4-hop accuracy, so no multi-hop improvement claim is justified.

The result should be treated as a preliminary stop rather than a final three-seed statistical conclusion. Because the registered one-seed gate is already negative on every mechanism comparison, the protocol does not authorize the full-seed suite or a v0.2 learned prioritizer. No learned scorer, DPO, retrieval component, or v0.2 implementation was added.

## Reproducibility artifacts

The committed run directory contains the suite report, summary, metrics CSV, predictions, token-budget audits, data/environment audits, and training logs. The generated `adapter_model.pt` files are intentionally ignored and are not part of the Git commit.

## Why this is probably not simply “the oracle did not converge”

The logs show two different phenomena:

1. **Stage 1 fit is strong for the path methods.** The final cumulative token-weighted Stage-1 losses were `0.130` for oracle and `0.201` for all-paths, while random-path and More-QA were `1.734` and `1.468`. Thus the oracle is not failing to optimize its Stage-1 objective.
2. **Stage 2 has a distribution shift and is only partially adapted.** Stage 2 changes every method to answer-only prompts. Its first logged loss is `3.711` for oracle and `3.388` for all-paths, versus `0.654` for More-QA, whose Stage-1 prompt is exactly the same answer-only template. Oracle Stage-2 loss falls to `1.152`, but the cosine schedule has already reduced the learning rate to approximately `1.5e-5` by logged step 100, with only 103 actual optimizer steps. The loss is still descending, so undertraining/slow recovery can contribute, but it does not explain the very low Stage-1 loss by itself.

The strongest current explanation is **shortcut learning plus Stage-1-to-Stage-2 interference**. In the oracle supervision, the final answer entity is literally the tail node of the supplied gold path in all 716/716 rows. The model can therefore minimize the Stage-1 answer loss by copying the visible terminal entity from the evidence, rather than learning a graph-free mapping. The all-path condition also exposes the answer in every row (among four paths), while random-path exposes it only incidentally (16/716 rows). The low path-method Stage-1 losses are consequently not evidence that the answer has been internalized.

When Stage 2 removes all candidate evidence, the oracle adapter has to recover an answer-only mapping from a representation that was strongly optimized for evidence-conditioned copying. More-QA avoids this mismatch: it uses the same answer-only prompt template in both stages, receives about `18,006` supervised answer tokens, and starts Stage 2 at loss `0.654`; oracle receives about `10,065` Stage-1 supervised tokens and starts at `3.711`. This explains why More-QA wins without requiring a hardware or CUDA failure.

There is also a secondary comparability issue in the implementation: equal **input-token** totals do not imply equal supervised-token totals or equal optimizer updates. Stage 1 used 83, 93, 89, and 89 optimizer steps for all-paths, More-QA, oracle, and random-path respectively, and supervised-token totals ranged from `4,958` to `18,006`. Gradient accumulation thresholds are crossed at micro-batch boundaries, and the scheduler is based on estimated rather than actual optimizer steps. These differences are small enough that the token gate passes, but they can affect optimization and make the smoke unsuitable for a final mechanism claim.

## Most informative follow-up diagnostics

Before changing the research conclusion, run a cheap oracle-only diagnostic with saved checkpoints or equivalent instrumentation:

1. Evaluate graph-free validation/test **after Stage 1 and before Stage 2**. Near-zero performance would confirm that Stage-1 evidence loss is mostly copying and not internalization.
2. Evaluate after each Stage-2 epoch. If accuracy recovers with more epochs or a smaller Stage-2 learning rate, the main issue is catastrophic interference/undertraining; if it stays flat, the oracle signal is not transferring.
3. Log separate loss terms for answer-token positions that are textually present in the evidence versus absent from it.
4. Repeat a minimal ablation with the terminal answer masked in the training evidence, or with a held-out evidence-to-answer format, to test whether the oracle benefit survives removal of the copy shortcut.
5. Fix token-budget training to report/equalize supervised tokens and actual optimizer steps, and normalize each accumulated gradient by the actual accumulated token count.

These diagnostics should precede any learned prioritizer implementation. The present result is best described as: **the oracle path objective fit successfully, but the fitted evidence-conditioned behavior did not transfer through the current answer-only Stage-2 protocol; the smoke therefore cannot distinguish optimization interference from a genuinely unhelpful path teacher.**
