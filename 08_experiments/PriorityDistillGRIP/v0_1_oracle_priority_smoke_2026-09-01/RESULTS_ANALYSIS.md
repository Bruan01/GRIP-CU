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
