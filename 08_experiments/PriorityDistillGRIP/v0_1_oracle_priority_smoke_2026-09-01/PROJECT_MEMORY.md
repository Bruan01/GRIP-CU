# Project Memory — PriorityDistill-GRIP v0.1

## Frozen thesis

Graph-path priority is used as a **training-time teacher**, not a test-time retrieval component. The desired endpoint is ordinary parametric LoRA that answers without graph access.

## Why this version exists

StructuredLoRA v0.2 produced `PRELIMINARY_STOP`: ordered depth-prefix routing did not beat monolithic LoRA and did not improve 3/4-hop. The next idea therefore changes the source of supervision rather than further partitioning rank dimensions.

## Frozen mechanism question

> Under equal Stage-1 input-token budget, is a perfect gold path more useful for parametric internalization than additional QA, an unrelated legal path, or an unprioritized candidate set?

## Frozen controls

```text
answer_only
more_qa_equal_token
random_path_equal_token
all_paths_equal_token
oracle_priority_equal_token
```

Do not delete or rename these controls inside v0.1.

## Frozen data rules

- NELL23K first.
- Strict exact-hop 1–4 data only.
- Candidate construction is train-only.
- Three same-depth, different-answer distractors.
- No validation/test path may enter training evidence.
- Validation/test prompts contain no candidate evidence.

## Frozen decision rule

One seed is preliminary. Three seeds are required for final Go/Stop. Only `GO_LEARNED_PRIORITIZER` permits v0.2 semantic path scoring or path-wise preference learning.

## Interpretation discipline

- If oracle does not beat equal-token More-QA, the path-priority story is not supported.
- If random path matches oracle, the effect is generic context/token exposure.
- If all paths matches oracle, explicit prioritization is unnecessary.
- If only 1-hop rises, do not claim improved multi-hop reasoning.
- If runtime token totals diverge materially, report the mismatch before interpreting accuracy.
- If any Stage-1 prompt is truncated, stop the mechanism interpretation until context length or prompt construction is fixed.

## Environment memory

- macOS is for editing, deterministic data construction, unit tests, and static audit.
- WSL2 + RTX 3090 24GB is for PyTorch/Transformers runtime.
- The local macOS Python used on 2026-09-01 does not contain `torch`; this is expected and checked again in WSL preflight.

## 2026-09-01 WSL smoke result

Run `wsl3090_priority_distill_smoke_01` completed in conda `guardenv` on WSL2 Ubuntu with an RTX 3090. The local model snapshot used was `/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`; using this explicit path avoids model-alias/cache-resolution ambiguity. Runtime versions were Python 3.10.19, PyTorch 2.12.0+cu130, Transformers 4.57.3, and PEFT 0.17.1; `torch.cuda.is_available()` was true.

All five seed-42 methods completed. Test accuracy was: `answer_only=0.3141`, `more_qa_equal_token=0.3526`, `random_path_equal_token=0.3397`, `all_paths_equal_token=0.3397`, and `oracle_priority_equal_token=0.3141`. Oracle deep 3/4 accuracy was `0.4487`, below answer-only `0.4872`. Equal-token audit passed: path methods were within `0.02558%` relative Stage-1 input-token gap and had zero prompt truncation. Every run was graph-free at inference (`inference_graph_access=false`) and used CUDA.

The suite decision is `PRELIMINARY_STOP`: every mechanism comparison involving oracle failed, while only token-budget and no-truncation checks passed. Do not run the registered full seeds and do not implement `GO_LEARNED_PRIORITIZER`/v0.2 learned scoring from this smoke. Preserve the result as a negative preliminary result.

For future WSL execution, load conda explicitly before invoking scripts: `source /home/kieran/miniconda3/etc/profile.d/conda.sh && conda activate guardenv`. Pass the explicit local model path when a script's model resolver does not recognize the logical alias.
