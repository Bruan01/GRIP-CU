# Project Memory — PriorityDistill-GRIP v0.1.2 Controlled Protocol

## Purpose

This version answers one causal question: does seeing a gold path in Stage 1 improve later graph-free answering, compared with spending the same token budget on answer-only training from the start?

## Protocols

- `direct_answer_only`: every segment trains `question -> answer`; Stage 1 is skipped.
- `oracle_two_stage`: first segment trains `question + gold path -> answer`, followed by eight answer-only segments.

Both protocols use the same model, LoRA modules, seed, splits, token accumulation, optimizer, validation/test code, and nominal token budget. The first segment has the token budget of the `all_paths_equal_token` reference. Direct replaces that segment with answer-only rows; oracle uses gold-path rows. The remaining eight segments are answer-only and have the same target as one answer-only train pass.

## Selection rule

Select the checkpoint using only `graph_free_validation` accuracy. Read the held-out test accuracy from that selected checkpoint afterward. Never choose an epoch using test accuracy.

## Interpretation

- Direct > oracle: the oracle path stage is unnecessary or harmful.
- Oracle > direct across seeds: evidence exposure likely adds value.
- Similar results: the first stage has no measurable benefit under this budget.
- An oracle gain without anti-copy gains is not proof of knowledge internalization.

## Environment memory

Run in WSL2 with conda environment `guardenv`, RTX 3090 24GB. The local model is:

`/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`

Adapter checkpoints under `results/runs/**/checkpoints/**/adapter_model.pt` remain ignored and must not be committed. Metrics, predictions, logs, reports, and audits are safe to commit.

## Seed-42 controlled result (2026-09-01)

Runs `wsl3090_control_direct_20260901_01` and `wsl3090_control_oracle2_20260901_01` completed with nearly identical token budgets (638,830 vs 638,785 input tokens). Validation-only selection chose `stage2_epoch5` for both. Direct answer-only scored 0.3224 validation and 0.2692 test (42/156); oracle two-stage scored 0.3026 validation and 0.3269 test (51/156), a +5.77 point single-seed held-out difference. This is promising but inconclusive: direct had the better validation score, both curves are non-monotonic, test n=156, and only seed 42 exists. Epoch-7 test peaks (direct 0.3462, oracle 0.3526) are test-selected observations and must not be reported as official scores. Run seeds 43 and 44 with the same configs before any learned prioritizer.
