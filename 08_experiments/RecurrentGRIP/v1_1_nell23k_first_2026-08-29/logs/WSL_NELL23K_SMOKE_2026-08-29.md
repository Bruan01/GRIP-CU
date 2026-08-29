# WSL2 RTX 3090 NELL23K Smoke — 2026-08-29

## Scope and decision

This record covers the fixed NELL23K single-graph RecurrentGRIP v1.1 smoke on
WSL2 with an RTX 3090. It validates the end-to-end training, adapter save/reload,
CUDA evaluation, fixed-depth K sweep, output schema, and analysis path. It is a
smoke/integration result, not a quality claim: the run uses only one graph and a
small fixed QA set, and its graph-memory context is capped at 256 samples for the
45-minute smoke budget.

**Pilot recommendation: do not start the two-hour Pilot from this smoke alone.**
The execution chain is healthy, but K=2 substantially underperforms K=1 and the
smoke is too small to support a research-quality conclusion. A later Pilot should
be separately approved after reviewing this result and the documented smoke-only
context cap.

## Repository and source

- Repository: `GRIP-CU`
- Branch: `wsl/nell23k-smoke-20260829`
- Base commit before fixes: `a9d3f7794491eba504e91d78593599175a524aec`
- Original GRIP submodule: `2835b440bfd2c4de36f0380ae19bc1c22e6cb459` (clean and unchanged)
- `13_base_method/grip-exp/`: unchanged
- `v1_fixed_depth_2026-08-28/`: unchanged
- Smoke date: August 29, 2026 (Asia/Shanghai)

## Environment

Recorded in the run's `environment.txt`:

- WSL2 kernel: `6.6.87.2-microsoft-standard-WSL2`
- GPU: NVIDIA GeForce RTX 3090, 24 GiB (`24576 MiB`)
- NVIDIA driver: `581.80` (nvidia-smi 580.105.07), reported CUDA: `13.0`
- Python: `3.11.15`
- Torch: `2.7.1+cu126`, CUDA build `12.6`
- Transformers: `4.56.1`
- PEFT: `0.17.1`
- BF16: supported (`compute capability 8.6`)
- Evaluation placement: `cuda:0`
- Attention implementation: SDPA
- Model: `Qwen/Qwen2.5-0.5B-Instruct`

The smoke log reports CUDA availability as true throughout. No CPU evaluation
fallback, CUDA error, OOM, NaN, or failed adapter reload was observed.

## Validation gates

- `bash configs/check_mac_static.sh`: **PASS** — AST check (85 Python files),
  Bash/JSON checks, and the two NELL23K preparation tests.
- `bash configs/test_wsl3090.sh`: **PASS** — 21 unittest cases, `OK (skipped=1)`.
  The single skip is the test that expects CUDA to be unavailable; CUDA is
  available in this WSL runtime. The CUDA placement test passed.
- NELL23K semantic validation: **PASS** — split provenance, candidate answers,
  train-only graph, undirected train-graph BFS distances, nullable unknown hops,
  and source triple indices.
- Deterministic preparation: **PASS** — repeated seed `2026` generation produced
  identical JSON and stats hashes.

## Fixed smoke input

- Dataset: NELL23K
- Seed: `2026`
- Candidates per question: `10`
- Selected questions: train/validation/test = `64/32/64`
- Graph: `20,799` nodes, `24,321` train edges, `198` relations
- Validation structural distances: 1×3, 2×4, 3×18, 4×3, 5×2, unreachable×2
- Test structural distances: 1×7, 2×6, 3×24, 4×8, 5×6, 6×2, 7×2,
  9×1, 10×1, unreachable×7
- Source SHA256:
  - `train.txt`: `a55a45f9415df76cc75908775741fbddf0a7530d21430281efb238f006202f21`
  - `valid.txt`: `a40951319088c26e46dc6711b55b06c58be924aa7eea2f671d89c3ccec0bcabb`
  - `test.txt`: `9d48fa18ac395e4960521bcb1563883624e9d695fec6a4c44f2fb657194440c2`
  - `entity2text.json`: `86397d9291c0559bcb3a52f11454000cfc6259a26ad855e2feccd35c74444195`

## Run and artifacts

- RUN_ID: `wsl3090_nell23k_smoke_20260829_06`
- RUN_DIR:
  `results/runs/wsl3090_nell23k_smoke_20260829_06_nell23k_qwen05b_smoke`
- Wall time from first to last structured log event: approximately 87 seconds.
- Predictions: `384` = 96 evaluation questions × 2 K values × 2 adapter controls.
- K sweep: `K=1,2`
- Adapter controls: `correct`, `none`
- `shuffled` control: explicitly skipped because a single graph cannot provide a
  distinct shuffled graph (`requires_at_least_two_graphs`).
- Adapter manifest confirms executor layer index `12`, recurrent train depth `2`,
  and target modules `q_proj`, `k_proj`, `v_proj`.
- Peak per-sample CUDA allocated memory: approximately `1.933 GiB`.
- Mean per-sample latency:
  - correct adapter, K=1: `0.482 s`
  - correct adapter, K=2: `0.472 s`
  - no adapter, K=1: `0.593 s`
  - no adapter, K=2: `0.729 s`

Smoke-only setting:

```text
--max_context_samples 256
```

The input QA remains 64/32/64; this cap limits only graph-memory training
samples and was added to keep the integration smoke within its hard timeout. The
run log records `recurrent_context_samples_capped` with 256 selected samples.

## Accuracy and analysis

Generated files:

- `analysis/summary.json`
- `analysis/k_adapter_accuracy.csv`
- `analysis/split_k_adapter_accuracy.csv`
- `analysis/hop_k_accuracy.csv`

From `analysis/summary.json` (384 predictions):

| Adapter control | K | Correct | Accuracy |
|---|---:|---:|---:|
| correct | 1 | 29/96 | 30.21% |
| correct | 2 | 3/96 | 3.13% |
| none | 1 | 25/96 | 26.04% |
| none | 2 | 1/96 | 1.04% |

Additional summary values:

- Overall accuracy across controls and K: `15.10%`
- Correct-adapter accuracy across both K values: `16.67%`
- Known structural-hop predictions: `348`; unknown-hop predictions retained in
  accuracy: `36`
- Solved questions under correct adapter: `31/96`
- Best-K vs known true-hop Spearman: `0.0912` (descriptive only; not a decision
  statistic for this smoke)

The zero accuracy seen in the earlier `_05` run was traced to the parser treating
Qwen's untagged bracket-form output such as `[concept:...]` as an empty answer.
The parser now accepts both `<answer>...</answer>` and the documented bracket/plain
forms; `_06` was rerun with this fix and is the authoritative smoke result.

## Fixes included

1. Added `tiktoken` to project/setup dependencies so the complete unittest import
   path works.
2. Exported `PYTHONPATH` in smoke and analysis wrappers so top-level project
   modules resolve when scripts are launched directly.
3. Forwarded `attention_type` and `layer_idx` from the wrapped decoder layer to
   `FixedDepthRecurrentBlock` for Transformers 4.56/Qwen2 causal-mask metadata.
4. Added validated `max_context_samples` and smoke-only graph-memory capping,
   with an explicit runtime event.
5. Added parser coverage for bracket and tagged answers, and kept the adapter
   metadata forwarding test.

## Failure attempts retained for audit

- Initial setup was interrupted after most dependencies were installed; rerun used
  the existing virtual environment.
- Full unittest initially failed on missing `tiktoken`; dependency and lockfile
  were updated, then all 21 tests passed.
- Smoke `_01` initially failed because `PYTHONPATH` did not expose top-level
  `arguments`.
- Subsequent smoke attempt failed because the recurrent wrapper did not expose
  `attention_type` required by the installed Transformers/Qwen2 stack.
- Two attempts without the context cap were manually stopped because full NELL23K
  graph-memory context made the smoke exceed the intended budget.
- `_05` completed technically but had invalid zero scores due to the bracket-output
  parsing bug; it is retained and not used as the authoritative result.
- `_06` completed after the parser fix and was analyzed above.

No two-hour Pilot was started, no CLEGR data was downloaded, and no existing run
directory was overwritten.
