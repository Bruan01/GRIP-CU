# Project Memory — PriorityDistill-GRIP v0.1.6

## Environment (must be reproduced)

- WSL2 + NVIDIA RTX 3090 24 GB.
- Conda environment: `guardenv`.
- Use the environment's interpreter explicitly when running scripts:
  `/home/kieran/miniconda3/envs/guardenv/bin/python`.
- Verified runtime: PyTorch `2.12.0+cu130`, Transformers `4.57.3`.
- Local base model snapshot:
  `/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`.
- Checkpoints and model caches are local/ignored. Commit code, configs, logs, predictions and reports, but never `adapter_model.pt` or model-cache files.
- Do not modify `13_base_method/grip-exp/`.

## Why this version exists

Earlier candidate-path experiments could be solved by copying the last node of a visible path. v0.1.6 tests a graph-free bridge directly: the model is trained to emit both a trace and a separate answer.

Training target:

```text
Trace: head -> intermediate_nodes -> <MASKED_TERMINAL>
Answer: final_entity
```

The terminal is masked in the trace so the answer cannot be copied from the last trace node. The graph-free evaluation prompt contains no graph, candidate path or evidence.

## Protocol and fairness controls

- Protocol: `graph_free_trace_joint`.
- NELL23K exact-hop split: train/validation/test = `716/152/156`; depths 1/2/3/4 are balanced within each split.
- Qwen2.5-0.5B-Instruct, LoRA rank 8, alpha 16, target modules `down_proj`, `up_proj`, `gate_proj`.
- Stage 1 and each Stage 2 epoch use tokenized-input budgets; actual input tokens, supervised tokens and optimizer steps are recorded.
- Checkpoint selection is validation-only: maximize graph-free validation answer accuracy, then read test once for the selected checkpoint.
- `runtime.disable_progress=false`; training/evaluation loops retain tqdm progress bars.

## Important implementation lessons

1. **Use the right interpreter.** A bare `python`/`python3` may resolve to base Python 3.13, which does not have the `guardenv` PyTorch installation. Prefer:
   ```bash
   PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
   $PYTHON ...
   ```
   or `conda run -n guardenv python ...`.
2. **Keep CLI names synchronized.** The parser accepts `--local_files_only`, not `--local-files-only`.
3. **Keep model alias resolution separate from path resolution.** The recurrent model wrapper expects a logical alias unless the experiment explicitly supports a resolved local path. Do not pass an arbitrary snapshot path into an alias-only `_model_id()` function.
4. **Generation budget is part of the experiment.** The old `max_new_tokens=24` truncated most structured traces (train: 605/716, validation: 129/152, test: 130/156 exceeded 24 tokens). All structured-output evaluations must use a budget large enough for the target; the corrected audit uses `max_new_tokens=80`.
5. **Every custom logits processor must match the installed Transformers API.** The old constrained decoder failed because `CandidateTrieLogitsProcessor` had a call signature incompatible with Transformers 4.57.3. Unit tests and an actual constrained-decoder smoke test should run in `guardenv` before a GPU experiment.
6. **Do not infer internalization from loss alone.** A falling teacher-forced loss can coexist with memorization, malformed/free-generation outputs, and poor compositional generalization. Always report train-vs-held-out, candidate ranking, trace, answer and joint metrics.

## Completed run

Run ID: `wsl3090_v016_joint_20260902_01`.

Original evaluation used `max_new_tokens=24` and gave graph-free test answer `3/156 = 1.92%`; this number is not a fair final score for a structured trace task because the generation budget was too short.

Corrected post-hoc evaluation:

```text
configs/graph_free_trace_joint_corrected_eval.json
results/runs/wsl3090_v016_joint_20260902_01/corrected_eval/
```

The corrected evaluation uses `max_new_tokens=80` and does not retrain.

## Corrected checkpoint curve

| checkpoint | graph-free validation | graph-free test | trace test | joint test |
|---|---:|---:|---:|---:|
| stage1 | 16.45% | 22.44% | 29.49% | 5.13% |
| stage2_epoch1 | 22.37% | 23.08% | 29.49% | 4.49% |
| stage2_epoch2 | 20.39% | 28.21% | 28.85% | 5.77% |
| stage2_epoch3 | 25.66% | 28.85% | 30.77% | 7.05% |
| stage2_epoch4 | 21.05% | 32.05% | 31.41% | 8.97% |
| **stage2_epoch5 (validation-selected)** | **27.63% (42/152)** | **29.49% (46/156)** | **31.41%** | **7.05%** |
| stage2_epoch6 | 26.32% | 30.77% | 36.54% | 11.54% |
| stage2_epoch7 | 25.66% | 35.90% | 33.97% | 10.90% |
| stage2_epoch8 | 26.97% | 39.74% | 40.38% | 15.38% |

The official result under the pre-declared validation-only rule is stage2_epoch5 and test `46/156 = 29.49%`. Stage2_epoch8's `39.74%` is a useful curve diagnostic, not a selectable test result, because its validation score is lower than epoch5.

## Corrected checkpoint audit (stage2_epoch8, max_new_tokens=80)

Output:

```text
results/runs/wsl3090_v016_joint_20260902_01/checkpoint_audit_corrected/checkpoint_audit.json
```

| condition | split | answer accuracy |
|---|---|---:|
| graph-free | train | **572/716 = 79.89%** |
| graph-free | test (curve) | **62/156 = 39.74%** |
| gold trace, terminal masked | validation | 54/152 = 35.53% |
| gold trace, terminal masked | test | 57/156 = 36.54% |
| gold trace, terminal visible (oracle-only) | validation | **78/152 = 51.32%** |
| gold trace, terminal visible (oracle-only) | test | **82/156 = 52.56%** |

The train/held-out gap is large: 79.89% train versus 39.74% graph-free test at the same checkpoint. This is evidence of memorization/weak compositional generalization, not simply “the optimizer failed to converge.” Revealing the terminal adds about 16 percentage points on test (36.54% -> 52.56%), so terminal recovery and structured output are important bottlenecks. `gold_trace_unmasked` is an oracle-only sanity check and must never be presented as graph-free performance; its trace/joint metrics are not comparable because the expected target format is intentionally different.

## Candidate-ranking diagnosis

Teacher-forced scoring of one gold answer plus three train distractors gave candidate rank-1:

- graph-free: `101/156 = 64.74%`;
- masked gold-trace: `98/156 = 62.82%`;
- oracle-evidence: `88/156 = 56.41%`.

Therefore free-generation exact match underestimates some of the model's relative answer signal. Typical errors stop at a shared entity prefix, e.g. `concept_sportsleague` instead of `concept_sportsleague_nba`. Candidate scoring/constrained entity decoding is a more appropriate next diagnostic than repeatedly increasing epochs.

## Current conclusion

The v0.1.6 code and training run are valid after the corrected evaluation. The joint trace-answer loss decreases, and the model memorizes much of the training set. It also has non-random candidate-level answer preferences. However, the method does not yet demonstrate reliable graph-free compositional knowledge internalization: held-out free generation is only 29.49% under validation selection, joint trace+answer exact match is only 7.05% at that selected checkpoint, and performance is unstable across epochs.

Do **not** conclude that every LoRA + textual-trace method is impossible. The defensible conclusion is narrower: this v0.1.6 objective and output format do not provide a reliable bridge from supervised traces to held-out graph-free reasoning. The next experiment should use a candidate-set answer scorer/constrained entity decoder and a small seen-composition versus novel-composition split before adding a learned router or a full explicit graph module.
