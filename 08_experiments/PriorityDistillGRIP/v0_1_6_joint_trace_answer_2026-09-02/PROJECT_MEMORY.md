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
7. **Candidate pools must deduplicate answer strings.** Multiple train paths may have the same terminal entity. If distractors are selected by row only, a nominal 4-way pool can silently contain only 2–3 unique answers and distort the random baseline. Enforce unique distractor answers before constrained decoding.
8. **A trie-constrained greedy decoder is not sequence-level candidate scoring.** The former improved stage2_epoch5 test from 29.49% free generation to 50.00% on a gold-plus-three-distractor pool; teacher-forced full-candidate scoring reached 63.46% on the same checkpoint. Keep these as separate diagnostics.

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

## Candidate-ranking and constrained-decoding diagnosis

After enforcing unique distractor answer strings, teacher-forced scoring of one gold answer plus three train distractors at the validation-selected stage2_epoch5 checkpoint gave candidate rank-1:

- graph-free: `99/156 = 63.46%`;
- masked gold-trace: `94/156 = 60.26%`;
- oracle-evidence: `94/156 = 60.26%`.

The new trie-constrained greedy decoder gave:

- validation: `73/152 = 48.03%`;
- test: `78/156 = 50.00%`.

The v0.1.5 direct answer-only baseline, evaluated with the same candidate construction and decoder, gave validation `84/152 = 55.26%` and test `87/156 = 55.77%`. Therefore the current joint checkpoint does not beat the direct baseline under this fair candidate-set diagnostic.

At stage2_epoch8 the corresponding constrained test score was `84/156 = 53.85%`. These are gold-plus-three-distractor candidate-set diagnostics, not full-vocabulary accuracy. The gap between 63.46% sequence-level teacher-forced ranking and 50.00% greedy constrained decoding shows that token-by-token greedy decoding remains suboptimal. Typical unconstrained errors still stop at a shared entity prefix, e.g. `concept_sportsleague` instead of `concept_sportsleague_nba`.

## Current conclusion

The v0.1.6 code and training run are valid after the corrected evaluation. The joint trace-answer loss decreases, and the model memorizes much of the training set. It also has non-random candidate-level answer preferences. However, the method does not yet demonstrate reliable graph-free compositional knowledge internalization: held-out free generation is only 29.49% under validation selection, joint trace+answer exact match is only 7.05% at that selected checkpoint, and performance is unstable across epochs.

Do **not** conclude that every LoRA + textual-trace method is impossible. The defensible conclusion is narrower: this v0.1.6 objective and output format do not provide a reliable bridge from supervised traces to held-out graph-free reasoning. The candidate-set decoder diagnostic is now complete. The first direct-baseline comparison is also complete: direct constrained test `87/156 = 55.77%` versus joint `78/156 = 50.00%`. The next experiment should implement a true sequence-level candidate decoder/beam search and use stricter answer/entity/relation-composition splits before adding a learned router or a full explicit graph module.

## Constrained-decoding artifacts

```text
priority_distill/constrained.py
scripts/evaluate_constrained_answers.py
results/runs/wsl3090_v016_joint_20260902_01/constrained_entity_stage2_epoch5/
results/runs/wsl3090_v016_joint_20260902_01/constrained_entity_stage2_epoch8/
```
## Generalization split audit (2026-09-03)

Added `scripts/analyze_generalization_splits.py` with strict prediction coverage checks and tests. It reports: answer seen/unseen in train, starting head seen/unseen, exact ordered relation composition seen/novel, all component relations individually seen, and the novel-composition subset split by relation coverage.

For the current constrained test predictions:

- direct: seen composition `30/67 = 44.78%`, novel composition `57/89 = 64.04%`;
- joint: seen composition `32/67 = 47.76%`, novel composition `46/89 = 51.69%`.

For current graph-free sequence-level ranking:

- direct: seen `19/67 = 28.36%`, novel `36/89 = 40.45%`;
- joint: seen `41/67 = 61.19%`, novel `59/89 = 66.29%`.

These are diagnostics on 4-way candidate pools, not full-vocabulary accuracy. The disagreement between constrained greedy and teacher-forced ranking is itself evidence that decoding must be measured separately from candidate preference.
## Fair deployment-prompt candidate diagnostic (2026-09-03)

The older `scripts/score_answer_candidates.py` uses a gold intermediate trace as a teacher-forced prefix. It is useful for separating candidate preference from output formatting, but its `graph_free` label is not a pure deployment condition. Do not use it alone as evidence of graph-free reasoning or as a fully fair direct-vs-joint comparison.

Added `scripts/score_deployment_candidates.py`, which scores the same gold-plus-three unique train distractors after the exact answer-only deployment prompt (`build_evaluation_prompt`) for both checkpoints. Test results: direct answer-only `81/156 = 51.92%` sequence rank-1 versus v0.1.6 joint `77/156 = 49.36%`; constrained greedy is direct `87/156 = 55.77%` versus joint `78/156 = 50.00%`. The paired deployment-prompt comparison has joint wins 17 and direct wins 21.

Therefore the current strongest claim is: the joint checkpoint has non-random candidate preference, but it does not beat the direct answer-only baseline under the same graph-free candidate prompt.


## v0.1.7 fair multi-seed follow-up (2026-09-03)

为判断 v0.1.6 joint 与 v0.1.5 direct 的差异是否只是单个 seed 的偶然现象，新增了公平 seed-sweep 接口：

- `configs/direct_answer_only_fair_20260903.json`
- `configs/graph_free_trace_joint_fair_20260903.json`
- `configs/run_fair_seed_sweep_wsl.sh`
- `scripts/aggregate_seed_results.py`

两种协议保持相同的 NELL23K exact-hop 数据、Qwen2.5-0.5B 本地快照、LoRA 配置、candidate pool 和 `max_new_tokens=80`；只改变 `training.seed`。`scripts/run_controlled.py --training-seed N` 只覆盖运行时 seed，不修改注册 JSON。每个 run 仍然只用 graph-free validation 选择 checkpoint，再读取 test；不能用 test 反选 epoch。

运行环境固定为 conda `guardenv`，模型快照固定为：

```text
/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

候选诊断放在 `candidate_diagnostics_<selected_checkpoint>/`，只提交 JSON/JSONL 指标和预测，不提交 `adapter_model.pt`、`*.safetensors` 或模型缓存。
