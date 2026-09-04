# PriorityDistill-GRIP v0.1.6 — Joint Graph-Free Trace + Answer

- **日期**：2026-09-02
- **状态**：`COMPLETED_GPU_RUN_CORRECTED_AUDIT_AND_CONSTRAINED_DECODING`
- **环境**：WSL2 + RTX 3090 24GB + conda `guardenv`

## 目的

验证在没有图证据时，模型是否能同时生成中间推理 trace 和最终实体答案，并检查“trace 联合监督”是否比单纯 answer-only 更接近组合知识内化。

## 训练目标

```text
Trace: head -> intermediate_nodes -> <MASKED_TERMINAL>
Answer: final_entity
```

trace 的 terminal 被遮住，避免模型直接从 trace 末节点复制答案；`Answer:` 单独监督 final entity。graph-free 测试时不给任何 graph、candidate path 或 evidence。

## 数据和配置

- NELL23K exact-hop：train/validation/test = `716/152/156`。
- 深度 1/2/3/4 均衡。
- Qwen2.5-0.5B-Instruct 本地快照：
  `/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- LoRA：rank 8，alpha 16，target modules `down_proj/up_proj/gate_proj`。
- 评估时使用 corrected `max_new_tokens=80`；原始 24 token 版本会截断多数 trace。
- 进度条保留：`runtime.disable_progress=false`。

## 环境命令

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/PriorityDistillGRIP/v0_1_6_joint_trace_answer_2026-09-02
PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
$PYTHON -m unittest discover -s tests -p 'test_*.py' -v
$PYTHON scripts/runtime_self_test.py
$PYTHON scripts/validate_setup.py --config configs/graph_free_trace_joint.json
```

不要依赖 bare `python3`；它可能指向没有 torch 的 base Python。

## 已完成结果

原始 `max_new_tokens=24` 的 graph-free test 是 `3/156 = 1.92%`，但该评估被输出截断污染。使用同一批 checkpoint、只将生成长度改为 80 后：

| checkpoint | validation answer | test answer | test trace | test joint |
|---|---:|---:|---:|---:|
| stage1 | 16.45% | 22.44% | 29.49% | 5.13% |
| stage2_epoch1 | 22.37% | 23.08% | 29.49% | 4.49% |
| stage2_epoch2 | 20.39% | 28.21% | 28.85% | 5.77% |
| stage2_epoch3 | 25.66% | 28.85% | 30.77% | 7.05% |
| stage2_epoch4 | 21.05% | 32.05% | 31.41% | 8.97% |
| **stage2_epoch5（validation 选择）** | **27.63%** | **29.49%** | **31.41%** | **7.05%** |
| stage2_epoch6 | 26.32% | 30.77% | 36.54% | 11.54% |
| stage2_epoch7 | 25.66% | 35.90% | 33.97% | 10.90% |
| stage2_epoch8 | 26.97% | 39.74% | 40.38% | 15.38% |

正式结果遵守 validation-only checkpoint selection，因此报告 stage2_epoch5 的 test `46/156 = 29.49%`。epoch8 的 test 只作为曲线诊断。

## 诊断结论

- stage2_epoch8 corrected graph-free train answer：`572/716 = 79.89%`；
- 同 checkpoint corrected graph-free test：`62/156 = 39.74%`；
- gold trace + masked terminal test：`57/156 = 36.54%`；
- gold trace + visible terminal（oracle-only）test：`82/156 = 52.56%`；
- candidate teacher-forced rank-1：graph-free `64.74%`。

结论是：训练目标确实被优化，模型也有一定候选答案信号，但存在明显训练集记忆、组合泛化和实体完整生成问题。当前版本没有证明仅靠 LoRA + 文本 trace 就能稳定实现 graph-free 知识内化，也不能把结论扩大到所有类似方法。

## 结果路径

- `results/runs/wsl3090_v016_joint_20260902_01/corrected_eval/`
- `results/runs/wsl3090_v016_joint_20260902_01/checkpoint_audit_corrected/`
- `results/runs/wsl3090_v016_joint_20260902_01/candidate_ranking_audit/`
- 详细分析：`RESULTS_ANALYSIS.md`
- 环境与长期经验：`PROJECT_MEMORY.md`

## Candidate-set 受限解码（已完成）

新增加了 `scripts/evaluate_constrained_answers.py` 和 `priority_distill/constrained.py`。每题使用 gold + 3 个答案去重后的同深度训练 distractors，并限制生成只能落在候选实体集合中。重新评估结果：

| checkpoint | validation | test |
|---|---:|---:|
| stage2_epoch5（正式选择） | 73/152 = 48.03% | 78/156 = 50.00% |
| stage2_epoch8（诊断） | 69/152 = 45.39% | 84/156 = 53.85% |

这是候选集合诊断，不是完整实体词表准确率。旧的 `score_answer_candidates.py` 会把 gold intermediate trace 作为 teacher-forced 前缀，因此它不是纯 graph-free 部署评估。新增的 `score_deployment_candidates.py` 使用 direct/joint 完全相同的 answer-only deployment prompt，sequence-level rank-1 为 direct `81/156 = 51.92%`、joint `77/156 = 49.36%`。公平 direct answer-only 对照的 constrained test 为 `87/156 = 55.77%`，高于 joint 的 `78/156 = 50.00%`，所以当前 joint 尚未证明有增益。

## Fair multi-seed follow-up（2026-09-03）

已完成 seed 43/44 的 direct answer-only 与 graph-free trace + answer 公平训练。validation-controlled test 结果：direct **31.73% ± 2.27 pp**，joint **33.97% ± 0.91 pp**（两 seed，样本标准差）；validation 均值分别为 **32.89% ± 0.93 pp** 与 **32.57% ± 2.33 pp**。joint 的 test 均值高 2.24 个百分点，但尚未达到 three-seed 稳定增益 gate。汇总文件：`results/fair_seed_sweep_20260903.json`。

seed 43 的候选集诊断已完成；seed 44 当前只有核心训练结果，尚未生成 candidate diagnostics，因此不能把 candidate rank-1 / constrained greedy 作为完整多 seed 结论。详见 `RESULTS_ANALYSIS.md` 第 12 节。

## 下一步

1. 实现真正的 sequence-level candidate decoder / beam search；
2. 用 `scripts/analyze_generalization_splits.py` 报告 answer/entity/relation composition 的严格泛化切分；
3. 在同一候选池上比较 direct 与 joint，重点看 novel composition；
4. 只有在解码修复后 joint 仍不超过 direct，才考虑 learned router 或显式神经关系组合器。

不要只继续增加 epoch、rank 或学习率。
