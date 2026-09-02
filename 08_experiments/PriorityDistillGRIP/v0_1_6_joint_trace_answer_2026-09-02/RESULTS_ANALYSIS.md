# PriorityDistill-GRIP v0.1.6 Results Analysis

- **分析日期**：2026-09-02（Asia/Shanghai）
- **环境**：WSL2 + RTX 3090 24GB + conda `guardenv`
- **模型**：本地 `Qwen/Qwen2.5-0.5B-Instruct`
- **数据**：NELL23K exact-hop，train/validation/test = 716/152/156
- **实验目录**：`v0_1_6_joint_trace_answer_2026-09-02`
- **运行目录**：`results/runs/wsl3090_v016_joint_20260902_01`

## 一句话结论

这次不是“完全没收敛”。模型在训练集上已经学会了不少东西，但主要表现为**记忆训练样本 + 学会输出格式**；到了没有图证据的新组合上，答案泛化仍然不稳定。更大的生成长度修复了一个严重的评估问题，但没有把方法变成可靠的组合推理器。

## 1. 先区分两个实验结果

原始运行的 `max_new_tokens=24` 对结构化 trace 输出太短：

- train 中 605/716 条目标超过 24 token；
- validation 中 129/152 条超过 24 token；
- test 中 130/156 条超过 24 token。

所以原始 graph-free test `3/156 = 1.92%` 不能直接当作最终方法能力。它混合了真实推理错误和“输出还没写完就被截断”两种错误。

随后使用同一批 checkpoint、同一模型和同一数据，只把生成预算改为 `max_new_tokens=80`，重新做了 corrected post-hoc evaluation；没有重新训练。

## 2. corrected checkpoint curve

| checkpoint | validation answer | test answer | test trace | test joint |
|---|---:|---:|---:|---:|
| stage1 | 16.45% | 22.44% | 29.49% | 5.13% |
| stage2_epoch1 | 22.37% | 23.08% | 29.49% | 4.49% |
| stage2_epoch2 | 20.39% | 28.21% | 28.85% | 5.77% |
| stage2_epoch3 | 25.66% | 28.85% | 30.77% | 7.05% |
| stage2_epoch4 | 21.05% | 32.05% | 31.41% | 8.97% |
| **stage2_epoch5** | **27.63%** | **29.49%** | **31.41%** | **7.05%** |
| stage2_epoch6 | 26.32% | 30.77% | 36.54% | 11.54% |
| stage2_epoch7 | 25.66% | 35.90% | 33.97% | 10.90% |
| stage2_epoch8 | 26.97% | 39.74% | 40.38% | 15.38% |

按照预先规定的“只看 validation 选 checkpoint”规则，正式 checkpoint 是 `stage2_epoch5`，所以正式 graph-free test 是 **46/156 = 29.49%**。`stage2_epoch8` 的 39.74% 只能作为训练曲线诊断，不能因为 test 高就倒过来选它；它的 validation 只有 26.97%，低于 epoch5 的 27.63%。

这个曲线告诉我们两点：

1. 生成长度修复很重要：从原始 1.92% 到 corrected curve 的 20%--40% 区间，说明 24 token 确实严重污染了评估。
2. 训练不是稳定单调变好：validation 在 epoch 5 达到峰值后波动，test 却在 epoch 8 继续升高。152/156 个样本的测试规模较小，不能用最后 test 峰值替代 validation 选择。

## 3. stage2_epoch8 checkpoint audit：训练集和“给证据”对照

使用 corrected `max_new_tokens=80` 对 `stage2_epoch8` 做了 post-hoc audit：

| 条件 | 数据 | answer accuracy |
|---|---|---:|
| graph-free | train | **572/716 = 79.89%** |
| graph-free | test（curve） | **62/156 = 39.74%** |
| gold trace，terminal masked | validation | 54/152 = 35.53% |
| gold trace，terminal masked | test | 57/156 = 36.54% |
| gold trace，terminal visible | validation | 78/152 = 51.32% |
| gold trace，terminal visible | test | 82/156 = 52.56% |

说明：audit 额外跑了 train 和 gold-trace 条件；graph-free validation/test 采用 corrected checkpoint curve 的统一输出。因此 stage2_epoch8 的 graph-free test 正式记为 **62/156 = 39.74%**。

训练集远高于测试集，说明有明显的记忆/组合泛化鸿沟。terminal 可见时比 terminal masked 高约 16 个百分点，说明实体终点恢复和结构化生成仍是主要瓶颈之一。`gold_trace_unmasked` 是 oracle-only sanity check，不是 graph-free 结果。

## 4. 候选打分诊断

之前的 teacher-forced candidate ranking 不要求模型自由生成完整实体，而是比较 gold answer 与三个 train distractor 的条件 log-likelihood：

- graph-free rank-1：101/156 = 64.74%；
- masked gold-trace rank-1：98/156 = 62.82%；
- oracle-evidence rank-1：88/156 = 56.41%。

这说明模型不是完全没有答案信号。自由生成 exact match 很低的部分原因，是模型经常只生成实体公共前缀，例如：

```text
预测：concept_sportsleague
真实：concept_sportsleague_nba
```

所以现在至少要把两个问题分开：

1. 模型是否在候选集合中偏好正确实体？——有一定能力；
2. 模型能否从开放词表中完整生成正确实体并同时满足 trace 格式？——目前较弱。

## 5. 回答“是不是没收敛”

不能只回答“没收敛”。训练 loss 从约 0.93 降到约 0.06，说明 teacher-forced 训练目标确实被优化；corrected train answer 79.89% 也说明模型在训练分布上已经学到很多。

更准确的判断是：

- **优化层面**：收敛到一个能拟合训练目标的解；
- **泛化层面**：没有稳定学会新实体组合上的图推理；
- **生成层面**：结构化目标较长，且实体 ID 共享前缀，容易输出不完整字符串；
- **选择层面**：validation 峰值与 test 峰值不一致，说明单次小 split 上的 checkpoint 选择有波动。

因此继续无条件增加 epoch、LoRA rank 或学习率，不是当前最有信息量的动作。

## 6. 当前对 idea 的判断

可以继续，但要缩小论断：

> v0.1.6 证明了文本 trace 联合监督能够让模型学会一部分 trace/answer 行为，也保留了非随机的候选答案信号；但它没有证明仅靠这种 LoRA + 文本 trace 目标就能稳定完成 graph-free 的组合知识内化。

不能扩大成“所有 LoRA + trace 都不行”，也不能仅凭 29%--40% 的自由生成准确率就断言必须立即换成完整显式图模块。当前更像是三个瓶颈叠加：

1. 训练/测试的组合泛化鸿沟；
2. 开放词表实体 ID 的完整生成问题；
3. trace 和 final answer 的联合格式过于脆弱。

## 7. 下一步

按信息增益排序：

1. **先做 candidate-set scorer/constrained entity decoder**：在固定候选集合中选答案，避免公共前缀和开放词表生成误差。
2. **拆分 seen-composition / novel-composition**：先验证模型是否至少能在“已见关系组合、换实体”上工作，再看全新组合。
3. **保留 answer-only direct baseline，并统一 `max_new_tokens=80` 的评估预算**。
4. **只在 bridge/scorer 有稳定收益后**，再实现 learned router 或显式神经关系组合器。
5. 先跑一个 seed 做机制检查；若方向成立，再做至少 3 个 seed。

## 结果文件

- corrected curve：`results/runs/wsl3090_v016_joint_20260902_01/corrected_eval/corrected_checkpoint_curve.json`
- corrected selected metrics：`results/runs/wsl3090_v016_joint_20260902_01/corrected_eval/corrected_selected_checkpoint_metrics.json`
- corrected audit：`results/runs/wsl3090_v016_joint_20260902_01/checkpoint_audit_corrected/checkpoint_audit.json`
- candidate ranking：`results/runs/wsl3090_v016_joint_20260902_01/candidate_ranking_audit/`
