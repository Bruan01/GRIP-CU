# PriorityDistill-GRIP v0.1.4 Results Analysis

- **分析日期**：2026-09-02（Asia/Shanghai）
- **环境**：WSL2 + RTX 3090 24GB + conda `guardenv`
- **模型**：本地 `Qwen/Qwen2.5-0.5B-Instruct`
- **数据**：NELL23K exact-hop，train/validation/test = 716/152/156，深度 1/2/3/4 均衡
- **checkpoint 选择**：只按 graph-free validation 选择，test 只在选择后读取
- **目标**：判断“候选路径选择/anti-copy 训练”是否能迁移为没有图证据时的答案能力

## 1. 主要结果

作为对照，使用 v0.1.3 的同 seed direct baseline：

| protocol | selected checkpoint | graph-free validation | graph-free test | 相对 direct test |
|---|---:|---:|---:|---:|
| `direct_answer_only`（v0.1.3 对照） | `stage2_epoch4` | 32.24% | 34.62%（54/156） | — |
| `candidate_index_two_stage` | `stage2_epoch6` | 30.92% | 30.77%（48/156） | -3.85 pp |
| `candidate_index_anti_copy` | `stage2_epoch6` | 30.26% | 32.69%（51/156） | -1.93 pp |
| `candidate_index_anti_copy_replay` | `stage2_epoch7` | 30.26% | **35.26%（55/156）** | **+0.64 pp** |

`candidate_index_anti_copy_replay` 比 direct 多答对 1 个 test 样本，因此不能称为可靠提升。validation 没有提升，反而比 direct 低 1.98 个百分点；这说明 35.26% 更像单次 seed/小测试集波动，而不是已经证明的效果。

## 2. 这不是简单的“代码没跑好”或“完全没收敛”

训练过程本身是正常的：

- anti-copy replay 的 Stage 1 loss 从约 `0.87` 降到 `0.23`；
- Stage 2 loss 从约 `3.46` 降到 `0.17`；
- GPU、模型、LoRA、数据和 checkpoint 都正常完成；
- graph-free accuracy 在 Stage 2 从 0% 逐步恢复到约 30%--35%。

但“训练 loss 下降”只说明模型越来越会完成**训练时的格式和训练样本**，不等于它学会了测试时要求的“没图也能组合推理”。

更准确的说法是：

> 训练目标在优化；真正关心的 graph-free 泛化没有稳定收敛，而且随着 epoch 会上下波动。

例如 anti-copy replay 的 graph-free test 曲线为：

```text
6.41% -> 17.31% -> 27.56% -> 25.00% -> 28.85% -> 26.92% -> 35.26% -> 35.26%
```

所以继续无条件增加 epoch 不是解决方案。

## 3. 机制诊断：模型学会了“选路”，但没有稳定学会“用路得到答案”

在候选路径条件下，路径选择准确率很高：

- ordinary candidate selection：test 约 `98.1%`；
- terminal-masked selection：test 约 `98.7%`（no replay 选中 checkpoint）；
- anti-copy replay 选中 checkpoint 的 terminal-masked selection：`147/156 = 94.23%`。

但是在 terminal 被遮住后，最终答案准确率明显低得多：

- no replay：`65/156 = 41.67%`；
- replay：`88/156 = 56.41%`。

这说明模型能回答：

> “四条候选里哪一条更像正确路径？”

但它还不能稳定回答：

> “我已经找到这条路径后，怎样把路径上的关系一步一步算到最终实体？”

因此，当前实验已经支持一个**较弱但重要的结论**：候选选择不是随机的，也不完全是复制路径末节点；但它还没有证明路径信息被内化成了 graph-free 的组合推理能力。

## 4. replay 说明了什么

anti-copy replay 的 graph-free test 为 `35.26%`，高于 no-replay 的 `32.69%`，增加 `2.57` 个百分点；但它仍然只比 direct 高 `0.64` 个百分点。

replay 的作用更像是：

- Stage 1 学到的“看证据”能力，在 Stage 2 中遗忘得少一些；
- 但保留“看证据”的能力，并没有自动变成“没证据也能推理”。

所以 replay 暂时只能证明存在**分布切换/遗忘问题**，不能证明 PriorityDistill-GRIP 的核心 idea 已经有效。

## 5. 为什么 graph-free 结果一直在 30% 左右

当前数据划分本身可能比想象中更难：

- validation/test 中完整的“起点 + 关系链”在 train 中没有重复；
- 实体是 `concept_xxx` 这种几乎没有自然语言语义的符号 ID；
- 因此测试要求模型从训练中没有见过的完整组合恢复答案；
- 一个 0.5B 模型即使 loss 很低，也可能只是在记忆训练组合，而不是学会可组合的关系运算。

这意味着现在看到的瓶颈不一定只是 LoRA 学习率、epoch 或 rank；也可能是 benchmark 设计让 graph-free transfer 过于困难，甚至无法区分“方法没学会”和“任务没有足够可迁移信息”。

## 6. 当前结论

### 可以继续做，但不能按原路线继续堆训练

目前不建议直接实现 learned router/scorer，也不建议继续只调：

- 更多 epoch；
- 更大的 LoRA rank；
- 更高/更低学习率；
- 更多普通 replay。

更准确的结论是：

> **PriorityDistill-GRIP 的“学习候选路径优先级”部分有实验依据；但“优先级训练能够改善无图组合推理”目前没有被证明。idea 尚未被否定，但当前两阶段实现没有形成有效的 bridge。**

## 7. 下一步（按优先级）

### A. 先修正/拆分泛化测试

建立三个难度明确的 split：

1. **seen-composition**：训练中见过相同关系组合，只换实体；
2. **novel-composition**：单跳边/关系在 train 中出现，但关系链组合在 test 中新；
3. **fully-unseen-combination**：保留当前最难设置，作为压力测试。

这样可以区分：模型不会推理，还是测试组合根本没有可迁移的训练支撑。

### B. 实现 bridge supervision，而不是继续纯两阶段 replay

建议 v0.1.5 使用 graph-free 输入，训练目标同时包含中间实体和最终答案，例如：

```text
Trace: head -> intermediate_1 -> intermediate_2 -> <MASKED_TERMINAL>
Answer: final_entity
```

关键点：

- 输入不包含 candidate evidence；
- trace 的最后实体必须遮住，避免把答案直接复制进 trace；
- `Answer:` 单独监督最终实体；
- depth=1 也必须生成合法格式；
- 评估时仍使用普通 graph-free 问题，不给图和候选路径。

### C. 公平对照

在相同 input-token budget、相同 LoRA、相同训练步数下比较：

- direct answer-only；
- graph-free trace；
- candidate selection + graph-free trace bridge。

先跑一个 seed 做机制检查，再跑至少 3 个 seed 做结论。checkpoint 仍然只能按 validation 选择。

### D. 继续/停止标准

只有在同一数据 split 和 token budget 下，bridge 方法同时满足以下条件，才值得进入 learned prioritizer：

- graph-free validation 和 test 都稳定高于 direct；
- deep 3/4-hop 不只是偶然提升；
- terminal-masked 的 selection、answer、joint 三者都有合理改善；
- 至少多个 seed 的方向一致。

如果 bridge 在修正后的 split 上仍然约 30%--35%，而 direct 同样水平，就应停止继续堆叠两阶段设计，转向数据划分/任务定义或显式图推理模块。
