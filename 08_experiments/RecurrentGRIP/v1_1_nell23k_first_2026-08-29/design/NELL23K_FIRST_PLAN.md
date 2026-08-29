# NELL23K-First Experiment Plan

## Decision

NELL23K 作为 RecurrentGRIP 的首个 WSL 集成和效果验证数据集；CLEGR 延后到机制
确认阶段。该顺序降低数据准备成本，同时先回答“机制实现是否能在 Original GRIP
基准上运行并产生收益”。

## Questions

1. recurrent executor 是否在不访问原图的情况下完成 NELL23K relation prediction？
2. correct adapter 是否优于 disabled adapter？
3. 在相同 adapter 下，K=1/2/3/4 是否产生稳定且可解释的性能差异？
4. 结构距离更大的实体对是否更受益于更大的 K？
5. 相同 LoRA rank、训练 QA 数和推理预算下，RecurrentGRIP 是否优于 Original GRIP？

## Protocol

- 图：仅 `train.txt` 三元组；
- recurrent QA train：从 `train.txt` 确定性采样；
- validation：从 `valid.txt` 确定性采样；
- test：从 `test.txt` 确定性采样；
- 候选关系：训练关系词表内 10-way candidate set；
- 结构距离：train graph 上无向 BFS；
- 不可达问题：保留于整体 accuracy，排除于 hop/K correlation；
- seed：2026；
- 推理：closed-book，prompt 中不提供图。

## Stages

### S0 — Data validation

- 检查答案一定出现在 candidates；
- 检查 train/validation/test 来源不混用；
- 检查固定 seed 输出完全一致；
- 汇总 train graph 节点、边、关系和 distance buckets。

### S1 — 0.5B smoke

- 64/32/64 QA；
- K_train=2；
- K_eval=1,2；
- correct/none；
- 仅判断运行链、输出模式和资源可行性，不形成论文结论。

### S2 — 0.5B pilot

- 512/128/512 QA；
- K_train=2；
- K_eval=1,2,3,4；
- correct/none；
- 最长两小时软限制、三小时 OS 硬停止。

### S3 — Fair baseline

同一模型、LoRA rank、target modules、训练 QA、epoch、seed 和生成设置下运行：

```text
Base LLM
Original GRIP
RecurrentGRIP K=1/2/3/4
RecurrentGRIP no-adapter
```

### S4 — Mechanism confirmation

只有 NELL23K 显示 recurrent depth 信号后，才运行 CLEGR 的精确 K↔hop 实验。
NELL23K 的 shortest-path bucket 是相关性诊断，不代替 CLEGR 的可控机制证据。

## Stop Rules

- smoke 未通过：不启动 pilot；
- correct adapter 与 none 无差异：先检查 adapter scope/load，不扩大模型；
- 所有 K 输出完全相同：检查 recurrent block 是否确实重复执行；
- OOM：先减少 QA/生成长度，不改变方法定义；
- 只有速度变化而无准确率或行为变化：不宣称递归推理机制成立。
