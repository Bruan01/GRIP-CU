# Intake audit：结构反事实对比学习论文

日期：2026-10-23

## 1. 研究锚点

用户已明确将论文主线改为“结构反事实对比学习”。因此本文不再把 ErrorMemory-9
作为主方法；它只保留为可选辅助对照。当前 incumbent 是原始 GRIP/B1，论文主方法
是 `Structure-Counterfactual-9 + InfoNCE`。

## 2. 证据分级

| 对象 | 当前状态 | 可用于正文的范围 |
|---|---|---|
| 原始 GRIP/B1 代码 | 已存在于 `13_base_method/grip-exp` | 可作为基线定义；正式数字仍需按同一协议复核 |
| SC-CL sampler | 已实现并有单元测试 | 可写方法、契约、审计门；不能写性能提升 |
| train-only 过滤 | 代码和 manifest 校验已实现 | 可写数据泄漏防护设计 |
| NELL23K / FB15K237-2 数据资产 | 本地存在并已盘点规模 | 可写数据设置；运行前仍需 hash/版本登记 |
| NELL23K SC-CL seed=2026 | 已完成训练、bank、wiring 和 full candidate rerank | 可作为单种子实测；不能外推五种子或完整基线 |
| FB15K237-2 历史 B1/InfoNCE-i0 | 10,000 条候选评分可复核 | 只能作辅助历史证据，不等同于当前 SC-CL |
| FB15K237-2 当前 SC-CL | 未发现正式 bank/training run | 仍是主实验阻塞项 |
| 旧 RCCGRIP 结果 | 与当前方法不同、不可直接合并 | 只作为历史参考，不进入本文主表 |

## 3. 已核对的实现事实

1. SC-CL 只改变原始 GRIP 的候选关系学习阶段。
2. Stage 1 graph-context 训练、prompt、生成损失和 LoRA 路径保持不变。
3. 候选关系来自 train relation vocabulary，过滤当前实体对的 train-known relations。
4. 关系画像来自 train graph；B1 score violation 也只来自 train split 的冻结分数。
5. bank 冻结后训练不在线重采样，验证/测试不进入 bank。
6. 默认 `K=9`、`rho=1/3`、`alpha=1`、`beta=1`、`tau=0.25`。

## 4. 风险与边界

- “反事实”不是开放世界真值判断；它是一个经过 train-known 过滤的候选替换操作。
- 关系画像相近并不保证候选关系是语义上可混淆的；这一点需要消融和错误分析验证。
- NELL 的 full SC-CL 候选结果不能替代自由生成和同协议 B1/Random-9 对照。
- NELL 当前只有一个训练 seed，不能声称统计稳定或显著。
- FB15K237-2 的当前 SC-CL bank、adapter parity 和 train-only wiring 仍需重新确认并记录 hash。
