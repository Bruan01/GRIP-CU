# Section plan：SC-CL

更新日期：2026-10-25。本文的叙事主线是：开放世界关系候选学习需要可追溯的候选来源；
SC-CL 提供 train-only、冻结式的结构画像与 B1 violation 组合候选库。它是否比同预算
随机候选更有效仍是假设，需由闭集排序、自由生成、假阴性审计和成本报告共同检验。

| 章节 | 核心问题 | 证据/展示 |
|---|---|---|
| Abstract | 为什么需要可审计候选、目前知道什么 | 方法、NELL 单次与 prefix-200 配对观察、证据边界 |
| Introduction | 候选来源怎样改变学习问题 | 开放世界风险、受限的结构反事实定义、排序与生成区分 |
| Related Work | 与 KGC、对比学习、SANS/ProGCL 的边界 | 真实文献；加入 YAGO3 与 CoDEx 基准定位 |
| Formulation | 合法 train-only candidate 是什么 | 训练图、known-pair 过滤、开放世界边界、InfoNCE |
| Method | 结构画像与 B1 violation 如何采样 | Eq. 1--8；冻结 bank；复杂度与实现边界 |
| Experiments: integrity | 这次 NELL 运行是否可追溯 | 3,253 行 bank、wiring、manifest、训练与预测产物 |
| Experiments: paired result | 哪些变化可直接比较 | NELL 同候选 prefix-200；9 rescue、4 regression；full SC-CL 单列 |
| Experiments: generation | 闭集观察是否等于生成收益 | B1/SC-CL 自由 EM 87.46%/87.03%，分开解释 |
| Experiments: boundary | 还缺什么证据 | 全量 B1、随机与频率匹配对照、多种子及风险审计 |
| Discussion | 现有观察可以支持何种解释 | 候选作用未归因、生成取舍、开放世界真值风险、成本 |
| Appendix | 怎样继续验证 | 历史 FB 单列；超参、schema、同预算与假阴性审计协议 |

## 主要可证伪预测

- 在相同候选预算和训练预算下，SC-CL-9 的闭集 MRR/Hits 应高于 Random-9 和
  frequency-matched Random-9；
- 自由生成 EM/OOV 不应因候选损失出现超过 0.5 个百分点的实质恶化；
- 去掉结构画像、B1 violation 或均匀下限会降低边界间隔或跨 seed 稳定性；
- train-known 过滤应保持硬门为零冲突；CoDEx 的实体替换 verified negatives 不能直接标注固定实体对关系候选的真假，相关审计必须保持任务边界；
- 若上述预测不成立，保留负结果并缩小贡献表述，不用预设数字反向调参。

## 证据边界

当前证据分层：NELL seed=2026 有训练、bank、wiring、预测和候选重排原始文件；FB
有 B1 与历史 InfoNCE-i0 候选评分，但没有当前 SC-CL bank；其他性能、跨数据集迁移和
统计显著性仍为 planned。主文、矩阵、证据账本和 TODO 必须继续同步这一边界。
