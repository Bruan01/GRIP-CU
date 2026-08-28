# Results Analysis Plan

更新日期：2026-08-28

## 1. 首先区分三类失败

### Storage Failure

模型连一跳事实都无法可靠恢复，说明图未被正确写入。

### Retrieval Failure

事实可通过直接问题恢复，但在目标查询中没有被正确激活。

### Execution Failure

相关事实可被恢复，却不能完成关系组合或多跳传播。

所有错误分析必须先归入这三类之一。

## 2. 主结论判定

### 结论 A：Recurrence 有效

需要 RecurrentGRIP 在 Length OOD 上优于 Original GRIP、More-QA 和 ComputeMatch。

### 结论 B：Recurrence 对应 graph hop

需要同时满足：

1. 最优 \(K\) 与 hop 正相关；
2. frontier 可线性解码；
3. 中间状态 patch 对答案有因果影响。

### 结论 C：存储与执行可以分离

需要 adapter swap 和 executor ablation 显示二者承担不同职责。

## 3. 必画图表

- accuracy × true hop × recurrence heatmap；
- seen-hop / unseen-hop 曲线；
- frontier probe F1 × recurrence；
- patch recovery × layer × recurrence；
- average recurrence × accuracy Pareto；
- dataset × failure type 堆叠图；
- rank / adapter size × OOD accuracy。

## 4. 重点失败案例

- 一跳正确、多跳错误；
- 前两步 frontier 正确、后续发散；
- 提前停止；
- 过度递归后正确答案被覆盖；
- 高分支图候选爆炸；
- relation sequence 未见导致组合失败；
- 图 adapter 混淆；
- 基座模型预训练知识与图内事实冲突。

## 5. 禁止的结果解释

- 单凭更高 EM 声称学到了图算法；
- 单凭 probe 结果声称有因果机制；
- 单凭 CLEGR 声称适用于真实 KG；
- 单凭一个随机种子声称稳定提升；
- 不匹配 FLOPs 时声称效率更高；
- 将未通过同协议的数字写成 SOTA 对比。
