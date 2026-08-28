# RecurrentGRIP Evaluation Protocol

更新日期：2026-08-28

## 1. 闭卷约束

主实验推理输入只能包含：

- 问题文本；
- 必要的输出格式说明；
- 已加载的图专属 adapter。

不得输入：

- 原始图；
- 邻接表；
- 三元组；
- 检索子图；
- oracle relation path；
- 真实 hop 数。

显式图输入仅用于 Graph Context 和 Oracle Path 上界。

## 2. 精确 Hop 定义

一个样本被标记为 \(h\)-hop，当且仅当在该任务定义的关系约束下：

\[
d_G(s,t)=h.
\]

采样到长度为 \(h\) 的任意路径，不足以证明样本是 \(h\)-hop。

## 3. 捷径检查

每个多跳样本必须自动检查：

- 是否存在直接目标边；
- 是否存在 inverse edge 一步恢复；
- 是否有别名或描述直接包含答案；
- 是否有更短等价关系路径；
- 基座模型在未加载图 adapter 时是否已经知道答案。

## 4. 主要切分

### ID Split

训练和测试 hop 范围一致，用于基本有效性。

### Length OOD

- 训练：1–2 hop；
- 测试：3–6 hop；
- CLEGR 扩展到 8 hop。

### Relation Composition OOD

测试 relation sequence 在训练集中不出现，但单个 relation 可出现。

### Graph Size OOD

训练使用小图，测试使用更大的图。

### Label OOD

实体名称随机化，消除预训练知识和语义捷径。

### Paraphrase OOD

训练模板和测试模板严格分离，测试自然语言表达变化。

## 5. 公平比较

所有主要基线保持：

- 相同基座模型；
- 相同图数据；
- 相同训练 QA 数；
- 相同最大训练 token；
- 相同随机种子；
- 报告参数量和 FLOPs；
- 分别给出参数匹配与计算匹配对照。

## 6. Recurrence Sweep

固定测试集合，报告：

\[
K\in\{1,2,3,4,5,6,8\}.
\]

对于每个 hop 分桶，记录每个 \(K\) 的准确率，不能只报告挑选后的最佳值。

## 7. Dynamic Halting

动态停止模型不得接收真实 hop。报告：

- 平均 recurrence；
- halt depth 与真实 hop 的误差；
- 提前停止率；
- 过度递归率；
- 准确率—计算量 Pareto 曲线。

## 8. 统计协议

- Pilot 可使用单种子做方向判断；
- 正式主结果至少 3 个随机种子；
- 关键主表报告均值、标准差和配对显著性检验；
- 所有负结果和被终止实验保留日志。
