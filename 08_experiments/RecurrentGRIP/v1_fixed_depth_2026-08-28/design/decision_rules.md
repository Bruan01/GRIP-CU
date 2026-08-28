# Decision Rules

更新日期：2026-08-28

## Pilot 继续条件

CLEGR pilot 至少满足以下两项，方向才进入完整实现：

1. 3–4 hop 相比 Original GRIP 提升至少 5 个绝对百分点；
2. \(K=3/4\) 在 3–4 hop 上明显优于 \(K=1\)；
3. 最优 recurrence 与真实 hop 呈正相关；
4. 1–2 hop 性能下降不超过 2 个百分点；
5. 改变 adapter 会系统性改变执行轨迹。

## Pilot 终止条件

出现任一情况，暂停 RecurrentGRIP 主线并回到机制诊断：

1. ComputeMatch baseline 获得同等提升；
2. 最优 recurrence 与 hop 无关；
3. 增加 recurrence 持续损害所有样本；
4. 节点随机重命名后全部收益消失；
5. 只有显式 frontier supervision 才能学习；
6. 共享 executor 在不加载正确 adapter 时仍能回答测试图问题。

## 正式论文 Claim 门槛

### 可以声称“提高多跳闭卷图推理”

需要六数据集中至少三个真实数据集稳定提升，并有多种子结果。

### 可以声称“长度外推”

需要训练和测试 hop 严格不重叠，并排除更短路径捷径。

### 可以声称“逐跳执行”

同时需要：

- hop–recurrence alignment；
- frontier probe；
- causal intervention。

只有 probe 结果不足以支持该 claim。

### 可以声称“效率更好”

必须报告相同硬件、batch、最大生成长度、FLOPs、延迟和平均 recurrence。

### 可以声称“优于 KG-RAG”

需要与强结构化检索系统在相同知识范围和评价协议下比较；否则只描述二者边界，不写替代关系。
