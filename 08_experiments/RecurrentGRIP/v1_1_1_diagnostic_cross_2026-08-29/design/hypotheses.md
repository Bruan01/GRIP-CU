# RecurrentGRIP Hypotheses

更新日期：2026-08-28

## H1：Hop–Recurrence Alignment

### Observation

普通 GRIP 使用一次标准前向完成所有 hop，无法判断是否逐跳执行。

### Hypothesis

如果图专属 LoRA 表示可调用的局部转移算子，则每次共享 recurrence 应推进一次隐式图传播。

### Prediction

最优 recurrence 深度与真实最短路径长度显著正相关：

\[
\rho(K_{best},h)>0.
\]

### 证伪条件

- 最优 \(K\) 与 hop 无关；
- 对所有问题固定更大的 \(K\) 都同样有效；
- ComputeMatch baseline 出现相同相关性。

## H2：Frontier Decodability

### Hypothesis

第 \(k\) 轮隐状态包含第 \(k\) 跳搜索 frontier。

### Prediction

冻结模型后的线性 probe 能从 \(Z^{(k)}\) 中预测 frontier，并显著优于：

- shuffled labels；
- 普通 GRIP 同层状态；
- 相邻错误步骤的状态。

### 证伪条件

只有使用非线性大 probe 或直接 frontier supervision 才能获得结果。

## H3：Length Extrapolation

### Hypothesis

参数共享使执行器学习可重复局部规则，而不是固定深度路径模板。

### Prediction

用 1–2 hop 训练后，测试时增加 recurrence 能改善 3–6 hop，且优于同数据、同 FLOPs 的非递归模型。

### 证伪条件

OOD 提升完全由更多训练 QA、更长输出或更多 FLOPs解释。

## H4：Causal Intermediate States

### Hypothesis

递归中间状态参与因果计算，而不是最终答案的伴随相关表示。

### Prediction

- correct-to-wrong patch 会破坏答案；
- wrong-to-correct patch 会恢复答案；
- step deletion 和 permutation 会系统性降低对应 hop 的性能。

### 证伪条件

状态替换不影响结果，或随机状态产生相同恢复率。

## H5：Storage–Execution Separation

### Hypothesis

图 LoRA 主要定义图内容，共享 executor 主要定义计算过程。

### Prediction

- 替换 adapter 会改变具体答案和 frontier；
- 替换 executor 会改变执行质量和长度泛化；
- executor 不加载正确 adapter 时不能恢复测试图事实。

### 证伪条件

共享 executor 单独记住测试图，或图 adapter 只影响输出词而不影响递归轨迹。

## H6：Adaptive Computation

### Hypothesis

不同问题需要不同 recurrence 深度，动态停止可以形成准确率—成本 Pareto 优势。

### Prediction

Adaptive halting 在接近固定大 \(K\) 准确率时，使用更少平均 recurrence。

### 证伪条件

停止头只学习问题表面长度，或固定 \(K\) 在所有成本点上占优。
