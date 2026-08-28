# RecurrentGRIP Design v1

更新日期：2026-08-28
状态：设计已确认，尚未进入代码实现

## 1. 设计目标

保持 GRIP 的核心闭卷设定，将一次性前向推理改造成对图专属 LoRA 的共享递归执行。

## 2. 架构

模型划分为：

\[
E_{pre}\rightarrow F_{rec}^{\times K}\rightarrow D_{post}.
\]

### Question Encoder

\[
Z^{(0)}=E_{pre}(q).
\]

负责抽取查询实体、关系约束和输出格式。

### Recurrent Graph Executor

\[
\widetilde Z^{(k+1)}=F_{\phi,\Delta W_G}(Z^{(k)},e_k),
\]

\[
Z^{(k+1)}=Z^{(k)}+g_k\odot\widetilde Z^{(k+1)}.
\]

- \(F\) 的基础参数 \(\phi\) 跨图共享；
- 图专属 LoRA \(\Delta W_G\) 只在正确 adapter 加载后生效；
- 所有 recurrence 共享同一组执行器权重；
- step embedding \(e_k\) 区分当前执行轮次；
- update gate \(g_k\) 控制状态改写。

### Halting Head

\[
p_k^{stop}=\sigma(W_h\operatorname{Pool}(Z^{(k)})).
\]

Pilot 不启用动态停止，正式 v2 再加入。

### Answer Decoder

\[
p(y\mid q,G)=D_{post}(Z^{(K)}).
\]

## 3. 参数职责

| 参数 | 是否跨图共享 | 作用 |
|---|---:|---|
| Base LM | 是 | 语言理解与生成 |
| Recurrent executor \(\phi\) | 是 | 一步图计算规则 |
| Graph LoRA \(\Delta W_G\) | 否 | 图内容和转移结构 |
| Halt head | 是 | 决定执行深度 |
| Probe heads | 分析阶段 | 只测量隐状态，不参与主任务推理 |

## 4. 训练

### Stage A：Executor Meta-Training

在多张训练图和 1–3 hop 问题上学习共享执行器。主设置只使用最终答案监督，真实 hop 仅控制训练 recurrence，不作为输入。

### Stage B：Per-Graph Internalization

冻结 base LM 和 executor，为每张图训练 \(\Delta W_G\)。沿用 GRIP 的 raw memory 与 reasoning QA 两阶段框架。

### Stage C：Closed-Book Execution

只输入问题和正确图 adapter，执行固定或动态次数 recurrence，然后解码答案。

## 5. 默认损失

\[
\mathcal L=\mathcal L_{ans}+\lambda_d\mathcal L_{depth}+\lambda_s\mathcal L_{stable}.
\]

动态停止版本增加：

\[
+\lambda_h\mathcal L_{halt}.
\]

Frontier supervision 只作为独立消融，不进入默认模型。

## 6. 与原始代码的接口

原始 GRIP 保持不变，新建平行实现：

```text
grip/recurrent/model.py
grip/recurrent/executor.py
grip/recurrent/halting.py
grip/recurrent/outputs.py
grip/recurrent/probes.py
grip/training/recurrent_trainer.py
arguments/recurrent_args.py
evaluation/recurrent_metrics.py
evaluation/mechanism_metrics.py
scripts/run_recurrent_grip.py
scripts/run_recurrent_pilot.py
scripts/analyze_recurrent_states.py
```

现有 `scripts/run_grip.py` 继续作为 Original GRIP baseline。

## 7. 版本路线

### v1：Fixed-Depth RecurrentGRIP

- 固定 \(K\)；
- 一个共享 executor；
- final answer loss；
- CLEGR pilot。

### v2：Adaptive RecurrentGRIP

- update gate；
- halt head；
- depth/stability loss；
- 计算量 Pareto 分析。

### v3：Mechanistic RecurrentGRIP

- frontier probing；
- activation patching；
- adapter swap；
- step deletion/permutation；
- 六数据集实验。

## 8. 非目标

v1 不实现：

- successor representation；
- 动态图增量更新；
- syndrome correction；
- 多 adapter 路由；
- 图检索器；
- 大规模 WebQSP/CWQ 扩展。

这些内容只有在主机制成立后才评估。
