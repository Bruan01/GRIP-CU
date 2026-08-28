# RecurrentGRIP Paper Outline

更新日期：2026-08-28

## Abstract

1. 背景：GRIP 类方法可以把图写入模型参数；
2. 问题：存储成功不等于图知识可以被组合执行；
3. 方法：共享递归执行器 + 图专属 LoRA；
4. 机制：hop alignment、frontier probing、activation patching；
5. 实验：六数据集、长度/规模/关系组合 OOD；
6. 结论：明确参数图记忆从 storage 到 execution 的能力边界。

## 1. Introduction

### 1.1 Parametric Graph Memory

介绍推理时不提供图的价值，以及与 KG-RAG 的不同。

### 1.2 Storage Is Not Execution

说明一跳事实回忆、路径模板记忆和真正图执行的区别。

### 1.3 Key Question

Can an internalized graph be executed as a recurrent parametric program?

### 1.4 Method Overview

- shared executor；
- graph-specific LoRA；
- recurrent hidden trajectory；
- optional adaptive halting。

### 1.5 Contributions

1. executable parametric graph memory 问题定义；
2. RecurrentGRIP 方法；
3. 表征和因果机制证据；
4. 六数据集的边界分析。

## 2. Related Work

### 2.1 In-Parameter Knowledge and Graph Memory

GRIP、参数知识内化、knowledge editing。

### 2.2 Recurrent and Adaptive Computation

Depth recurrence、test-time compute、recurrent reasoning。

### 2.3 Neural Algorithmic Graph Reasoning

reachability、BFS、长度外推、中间状态对齐。

### 2.4 KG-RAG and Explicit Graph Reasoning

作为强对照，不将弱文本 RAG 当作唯一检索基线。

## 3. Problem Formulation

### 3.1 Graph Internalization

定义图 \(G\)、图专属参数 \(\Delta W_G\) 和闭卷查询。

### 3.2 Storage, Retrieval, and Execution

分别定义三个成功条件和对应指标。

### 3.3 Executable Parametric Graph Memory

定义递归状态、执行深度和无图输入约束。

## 4. RecurrentGRIP

### 4.1 Architecture Overview

Encoder、recurrent executor、decoder。

### 4.2 Graph-Specific LoRA Memory

解释 \(\Delta W_G\) 如何沿用 GRIP 写入。

### 4.3 Shared Recurrent Execution

给出核心递推公式与参数共享。

### 4.4 Adaptive Halting

停止头、成本正则和过度思考控制。

### 4.5 Training Objective

答案、深度、稳定性和停止损失。

## 5. Experimental Setup

### 5.1 Datasets

CLEGR、Scene Graph、FB15K237、WN18RR、CoDEx-Medium、NELL23K。

### 5.2 Evaluation Splits

ID、Length OOD、Relation Composition OOD、Size OOD、Label OOD、Paraphrase OOD。

### 5.3 Baselines

Original GRIP、More-QA、Deep、ComputeMatch、Self-Refine、Graph Context、KG-RAG、Oracle Path。

### 5.4 Metrics

答案、泛化、机制、停止和效率指标。

### 5.5 Implementation Details

模型、LoRA、executor 层位置、recurrence sweep、训练预算和随机种子。

## 6. Main Results

### 6.1 ID Performance

证明方法没有以损害基础回忆为代价。

### 6.2 Length Extrapolation

全文主结果。

### 6.3 Relation and Graph-Size Generalization

证明不是固定路径模板。

### 6.4 Comparison with Explicit Graph Access

分析参数记忆与 KG-RAG 边界。

## 7. Mechanistic Analysis

### 7.1 Hop–Recurrence Alignment

热力图和相关系数。

### 7.2 Frontier Decoding

线性 probe 与控制实验。

### 7.3 Causal Activation Patching

正确/错误轨迹双向 patch。

### 7.4 Storage–Execution Factorization

adapter swap、executor ablation。

### 7.5 Fixed-Point and Overthinking Behavior

分析状态收敛和递归过度。

## 8. Ablations and Efficiency

- shared/unshared；
- layer position；
- gate；
- halt；
- loss；
- rank；
- 参数和 FLOPs 匹配。

## 9. Failure Analysis and Limitations

- 高分支；
- 多答案；
- 环；
- 真实 KG 噪声；
- per-graph adapter 成本；
- 不能快速更新动态图；
- recurrence 不保证严格符号正确性。

## 10. Conclusion

总结经过实验证明的 storage–execution 边界，不宣称参数记忆全面替代检索。
