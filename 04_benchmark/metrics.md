# Metrics

更新日期：2026-08-28

## 1. 答案质量

- Exact Match；
- token / entity F1；
- Hits@1、Hits@k；
- 多答案 Set Precision、Recall、F1；
- Triple Coverage；
- Hallucination Rate。

## 2. 长度泛化

### Hop-AUC

\[
\operatorname{HopAUC}=\frac{1}{H}\sum_{h=1}^{H}\operatorname{Acc}_h.
\]

### Extrapolation Gap

\[
\operatorname{EG}=\operatorname{Acc}_{seen-hop}-\operatorname{Acc}_{unseen-hop}.
\]

越小越好。

### Compute Scaling Gain

\[
\operatorname{CSG}(K_1,K_2)=\operatorname{Acc}(K_2)-\operatorname{Acc}(K_1).
\]

用于判断增加测试 recurrence 是否真正带来收益。

## 3. 机制指标

- 最优 recurrence 与真实 hop 的 Spearman \(\rho\)；
- frontier probe micro/macro F1；
- remaining-distance probe accuracy；
- correct-to-wrong patch damage；
- wrong-to-correct patch recovery；
- step deletion degradation；
- step permutation degradation；
- hidden trajectory 与 BFS state 的 representational similarity。

## 4. 停止指标

- Halt MAE：\(|K^*-h|\)；
- early-stop rate；
- overthinking rate；
- average recurrence；
- accuracy per unit FLOP。

## 5. 效率指标

- trainable parameters；
- graph adapter size；
- per-graph internalization time；
- inference latency；
- tokens generated；
- FLOPs；
- peak GPU memory。

## 6. 鲁棒性指标

- 节点随机重命名后的性能保持率；
- 问题改写后的性能保持率；
- 图规模增长后的性能下降；
- 分支因子增长后的性能下降；
- adapter 量化或 rank 截断后的性能变化。
