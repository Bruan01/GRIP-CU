# RecurrentGRIP v1.1.1 Diagnostic Cross：结果、主张边界与选题决策

更新日期：2026-08-30  
源提交：`8955bf5`  
源 run：`wsl3090_nell23k_diag_cross_20260830_01_nell23k_diagnostic_cross`

## 1. 结论先行

**停止把 fixed-depth RecurrentGRIP 作为刷榜论文主线；不放弃 GRIP，而是在 GRIP 内转向“图结构化参数记忆的选择性检索与组合”。**

下一主线建议命名为 **FactorGRIP / CompositionalGRIP**：把单个、纠缠的 graph LoRA 分解为关系/社区结构化的低秩专家，在固定总参数预算下由问题进行稀疏选择和组合。RecurrentGRIP 保留为失败诊断、稳定性对照和论文中的负结果，不再直接扩大数据规模。

在实现 FactorGRIP 前，先做一个便宜且有决策价值的 **Candidate-Energy Probe**：使用现有 checkpoint，不重新训练，比较自由生成、候选约束生成和十候选序列似然排序。它用于判断当前主要瓶颈位于 decoder/retrieval，还是 graph storage 本身。

## 2. 结果完整性审计

| 检查项 | 结果 |
|---|---|
| Git 拉取 | 本地与远程一致，HEAD=`8955bf5` |
| `cross_run_audit.status` | `pass` |
| prediction 数 | 768 |
| 设计格子 | `2 train K × 2 eval K × 2 controls × (32 val + 64 test)` |
| 每个条件计数 | validation=32，test=64 |
| context selection | train-K1/K2 使用相同 SHA256 |
| selection SHA256 | `d4a9cf3f6d2deee090af5137a2b52f89f8ce335926eabbd344ece85729d72885` |
| edge coverage | 224 edges，198/198 relations |
| train QA fact coverage | 64/64 |
| train QA endpoint coverage | 128/128 |
| trace | 768 条均有 hidden-state trace |
| 错误扫描 | 未发现 OOM、NaN、Traceback、timeout 或中途失败 |

因此，本次 run 足以进行 **smoke/diagnostic 级别** 的 Go/Pivot/Stop 判断；它不具备 paper-scale 性质，因为仅有一个 seed、Qwen2.5-0.5B、64 个训练 QA 和 64 个 test questions。

## 3. 核心 test 结果

validation 只用于方向一致性检查；下表以 test 为主。

| Train K | Eval K | Correct adapter | No adapter | Correct−None |
|---:|---:|---:|---:|---:|
| 1 | 1 | 11/64 = 17.19% | 16/64 = 25.00% | -7.81 pp |
| 1 | 2 | 2/64 = 3.13% | 0/64 = 0.00% | +3.13 pp |
| 2 | 1 | 4/64 = 6.25% | 16/64 = 25.00% | -18.75 pp |
| 2 | 2 | 3/64 = 4.69% | 0/64 = 0.00% | +4.69 pp |

配对 McNemar 结果：

- `train1/eval1` correct vs none：`p=0.302`；
- `train2/eval1` correct vs none：`p=0.000488`，但方向是 adapter **显著变差**；
- `train1/eval2` correct vs none：`p=0.5`；
- `train2/eval2` correct vs none：`p=0.25`。

**最佳 correct-adapter 条件只有 17.19%，低于 25% no-adapter 基线。** 这不是可扩大为刷榜 Pilot 的正信号。

### 3.1 Depth matching 修复了“灾难性塌缩”，但没有产生增量推理

Test 上 K1→K2 的逐题转移：

| Train K | adapter | K1/K2 都对 | K1 对→K2 错 | K1 错→K2 对 | 净变化 |
|---:|---|---:|---:|---:|---:|
| 1 | correct | 1 | 10 | 1 | -9 |
| 1 | none | 0 | 16 | 0 | -16 |
| 2 | correct | 1 | 3 | 2 | -1 |
| 2 | none | 0 | 16 | 0 | -16 |

`K_train=2` 的确把 correct-adapter 的 K2 净损失从 `-9` 降至 `-1`，说明训练深度匹配影响稳定性；但它只新解 2 题、破坏 3 题，**结果是近似停滞，不是新增多步推理收益**。

### 3.2 K2 主要造成输出流形漂移

Test candidate-hit rate：

| Train K | Eval K | correct | none |
|---:|---:|---:|---:|
| 1 | 1 | 37.50% | 42.19% |
| 1 | 2 | 6.25% | 1.56% |
| 2 | 1 | 10.94% | 42.19% |
| 2 | 2 | 21.88% | 1.56% |

观察：

- no-adapter 从 K1 到 K2 的 candidate hit 从 42.19% 降到 1.56%；
- train1/correct 的 EOS rate 从 96.88% 降到 32.81%，平均生成长度从 10.25 增到 20.52 tokens；
- train2/correct/eval2 能把更多输出拉回候选集合，但 14 个候选内输出中只有 3 个正确。

因此当前问题不只是 parser；第二次 block 执行会把生成分布推离任务候选空间，而 depth-matched adapter 主要恢复“像答案的输出”，没有恢复正确 relation retrieval。

### 3.3 粗粒度 hidden-state 指标不能解释谁会被 K2 改错

K2 平均呈现：

- final/initial norm ratio：约 `1.15–1.18`；
- consecutive cosine：约 `0.942–0.947`；
- relative delta：约 `0.38–0.41`。

但 `correct→wrong`、`wrong→correct` 与 `wrong→wrong` 的这些均值高度接近。当前 trace 证明第二步造成系统性大幅移动，却不能证明该移动对应 graph frontier、relation composition 或正确推理状态。要得到机制主张，需要 relation/frontier probe 或因果干预，而不是继续比较 norm/cosine。

## 4. Result-to-Claim 判定

| Claim | 判定 | 证据 | 混杂/缺口 | 所需补充实验 |
|---|---|---|---|---|
| C1：Stage-1 看到了 graph edge facts | **supported** | 224 edge、198/198 relation、64/64 train fact、128/128 endpoint coverage | 只证明输入覆盖，不证明参数存储 | 无；manifest 已足够支持输入侧陈述 |
| C2：graph-specific LoRA 形成了可用参数记忆 | **unsupported** | adapter 改变输出分布 | correct 在最佳 K1 条件低于 none；K2 小增益不显著；无 shuffled/wrong-graph control | wrong-graph/shuffled control、candidate-likelihood probe、训练事实 recall probe |
| C3：depth-matched training 修复 K2 | **partial** | test K1→K2 净损失由 -9 变 -1 | 绝对准确率仍极低，validation 仅持平；train2 损害 K1 | 多 seed；稳定更新与 compute-matched baseline |
| C4：第二次 recurrence 产生新增推理收益 | **unsupported** | train2/correct 仅 2 新解、3 被破坏 | 新解少且无 hop/frontier 对齐 | CLEGR/path-query、step probe、step deletion/patching |
| C5：当前版本值得直接扩大到多数据集 | **unsupported** | 没有正向 test effect | 0.5B、单 seed、small smoke；最佳 correct 低于 none | 先更换机制，再做多数据集 |
| C6：当前结果支撑刷榜/性能论文 | **unsupported** | 无一个条件超过 no-adapter；未与 Original GRIP full protocol 比较 | 与论文 Qwen2.5-7B、完整训练任务、完整 test 完全不同量级 | 复现 Original GRIP official protocol + 新方法公平主表 |

当前可写入项目的最强陈述是：

> Depth-matched training can suppress the catastrophic degradation caused by an unmatched repeated decoder block, but does not turn recurrence into useful graph reasoning; the dominant failure is selective retrieval/output-manifold control rather than verified multi-step execution.

## 5. Idea-discovery：三个可发表方向

### Idea A：FactorGRIP / Graph-Structured Adapter Composition（主推）

**问题。** 单个 graph LoRA 同时承担数百种 relation 的存储和检索，容易产生参数干扰；问题只需要其中很小一部分知识。

**方法。**

\[
\Delta W(q)=\sum_{m\in \operatorname{TopK}(g(q,C))}\alpha_m(q,C)\Delta W_m,
\]

其中专家不是任意 MoE，而是按 graph relation family、community 或 learned graph factor 构造；总 rank 和 monolithic GRIP 保持一致。单跳 relation prediction 使用稀疏选择，多跳 path-query 使用有序专家组合。

**论文故事。**

> Monolithic graph adapters conflate storage and retrieval. Graph-structured low-rank factorization provides selective, compositional access to internalized graph knowledge under an equal parameter budget.

**必须防守的 novelty 风险。** 通用 MoE-LoRA、动态 LoRA fusion 和 adapter routing 已经拥挤；近期直接近邻还研究了 per-entity LoRA bank 的 storage–retrieval gap。因此创新点必须落在：

1. graph-derived factorization，而非任意 expert；
2. equal-total-rank/equal-token 控制；
3. relation/path 的有序组合，而非一次 embedding router；
4. 闭卷图推理、多数据集和 wrong-factor/permutation 因果控制。

### Idea B：ScoreGRIP / Contrastive Candidate-Energy Retrieval（诊断和辅方法）

把自由生成改成十候选序列似然或 energy ranking，并用 graph-aware hard negatives 训练：

\[
s(q,r_i;\Delta W_G)=\log p(r_i\mid q,\Delta W_G),\qquad
\mathcal L=-\log \frac{e^{s(q,r^+)}}{\sum_i e^{s(q,r_i)}}.
\]

它直接针对本 run 的 candidate-manifold 漂移，开发速度快、最可能快速涨点。但 KG completion 的对比学习与候选排序已有大量工作，且 GRIP benchmark 本身就是十分类，因此 **单独作为论文主创新偏弱**。建议把它作为 P0 probe、FactorGRIP 的统一 decoder 和强 baseline。

### Idea C：Uncertainty-Routed GRIP + Minimal Graph Retrieval（高分但改变赛道）

根据 calibrated uncertainty 选择：参数回答、局部 graph retrieval 或 abstain/recompute，优化 accuracy–latency Pareto。它可能最容易做出高绝对准确率，但推理时重新访问图，改变了 GRIP 的 closed-book 核心设置。适合独立的 hybrid/system paper，不适合作为“原设定刷榜”的第一选择。

## 6. Idea-evaluator 评分

分数为 1–10；Higher=性能上限，Faster=获得可信正信号速度，Stronger=机制与证据强度，Cheaper=3090 成本，Broader=跨数据集/任务扩展性。

| 方向 | Higher | Faster | Stronger | Cheaper | Broader | Verdict |
|---|---:|---:|---:|---:|---:|---|
| FactorGRIP | 8 | 7 | 8 | 8 | 9 | **Accept with revisions；主线** |
| ScoreGRIP | 7 | 9 | 5 | 7 | 6 | **作为 probe/组件，不单独成文** |
| HybridGRIP | 9 | 6 | 7 | 5 | 9 | **另立 open-book 系统赛道** |
| Stable RecurrentGRIP v1.2 | 5 | 4 | 5 | 6 | 7 | **Reject as current main line** |

FactorGRIP 排第一的原因不是“模块更复杂”，而是它直接对应当前数据暴露出的 storage–retrieval interference，同时保留 GRIP 闭卷设定、适配四个官方 KG、参数成本可控，并能形成清晰的等预算消融。

## 7. 下一步实验路线

### P0：Candidate-Energy Probe（先做，0.5–1.5 GPU 小时）

不重新训练，在当前 96 questions 上比较：

1. current free generation；
2. candidate-constrained decoding；
3. batched sequence-likelihood ranking；
4. correct vs none；
5. K1 only，Recurrent K2 只保留为诊断。

Go gate：

- correct adapter 相对 none 在 validation/test 同方向；
- test 至少 `+5 pp`；
- scoring 相对 generation 至少 `+10 pp`；
- wrong-graph/shuffled control 明显下降。

若 scoring 仍不能使 correct 超过 none，说明 storage/representation 才是主要故障，直接进入 FactorGRIP，不再优化 decoder。

### P1：FactorGRIP smoke（2–6 GPU 小时）

- 数据：NELL23K 64/32/64；
- 模型：Qwen2.5-0.5B；
- 专家：`M ∈ {4, 8, 16}`；
- 固定总 rank：例如 monolithic `r=8` 对比 factorized total rank budget=8；
- partition：relation-family、graph community、random partition；
- routing：uniform、question gate、oracle upper bound；
- decoder：candidate energy；
- controls：none、wrong factor、permuted routing、monolithic equal-rank。

Go gate：test 上 FactorGRIP 相对 monolithic 至少 `+5 pp`，且 validation 同方向；random partition/permuted routing 明显更差。

### P2：NELL23K official-scale Pilot（建议 1.5B 筛选，7B 终验）

先复现 Original GRIP 的完整训练任务和 test protocol，再加入 FactorGRIP。最终主表至少包含：

- Base LM；
- Original GRIP；
- GRIP + More-QA；
- ScoreGRIP；
- FactorGRIP；
- compute/parameter-matched MoE-LoRA；
- random graph partition；
- wrong routing。

### P3：四 KG 主表

顺序建议：

1. NELL23K：开发和长尾 relation；
2. WN18RR：少 relation、层次语义，验证专家数不是越多越好；
3. CoDEx-Medium：稀疏、多样关系；
4. FB15K237：大图和高 rank，验证可扩展性。

最终使用至少 3 seeds，报告 mean±std、paired bootstrap、推理输入 token、adapter 参数量、训练 GPU 小时与吞吐。CLEGR 只在 FactorGRIP 出现正向多数据集结果后用于 compositional mechanism 证明，不再作为性能论文前置条件。

## 8. 发表边界

### 当前不能写

- RecurrentGRIP improves GRIP；
- recurrence depth corresponds to graph hop；
- graph LoRA is an executable program；
- NELL23K SOTA。

### FactorGRIP 达门后可写

- 在相同 graph、base model、训练 token、总 LoRA rank 和推理图访问限制下，图结构化 adapter factorization 提升四个 KG 的平均准确率；
- 提升主要来自减少 relation interference 和提高 selective retrieval，而不是增加参数或候选后处理；
- relation/path composition controls 为机制提供因果证据。

## 9. 统计解释边界

- 本次所有 p 值是 diagnostic 描述，不进行多重比较后的论文级显著性声明；
- question 共享实体和 relation，严格论文分析应按 relation/source cluster bootstrap；
- test 已参与多轮 smoke 诊断，不能继续把同一小 test 子集作为最终无偏 test；正式实验必须冻结配置并使用完整官方 test 或新建未触碰 holdout；
- 单 seed 结果不能支持一般性结论。

## 10. 产物

本目录包含：

- `analyze_paired_results.py`：纯 Python 可复现分析；
- `condition_metrics.csv`；
- `paired_comparisons.csv`；
- `mcnemar_tests.json`；
- `k1_k2_transitions.csv`；
- `state_by_transition.csv`；
- `failure_transition_examples.jsonl`；
- `analysis_manifest.json`。

## 11. 近邻文献核验记录

- GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models, arXiv:2511.07457 / KDD 2026。
- Parameter Memory vs. Retrieval: A Mechanistic Study of the Storage–Retrieval Gap in Graph Reasoning, arXiv:2608.25489。
- Depth-Recurrent Transformer, OpenReview `Q7boRE3kBy`。
- Recurrent Transformers Learn to Reason, OpenReview `Au7WqYeoHb`。
- MOELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning, arXiv:2401.06954。
- Multi-Head Routing of One-Shot Context Vectors for Efficient Transformer Adaptation, ICLR 2024 OpenReview。
- LoRA-Flow: Dynamic LoRA Fusion for Large Language Models in Generative Tasks, arXiv:2402.11455。

正式投稿前需再次执行按标题、作者、venue、版本号的完整 novelty audit。
