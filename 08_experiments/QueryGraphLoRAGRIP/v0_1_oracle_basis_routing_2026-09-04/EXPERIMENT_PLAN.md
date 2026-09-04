# E09 实验计划：QueryGraph-LoRA GRIP

## 1. 论文主张

### Primary claim

GRIP 的单一静态 LoRA 会把异质 relation/hop/composition 查询压进同一低秩子空间；query-conditioned 参数子空间路由能在 graph-free 推理协议下改善参数知识读取，尤其是 novel composition 与长路径查询。

### Supporting claim

路由后验不确定性与错误风险相关，可为后续参数记忆/KG-RAG 混合路由提供可校准信号。

### 必须排除的替代解释

- 提升仅来自更多可训练参数；
- 提升仅来自 oracle 标签泄漏；
- 任意随机/打乱分组都能提升；
- 提升仅来自 decoder 改变；
- Bayesian 模块只是增加计算量而非提供结构或校准价值。

## 2. 核心假设

- **H1 参数干扰**：relation family、hop 与 path composition 的梯度存在稳定冲突，单一 LoRA 形成折中解。
- **H2 可路由上界**：等参数预算下，正确结构分组的 LoRA basis mixture 优于 static LoRA、uniform、random 与 shuffled routing。
- **H3 query 可预测性**：仅凭 question tokens/base hidden state，learned router 能恢复 oracle 增益的至少一半。
- **H4 不确定性有效**：Bayesian posterior entropy/variance 与错误显著正相关，并改善 ECE/Brier 或 selective risk。

## 3. 模型定义

### 3.1 推荐主实现：共享 basis 路由

```text
ΔW_l(q) = Σ_{m=1..M} π_m(q) · B_{l,m} A_{l,m}
```

- `M` 个共享 LoRA basis/expert；
- 路由器输出 `π(q)`，可用 softmax 或 top-k sparse mixture；
- 同一 trainable-parameter budget 下与更大 static LoRA 比较；
- 首轮只在 GRIP 已使用的 LoRA target modules 上实验；
- 不逐样本生成所有层的完整 A/B，控制 3090 显存与延迟。

### 3.2 Phase C：Bayesian latent query graph

把问题 token/phrase/span 作为节点，推断稀疏潜在交互图，再由图表示产生 `π(q)` 的 posterior。仅在 deterministic router 已通过后加入 KL、采样和 uncertainty；不直接复制 iLoRA 的 microbiome feature graph。

## 4. Phase A — Oracle route upper bound

### 数据与拆分

- 首选 NELL23K，沿用当前 GRIP 的 train/validation/test split、prompt 与指标实现；
- 首轮只运行 validation；test 在协议和 checkpoint selection 冻结前关闭；
- oracle 标签只在训练和机制上界中使用，不作为部署输入。

### 对照矩阵

| ID | 系统 | 目的 |
|---|---|---|
| B0 | Original/static LoRA | 当前参数记忆基线 |
| B1 | Equal-parameter larger static LoRA | 排除更多参数解释 |
| B2 | Uniform basis mixture | 排除 basis 数量本身解释 |
| B3 | Random route | wrong-structure control |
| B4 | Shuffled route | 破坏标签—查询对应关系 |
| O1 | Oracle relation-family route | 检验关系异质性上界 |
| O2 | Oracle hop/depth route | 检验路径长度异质性上界 |
| O3 | Oracle relation-path cluster route | 检验组合结构上界 |

`O1–O3` 必须分别报告，不能在 validation 上挑最佳后当作单一预注册结果。

### Phase A Gate：`GO_QUERY_ROUTER`

同时满足：

1. 至少两个 direct checkpoints / 两 seeds 完成；
2. 最强预注册 oracle 相对 B1 平均 canonical EM `≥ +2.0 pp`；
3. novel-composition canonical EM 平均 `≥ +2.0 pp`；
4. seen-composition 平均下降不超过 `1.0 pp`；
5. B3/B4 不获得同等级提升；
6. trainable parameters、optimizer steps、supervised tokens 与 decoder 已审计匹配。

否则输出 `STOP_GRAPH_CONDITIONAL_LORA`，不开发 learned/Bayesian router。

## 5. Phase B — Deterministic query router

### 输入

只允许：question tokens、attention mask、冻结或可训练 backbone 的 query hidden states。禁止：query-specific KG/subgraph、gold relation/path/hop、答案候选和 test-derived type map。

### 变体

- R1 pooled hidden-state MLP；
- R2 token self-attention router；
- R3 deterministic latent token graph + GNN；
- R4 top-k sparse routing；
- controls：uniform、random、shuffled、wrong-route。

### Phase B Gate：`GO_BAYESIAN_GRAPH`

- 相对 B1 canonical EM `≥ +1.5 pp`；
- 保留 Phase-A oracle 增益的 `≥50%`；
- novel-composition 不退化；
- expert utilization 不塌缩；
- 推理延迟增幅建议 `≤20%`。

## 6. Phase C — Bayesian query graph

比较 deterministic router、Bayesian posterior mean、Monte Carlo routing、去 KL、去稀疏先验、random graph。Bayesian 版本必须同时证明准确率/鲁棒性或校准价值；若只增加复杂度，则保留 deterministic final method。

## 7. 固定公平性

- 相同 backbone、数据 split、prompt、max length 与 decoder；
- 相同训练输入、supervised tokens、optimizer steps、batch/effective batch；
- 参数匹配：basis 总参数与 equal-parameter static LoRA 对齐；
- 相同 seeds 与 checkpoint selection；
- validation/test graph-free；
- 复用 E03 机制验证后冻结的最佳 decoder，避免 L4 与 L6 同时变化。

## 8. 指标与机制证据

### 主指标

canonical EM、raw EM、hop 1/2/3/4、seen/novel composition、macro relation accuracy。

### 机制指标

- relation/path/hop 分桶；
- route accuracy、NMI、routing entropy；
- expert utilization 与 route collapse；
- wrong-route prediction flips；
- per-group gradient cosine/conflict；
- ECE、Brier、selective risk/coverage；
- trainable params、峰值显存、训练/推理延迟。

## 9. 运行顺序与资源

1. E03 gate；
2. 0.5B Phase-A oracle 两 seeds；
3. 仅 GO 后做 0.5B Phase-B；
4. 仅 GO 后做 Bayesian graph；
5. 最终 Qwen2.5-7B 三 seeds。

当前阶段只登记设计，不创建 GPU runner。单张 RTX 3090 优先采用 4 个 basis、top-1/top-2 routing 和共享低维 router；完整 `O(K²)` token graph 仅作后续消融。

## 10. 论文创新边界

不能写成“首次 graph-conditioned LoRA”。可写的创新是：

1. **graph-free query-only latent structure** 条件化参数图知识读取；
2. 将动态图适配从 multi-entity scientific prediction 迁移并重构为 KGQA 的 **parametric memory subspace routing**；
3. 用 oracle-first 设计直接证伪“静态 LoRA 干扰”机制；
4. 同时报告 composition、gradient conflict、wrong-route 和 calibration 证据，而非只报总准确率。
