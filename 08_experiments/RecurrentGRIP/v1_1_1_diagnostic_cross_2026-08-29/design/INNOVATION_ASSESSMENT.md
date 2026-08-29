# RecurrentGRIP 创新点评估与可证伪路线

更新日期：2026-08-29

## 结论

**当前的“固定深度重复一个 decoder block”不能单独作为论文创新点。** 共享参数循环、
latent recurrence 和通过增加循环次数扩展 test-time compute 已有直接先例。当前版本
`v1.1.1` 的价值是一个必要的 **falsification gate**：先把 graph-fact storage、adapter
因果效应和 recurrent execution stability 分开，再决定是否实现真正的方法版本。

RecurrentGRIP 仍可继续，但论文主张应从：

> 重复 decoder layer 可以提升图推理。

收缩并重构为：

> graph-specific parameter memory 能否成为一个可重复应用、可干预、可按问题复杂度分配
> 计算的图状态转移算子？它在什么 storage / stability / depth 边界内有效？

这不是措辞修改，而是要求实验同时证明：adapter 中有 graph-specific 信息、每次 recurrence
不是无意义漂移、增加的计算确实修复了一部分较难问题。

## 与已有工作的重合

| 工作 | 与当前方案的重合 | 对 RecurrentGRIP 的约束 |
|---|---|---|
| GRIP, arXiv:2511.07457 | 用 graph-specific LoRA 将图关系内化到参数，并在无原图推理 | “图知识写进 LoRA”本身是基线贡献，不是新增点 |
| Looped Transformers for Length Generalization, arXiv:2409.15647 / ICLR 2025 | 共享 Transformer 反复执行，并研究循环次数与长度泛化 | “重复共享 block + depth sweep”不新 |
| Scaling up Test-Time Compute with Latent Reasoning, arXiv:2502.05171 / NeurIPS 2025 | recurrent block、可变训练深度、测试时增加 latent compute | “K 增大即推理更深”的主张需要更强机制证据 |
| Stability and Generalization in Looped Transformers, arXiv:2604.15259 | 用 fixed-point、recall、outer normalization 分析循环稳定性 | 当前 direct composition 缺少 recall/outer norm，K=2 漂移有明确先验风险 |

参考：

- https://arxiv.org/abs/2511.07457
- https://arxiv.org/abs/2409.15647
- https://arxiv.org/abs/2502.05171
- https://arxiv.org/abs/2604.15259

## 当前实现的机制缺口

当前 executor 实际执行：

```text
H_0 = decoder 输入
H_{t+1} = Block(H_t; W + ΔW_G)
```

它没有：

- 对原始输入 `H_0` 的 recall；
- 循环外 normalization；
- step/depth conditioning；
- residual update scale 或 gate；
- variable-depth training；
- 中间图状态监督。

因此 v1.1 smoke 中 correct/none 同步发生 K=2 塌缩，与 direct composition 导致的表示漂移
相符；但由于旧 context cap 没有可靠注入 edge facts，旧结果还不能判断 graph memory 是否
参与了该漂移。

## v1.1.1 现在能回答什么

### 1. Storage-input validity

已经静态验证：

- 64/64 train QA exact facts 进入 Stage-1 context；
- 198/198 relations 被覆盖；
- 128/128 train QA endpoints 出现在 context；
- `train_k1` 与 `train_k2` 必须使用相同 selection SHA256。

这只证明“训练看到了事实”，不证明“LoRA 记住了事实”。

### 2. Parametric-storage effectiveness

由同题干预决定：

```text
correct adapter vs none adapter
```

如果 correct 与 none 基本相同，则当前 LoRA 主要学到任务格式或通用关系先验，不能把结果
解释为 graph-specific parameter memory。

### 3. Recurrent execution stability

除了 accuracy 和 K1→K2 transition，本版本现在直接保存并分析每一步 pooled hidden state：

- hidden norm；
- final/initial norm ratio；
- consecutive cosine；
- consecutive relative delta。

若 K=2 accuracy 塌缩同时伴随 norm 爆炸/收缩或大幅方向变化，才有证据把失败定位到
recurrent representation drift，而不只是输出 parser 或随机波动。

## Go / Pivot / Stop 判据

### Go：继续做真正的方法版本

至少同时观察到：

1. `correct - none` 在 validation 和 test 方向一致；
2. `train_k2/eval_k2` 明显优于 `train_k1/eval_k2`，说明 depth-matched training 能修复失配；
3. K=2 存在非零且稳定的 `K1 wrong → K2 correct` 样本，而不是只破坏 K1 已答对样本；
4. hidden-state dynamics 没有系统性爆炸、塌缩或近乎随机旋转；
5. 输出候选命中率、EOS 率和空输出率没有随 K=2 退化。

通过后再创建独立 `v1.2`，候选机制为：

```text
H_{t+1} = OuterNorm(
    H_t + α_t · Gate_t · [Block(H_t, Recall(H_0); W + ΔW_G) - H_t]
)
```

其中 recall、outer norm 和 depth-conditioned gate 必须分别做消融，不能一次全部加入后只报告
最终分数。

### Pivot：保留问题，替换 direct recurrence

出现以下模式时，转向“稳定图状态转移算子”，而不是继续增加 K：

- depth-matched training 有部分恢复，但 K=2 仍明显破坏 K1 正确样本；
- correct 有 adapter 信号，但 hidden-state drift 明显；
- K2-only gains 存在，但平均性能被不稳定更新抵消。

下一版优先单变量测试顺序：

1. outer normalization；
2. input recall；
3. residual scaling；
4. depth-conditioned gate / adaptive halt。

### Stop：当前 RecurrentGRIP 主线不成立

以下模式同时出现时，不再把 recurrence 作为主贡献：

- correct 与 none 无稳定差异；
- `train_k2/eval_k2` 仍与 `train_k1/eval_k2` 同幅塌缩；
- K2 几乎没有新增解题，只持续改坏 K1 的正确答案；
- state dynamics 显示系统性漂移；
- 在至少一个真正受控的多步数据集上也不能建立 step–difficulty 对齐。

此时可保留 v1.1.1 作为负结果和审计资产，研究主线回到 GRIP 的 storage / routing /
continual-update 问题。

## 数据集边界

NELL23K 适合先做：

- graph-fact storage 输入审计；
- correct/none adapter 干预；
- train/eval depth compatibility；
- executor 输出与表示稳定性诊断。

但当前 NELL23K relation prediction 的 `structural_distance` 只是 train graph 上的无向最短路
诊断量，不是受控的多跳推理标签。因此 NELL23K 可以决定该机制是否值得继续，却不能单独
支持“每次 recurrence 对应一步图推理”。论文阶段仍需从 GRIP 已支持数据中选择至少一个
真正具有受控多步结构的数据集，并加入 Original GRIP、More-QA 和 compute-matched baseline。

## 当前决策

```text
v1.1.1 diagnostic cross：继续运行
直接扩大 NELL23K Pilot：暂不运行
固定深度 direct recurrence 作为论文主创新：暂不成立
RecurrentGRIP 研究问题：保留
进入 v1.2 稳定化机制：等待 WSL diagnostic cross 触发 Go/Pivot 条件
```
