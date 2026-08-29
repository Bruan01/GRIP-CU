# RecurrentGRIP v1.1.1 — Storage × Execution Diagnostic Cross

## Question

现有 smoke 的 `K=2` 相比 `K=1` 在 correct 与 no-adapter 条件下都大幅下降，但旧的
`max_context_samples=256` 实际截取了上下文列表开头的 node declarations，未能证明
任何 NELL23K relation edge facts 被写入 adapter。因此现有结果不能直接否定或支持
“共享执行器递归计算”机制。

本版本只回答两个相互独立的问题：

1. **Storage**：graph-specific LoRA 是否真正接触到 relation edge facts？
2. **Execution**：`K_train` 与 `K_eval` 的匹配是否解释 `K=2` 塌缩？

本版本不加入 gate、step embedding、residual scaling 或 adaptive halting，避免同时改变
多个机制后无法归因。

## Storage Repair

每个 NELL23K adapter 固定选择：

```text
32 node declarations
224 edge facts
seed = 2026
```

edge facts 不再只按 relation 抽样，而是先锚定 64 个 train QA 对应的 exact graph facts，
再将剩余 160 个 edge slots 按 relation 分桶做 round-robin 补齐。在当前 NELL23K 训练图
有 198 个 relation 的前提下，该顺序同时保护训练监督涉及的事实，并覆盖长尾 relation。
每个 adapter 目录必须写出：

```text
context_sampling_manifest.json
```

其中包含 node/edge/relation 数量、train-QA relation/entity coverage、选中事实和 SHA256。
如果 manifest 不能证明 edge facts 被选中，该 run 不进入机制判断。

当前固定输入的静态审计结果为：20,799 个节点、24,321 条原始边、24,310 条清洗后
唯一边、198 个 relation；最终选择 32 个 node declarations 和 224 条 edge facts，覆盖
198/198 个 relation、64/64 个 train QA exact facts 与 128/128 个 train QA endpoints。
固定 seed=2026 时，selection SHA256 为
`d4a9cf3f6d2deee090af5137a2b52f89f8ce335926eabbd344ece85729d72885`。

该结果只通过 **storage-input validity**：训练输入确实含目标图事实。它没有通过
**parametric-storage effectiveness**；后者必须由 adapter intervention 结果决定。

## Execution Cross

运行矩阵：

| `K_train` | `K_eval=1` | `K_eval=2` |
|---:|---:|---:|
| 1 | run | run |
| 2 | run | run |

每个格子同时评估：

- correct adapter；
- adapter disabled (`none`)。

validation 与 test 分开汇总。每条 prediction 显式记录 `recurrent_train_k`、生成 token
数、EOS 命中、候选集合命中，以及每一步 pooled hidden state。

## Decision Rules

### G1 — Storage validity

必须同时满足：

- sampling strategy = `node_random_train_qa_anchor_edge_relation_round_robin_v2`；
- selected node/edge count = `32/224`；
- selected relation count = `198/198`，coverage = 1.0；
- train QA exact fact coverage = `64/64`，endpoint coverage = `128/128`；
- `train_k1` 与 `train_k2` 的 selection SHA256 完全一致；
- manifest、环境、预测原始文件均存在；
- correct adapter 与 none 使用同一 evaluation questions。

### G2 — Adapter signal

在 validation 和 test 至少一个 split 中，correct 相对 none 应出现稳定正增益；只看
单个 aggregate 百分点不构成结论，还需检查配对 question transition。

### G3 — Depth compatibility

重点比较：

```text
train1/eval1 vs train1/eval2
train2/eval1 vs train2/eval2
```

- 若只有 `train1/eval2` 明显下降，而 `train2/eval2` 恢复，说明 recurrence 需要
  depth-matched training，固定深度机制仍值得进入稳定化实验；
- 若 correct 的所有 `eval2` 都下降，但 none 同幅度下降，结合 hidden-state norm ratio、
  consecutive cosine 与 relative delta 判断 shared decoder 重复执行的数值/表示漂移；
- 若 correct 与 none 基本一致，说明当前 adapter 更像 QA task tuning，尚未形成可测的
  graph-specific parametric memory；
- 若 correct 在 K=1/2 都稳定优于 none，且 K=2 能解决 K=1 未解决的问题，可进入更大
  NELL23K pilot，并再用 CLEGR/多数据集验证 hop–compute 对齐。

## Claim Boundary

本 diagnostic 只能决定是否继续投入，不用于论文主结果。通过后仍需：

- 至少 3 个随机种子；
- Original GRIP / More-QA / compute-matched baseline；
- 多个 GRIP 数据集；
- 参数量、训练 token、推理 FLOPs 和 wall-time 对齐；
- failure transition、hidden-state norm/cosine/delta 与表示漂移分析。
