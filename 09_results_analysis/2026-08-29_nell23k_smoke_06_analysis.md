# RecurrentGRIP NELL23K Smoke `_06` 结果分析

更新日期：2026-08-29

## 1. 分析对象与证据边界

- 远程分支：`wsl/nell23k-smoke-20260829`
- 提交：`ca12e0f9952a0b08da7eab028b003532958b848b`
- 运行：`wsl3090_nell23k_smoke_20260829_06`
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 数据：NELL23K，train/validation/test=`64/32/64`
- 训练 recurrence：`K_train=2`
- 推理 recurrence：`K_eval=1,2`
- adapter control：`correct/none`
- 候选关系数：10
- 随机种子：2026

本次远程提交保存了汇总日志，但没有保存 `predictions.jsonl`、
`analysis/summary.json` 和三个 CSV。因此，本分析可以复核代码、汇总数字的一致性和
部分由汇总数字唯一确定的配对关系，但不能独立重算 split/hop bucket、输出空值率、
候选外输出率和 correct-vs-none 的配对显著性。

## 2. 结论摘要

### 2.1 可以确认的结论

1. **端到端工程链路已经打通。** 训练、adapter 保存/重载、CUDA 推理、K sweep、
   输出解析和结果分析均完成；WSL 测试为 21 cases 通过、1 case 因 CUDA 可用而按预期
   skip。
2. **K=2 在当前实现下发生显著性能塌缩。** correct adapter 从 K=1 的 30.21% 降到
   K=2 的 3.13%；none 从 26.04% 降到 1.04%。
3. **塌缩主要来自 recurrent executor，而不是 adapter。** correct 与 none 都下降约
   25–27 个百分点，下降方向和幅度高度一致。
4. **当前没有 hop–recurrence alignment 证据。** 汇总 Spearman 仅为 0.0912，接近零；
   而且 NELL23K 的结构距离只是 train graph 上的无向最短路诊断量，不是受控推理 hop。
5. **correct adapter 的增益很弱。** K=1 仅比 none 高 4.17 个百分点；K=2 仅高 2.08 个
   百分点。没有样本级交叉表时，不能判断该差异是否显著。
6. **当前配置不应直接进入两小时 Pilot。** 应先修复 context cap 的采样问题，并运行
   一个诊断 smoke。

### 2.2 不能据此声称的结论

- 不能声称 RecurrentGRIP 优于 Original GRIP；本次没有 Original GRIP 同协议 baseline。
- 不能声称 recurrence 对应图 hop。
- 不能声称 adapter 成功存储了 NELL23K 图事实。
- 不能声称长度外推、效率优势或论文级稳定提升。

## 3. 主结果

| Adapter | K | Correct | Accuracy | Wilson 95% CI |
|---|---:|---:|---:|---:|
| correct | 1 | 29/96 | 30.21% | [21.93%, 40.01%] |
| correct | 2 | 3/96 | 3.13% | [1.07%, 8.79%] |
| none | 1 | 25/96 | 26.04% | [18.31%, 35.62%] |
| none | 2 | 1/96 | 1.04% | [0.18%, 5.67%] |

效应量：

- correct：K=2 相对 K=1 **-27.08 pp**；
- none：K=2 相对 K=1 **-25.00 pp**；
- K=1 adapter 增益：**+4.17 pp**；
- K=2 adapter 增益：**+2.08 pp**。

10-way candidate 的均匀随机参考准确率为 10%。K=1 的 correct 和 none 都明显高于该
参考值，但 none 也达到 26.04%，说明主要信号不能归因于图 adapter。K=2 的两种控制均
低于 10%，更像执行/输出退化，而不是“多执行一步但收益不足”。

## 4. 可由汇总数字恢复的配对证据

correct adapter 在 K=1、K=2 下分别答对 29 和 3 个问题；跨两个 K 共解决 31/96 个唯一
问题。因此集合关系被唯一确定为：

- 两个 K 都正确：1；
- 仅 K=1 正确：28；
- 仅 K=2 正确：2；
- 两个 K 都错误：65。

对 28 vs 2 的 discordant pairs 做 exact McNemar test，双侧
`p = 8.68e-7`。这只说明在本次固定样本和单种子运行中，K=1 明显优于 K=2；它不等于
跨数据集、跨种子的论文级显著性。

这个集合关系还表明 K=2 不是简单复现 K=1：它改写了大量 K=1 已正确的答案，却只新
修复 2 个问题。当前 recurrence 的主要行为是破坏已有解，而非推进图推理。

## 5. 最关键的协议问题：256 context cap 实际上截取了节点文本

Smoke 新增了：

```python
context_samples = context_samples[:max_context_samples]
```

并设置：

```text
--max_context_samples 256
```

但 `GenGraphContextTask` 的生成顺序是：

1. 先追加全部 node context；
2. 再追加全部 edge context。

NELL23K train graph 有 20,799 个节点。因此前 256 个 context sample 位于 node context
区间，正常情况下不会包含表示三元组关系的 edge context。`sample_post_process` 只补 EOS，
不会重新打乱 node/edge sample 的总顺序。

这意味着本次 Stage-1 graph-memory 训练很可能只见到类似“图中存在节点 X”的声明，
没有见到“X 通过关系 R 指向 Y”的图事实。Stage-2 仍使用 64 个 train QA，因此 adapter
可能学习了关系分类任务或训练问答，但本次 smoke 不能验证图边知识已被参数化存储。

这也是 correct adapter 仅小幅高于 none 的一个直接候选解释。该问题必须在 Pilot 前修复：
cap 应在 node/edge 分层后采样，至少保证 edge context 占主要比例，并记录最终 node/edge
样本数、唯一关系覆盖率和与 QA 涉及实体/关系的覆盖率。

## 6. K=2 塌缩的机制判断

### 6.1 首选解释：重复执行同一 decoder block 导致表征退化

因为 no-adapter 条件也从 26.04% 降至 1.04%，塌缩不依赖 LoRA 内容。更可能的问题在：

- 同一预训练 decoder layer 被重复应用后产生分布漂移；
- 该 block 并未天然形成稳定的迭代算子；
- residual magnitude、RMSNorm 或 attention/MLP 更新在第二轮过度改写；
- K 改变生成分布和答案格式，产生候选外输出或长解释文本；
- 训练虽使用 K=2，但只有被选 executor layer 上的 LoRA 可训练，未必足以把基座 block
  约束成稳定 recurrence。

### 6.2 当前证据不支持“第二轮执行了错误的第二跳”

NELL23K 任务是关系预测，structural distance 并不等同于问题需要的组合推理步数。本次也
没有 frontier probe、step-level supervision 或 causal patch。因此，K=2 下降只能被描述为
**recurrent execution failure**，不能解释成逐跳推理走错。

### 6.3 需要补充的输出诊断

按 adapter × K 分别统计：

- empty response rate；
- candidate-exact rate；
- candidate-out-of-set rate；
- 平均生成 token 数；
- EOS 命中率；
- 重复 token / 重复短语比例；
- K=1 正确但 K=2 错误的 28 个样本的输出转移类型。

这些指标可以区分“内部表征被破坏”和“答案格式/生成长度退化”。

## 7. 汇总数字一致性检查

以下计数内部一致：

- 96 个 eval questions = 32 validation + 64 test；
- 96 × 2 K × 2 controls = 384 predictions；
- validation/test 不可达问题共 2+7=9；
- 9 × 2 K × 2 controls = 36 unknown-hop predictions；
- 384-36=348 known-hop predictions。

但报告的“首末 structured log event 约 87 秒”不能作为总 wall time。代码逐样本串行调用
`model.generate`；四个条件报告的平均延迟乘以各自 96 个样本，合计约 218.5 秒，尚未
包含训练、模型重载和分析。更可能是 structured-event 时间跨度没有覆盖完整推理阶段。
正式实验应使用 `/usr/bin/time -v` 或在 runner 最外层写入 monotonic start/end。

此外，correct K=2 的平均延迟略低于 K=1，说明当前 latency 强烈受生成 token 数和 warmup
影响。未报告生成 token 数与 token/s 前，不能用这些延迟声称 recurrence 的成本特性。

## 8. Validation/Test 使用问题

runner 将 validation 和 test 合并推理，汇总日志也报告合并后的 96-question accuracy。
虽然 `split_k_adapter_accuracy.csv` 已生成，但没有随远程提交保存，也没有在 smoke 记录中
列出。

后续应执行：

1. validation 只用于选择 K、parser 和诊断阈值；
2. test 在配置冻结后单独报告；
3. 主表禁止把 validation+test 合并为一个准确率；
4. 每个 split 保存 exact counts，而不只保存四舍五入后的比例。

本次 parser 是在 `_05` 的零分结果后修复，并以 `_06` 重跑，这是合理的工程修复；但 parser
规则应从现在起冻结，并在 Pilot 前增加候选内/候选外单测，避免继续根据 test 输出调整。

## 9. 对预注册假设和停止规则的判定

| 项目 | 当前判定 | 原因 |
|---|---|---|
| H1 Hop–Recurrence Alignment | 不支持 | Spearman=0.0912；K=2 全局塌缩 |
| H3 Length Extrapolation | 未测试 | 没有严格 train/test hop 隔离，也未评估 K=3/4 |
| H5 Storage–Execution Separation | 证据不足 | correct 仅小幅高于 none；context cap 疑似无 edge facts |
| H6 Adaptive Computation | 未测试 | 只有固定 K=1/2 |
| “Recurrence 有效” | 不成立 | 缺少 Original GRIP/More-QA/ComputeMatch，且 K=2 更差 |
| “逐跳执行” | 不成立 | 无 alignment、probe 和 causal intervention 联合证据 |

按照 `design/decision_rules.md`，当前符合“增加 recurrence 持续损害样本”的机制诊断条件，
因此应暂停扩大实验，而不是直接运行 512/128/512 Pilot。

## 10. 下一步：Diagnostic Smoke v1.1.1

### P0：保存可审计结果

将以下小型文件提交到远程，而不是只保存手写汇总：

- `analysis/summary.json`；
- `analysis/k_adapter_accuracy.csv`；
- `analysis/split_k_adapter_accuracy.csv`；
- `analysis/hop_k_accuracy.csv`；
- 去除 hidden states 后的 `predictions.jsonl`，或至少保存逐样本
  `question_id/split/hop/K/control/target/response/correct/latency/generated_tokens`。

### P0：修复 context cap

使用分层且确定性的采样，例如：

- edge context 224；
- node context 32；
- edge 按 relation 分层，优先覆盖 64 个 train QA 涉及的 relation/entity；
- 保存 `context_sampling_manifest.json`。

不要继续使用对 node-first 列表直接取前 256 项的方式。

### P0：运行 2×2 recurrence cross

保持同一数据和 seed，运行：

| Train depth | Eval K |
|---:|---:|
| 1 | 1 |
| 1 | 2 |
| 2 | 1 |
| 2 | 2 |

并保留 correct/none。该交叉实验用于区分：

- 训练深度不匹配；
- 重复 block 的固有不稳定；
- adapter 只适配某一执行深度；
- K=2 生成格式退化。

### P1：增加稳定性干预

若 K=2 在 no-adapter 下仍塌缩，再比较：

1. recurrent update residual scaling：`h <- h + alpha * delta`；
2. 每轮独立 RMSNorm；
3. recurrence gate；
4. K-aware position/step embedding；
5. ComputeMatch：不共享 block、但匹配 FLOPs。

### Pilot 启动门槛

只有同时满足以下条件后再运行 512/128/512：

1. cap 后的 context 明确包含 edge facts，并有覆盖率 manifest；
2. K=2 不再使 correct 和 none 同时塌缩到随机线以下；
3. correct adapter 相对 none 在 validation 上出现稳定的配对增益；
4. validation/test 分开输出；
5. 原始逐样本结果可审计；
6. parser 与评分规则冻结。

## 11. 最终决策

**当前决策：工程 smoke 通过，研究假设 smoke 未通过；不启动现有两小时 Pilot。**

下一次运行应是修复 context sampler 后的 v1.1.1 diagnostic smoke，而不是扩大 QA 数量。
当前最有价值的发现不是“RecurrentGRIP 已提升 NELL23K”，而是：

> 在当前单层共享 block 实现中，第二次 recurrence 对 correct/none 两种条件都造成近似幅度的
> 性能塌缩；同时 smoke 的 context cap 很可能没有注入 edge facts。因此必须先分离
> storage failure 与 execution failure，再讨论递归深度和图 hop 的关系。
