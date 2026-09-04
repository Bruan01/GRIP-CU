# E03 实验计划：Validation-First Entity-Constrained GRIP

## 1. 研究问题

> 在不访问 query-specific 图证据的条件下，train-KG 全局实体约束能否把 GRIP checkpoint 中已有但未被自由生成正确读出的实体信号转化为 semantic exact match？

Priority 完整审计表明自由生成的 valid-entity rate 已约为 80%–89%，公共前缀错误约为 0%–3%，因此不再预设“解码格式是主瓶颈”或自然获得 `≥5 pp` 涨点。

## 2. 假设与反证条件

- **H1 / semantic repair**：D1 不仅提高 valid-entity rate，还能将 invalid 或 valid-wrong D0 样本修复为正确实体。
- **H2 / non-destructive constraint**：D1 不应通过破坏原本正确答案换取局部收益。
- **H3 / composition robustness**：novel-composition 平均下降不得超过 1 pp。
- **反证条件**：两个 direct checkpoints 的平均 canonical EM 增益 `< +2 pp`，立即停止 decoder-primary 路线。

## 3. 冻结变量

- Qwen2.5-0.5B backbone/tokenizer；
- validation-selected LoRA checkpoints；
- exact-hop train/validation rows；
- answer-only prompt；
- train graph 20,789 entity vocabulary；
- 历史 D0 validation predictions 及其 SHA256；
- 所有 query 使用相同全局 trie。

唯一首轮干预为 `D0 free generation artifact → D1 global trie`。

## 4. 首轮比较

```text
split       = validation only
checkpoints = direct seed43 + direct seed44 + More-QA seed42
methods     = D0 artifact reuse + D1 global trie
```

- direct seed43/44 是 gate primary；
- More-QA seed42 是 reference diagnostic；
- joint seed43/44 是 secondary registry，不进入首轮 gate；
- Phase-A runner 不打开 test 文件。

## 5. 指标

### 主指标

- raw exact match；
- canonical entity exact match；
- valid/invalid entity rate；
- strict-prefix error rate。

### Error transitions

按 `task_id` 对齐 D0/D1：

- invalid → correct / wrong；
- valid-wrong → correct / wrong；
- correct → correct / wrong；
- wrong → wrong 总量。

### 分桶

- hop 1/2/3/4；
- seen/novel composition；
- train relation frequency；
- answer character/token length；
- answer token-prefix ambiguity。

## 6. Gate

输出 `PRELIMINARY_GO_D2` 当且仅当：

1. 至少两个 direct checkpoints 完成；
2. 平均 canonical EM 增益 `≥ +2 pp`；
3. 每个 direct checkpoint 的 raw/canonical 增益均非负；
4. novel-composition 平均增益 `≥ -1 pp`。

valid-rate 仅为解释指标。任一条件失败输出 `STOP_DECODER_PRIMARY`。

## 7. 条件式 D2 与 D3

- D2 仅在 `PRELIMINARY_GO_D2` 后运行 validation；sum/mean normalization 仍只由 validation 选择。
- D3 必须单独使用 `RUN_MODE=diagnostic`，其 gold-containing candidate pool 不进入主结果或 gate。
- 首轮与 D2 均不运行 test。

## 8. 官方 Phase B 锚点

机制验证通过后才创建新的官方实验版本：Qwen2.5-7B-Instruct、LoRA rank/alpha `4/8`、`down_proj/up_proj/gate_proj`、Context QA 8000、Reasoning QA 2000、Summarization 6000、Stage 1 一轮、Stage 2 最多十轮、effective batch 512、三随机种子。正式 test 只在官方协议与 checkpoint selection 冻结后打开。
