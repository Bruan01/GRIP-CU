# Decision Log: Select RecurrentGRIP

日期：2026-08-28

## 决策

将 RecurrentGRIP 设为当前第一研究主线。

## 被降级的方向

以下方向保留为 baseline、训练配方或消融，不作为主贡献：

- AutoMix-GRIP；
- EvalAlign-GRIP；
- HeteroDistill-GRIP；
- 普通数据配比；
- 单纯 loss 修改；
- 单纯扩大 LoRA rank。

## 比较过的三个机制方向

### RecurrentGRIP

- 来源：神经算法推理、共享递归计算；
- 优点：pilot 清晰、六数据集适配好、机制证据可做；
- 风险：可能被认为只是 recurrent Transformer 应用。

### SuccessorGRIP

- 来源：强化学习 successor representation、认知地图；
- 优点：数学故事强，适合动态图低秩更新；
- 风险：实现和验证复杂。

### CodeGRIP

- 来源：纠错码、factor graph、belief propagation；
- 优点：概念独特，适合鲁棒性；
- 风险：容易退化为普通 constraint regularization。

## 选择理由

RecurrentGRIP 最容易通过一个小规模实验快速证伪：

> 用短路径训练后，增加测试 recurrence 是否改善未见长路径？

它同时允许构建完整证据链：

```text
参数存储
→ 递归调用
→ frontier 表征
→ 因果干预
→ 长度外推
```

## 决策边界

第一轮只实现固定深度 recurrence。Successor representation 和 syndrome correction 不进入 v1，以避免概念堆叠。
