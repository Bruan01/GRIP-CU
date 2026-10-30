# Reviewer-first pass：SC-CL

这是对当前论文中期稿的模拟 ICLR 审阅摘要；它不是正式审稿意见。NELL23K 的单种子
结果按原始文件审计后纳入，但不把历史 FB InfoNCE-i0 或单次结果包装成统一主结论。

## 八个最可能的拒稿问题

1. **归因**：SC-CL 相对 Random-9 的差异是否来自结构反事实，而不是候选损失、关系
   频率或额外训练预算？
2. **泛化**：NELL23K 和 FB15K237-2 之外，方法是否能迁移到 WN18RR、canonical
   FB15K-237、YAGO3-10？
3. **假阴性**：train-known 过滤是否仍把持出真实关系或开放世界未知事实当成负例？
4. **泄漏**：bank 是否读取 valid/test、错误的 adapter、重复候选或不对齐的 question ID？
5. **稳定性**：结果是否依赖 seed、K、rho 或 tau_s 的单点选择？
6. **生成副作用**：闭集 MRR 上升是否以 EM、OOV 或 NLL 恶化为代价？
7. **成本**：B1 全关系 scoring、bank 写入和额外训练是否值得？
8. **外部定位**：与 SANS、RotatE、ConvE、SimKGC 等工作的比较是否使用了可比协议？

## 当前回应

正文已经把上述问题映射为 R1--R8，新增频率匹配随机、Profile-only、Violation-only、
五种子、K/rho/tau 敏感性、arity/degree/profile-sparsity 切片、CoDEx-S/M 审计以及
资源报告。数据集分成：

- 本地核心：NELL23K、FB15K237-2；
- 标准迁移：WN18RR、FB15K-237、YAGO3-10；
- 假阴性压力：CoDEx-S、CoDEx-M，CoDEx-L 为可选。

## 仍未解决的证据问题

- NELL 已完成一次正式 GPU 训练、冻结 bank 和 full candidate rerank，但尚未完成同协议
  的完整基线与五种子评估；
- 标准迁移和 CoDEx 任务还需要转换、关系文本映射与 adapter 检查；
- FB 当前 SC-CL、假阴性和跨数据集数字仍是 planned；NELL 资源数字只有单次记录；
- 因此当前推荐仍为“大修后再审”，不是“实验已完成”。
