# 论文结果回写策略（2026-10-23）

## 证据状态

- NELL23K：当前 SC-CL 训练 run `nell23k_structure_counterfactual_3090_20261021_06_train` 已完成，seed=2026；冻结 bank 写入 3,253 行、每行 K=9，wiring 与 artifact manifest 通过。官方测试候选重排覆盖 4,943 条问题。
- NELL23K：原始 B1 自由生成覆盖 4,943 条；B1 与 SC-CL 的同候选列表配对候选评分目前只有前 200 条。
- FB15K237-2：B1 与历史 `InfoNCE-i0` 候选评分各覆盖 10,000 条，数据 hash 与当前本地 processed test 一致；该 i0 使用历史 `paired_candidates` 和不同 LoRA/训练配置，不属于当前 SC-CL bank。
- FB15K237-2：当前工作区没有可证明为 SC-CL 结构反事实 bank + Stage 2 的正式运行，因此不把 i0 结果写成 SC-CL 主结果。

## 修订动作

1. 将摘要、引言、实验和结论从“实验前设计稿”改为“单种子实测 + 未完成项并存”的中期论文稿。
2. 用可复核的 NELL23K 与 FB 候选排序数字替换核心表中的预设数字，并在表格中标注 full/prefix-200/historical provenance。
3. 删除或改写 WN18RR、YAGO、CoDEx、消融、敏感性和五种子表中的虚构数值；未完成部分只保留协议与状态。
4. 将 NELL 的 train-only bank、wiring、hash、训练步数、token 和测试候选评分成本写入审计与资源段落。
5. 生成一张仅展示已完成候选排序证据的观测图；不绘制跨数据集 SC-CL 增益图，直到 FB SC-CL 与同协议对照完成。

## 不可写入正文的结论

- 不能声称 SC-CL 已在 FB15K237-2 上完成或优于 B1。
- 不能把 NELL 的候选重排增益写成自由生成能力提升。
- 不能写五种子均值、显著性、频率匹配随机、ErrorMemory 或结构信号消融已完成。
- 不能把历史 InfoNCE-i0 的结果与当前 SC-CL 共享训练预算后直接排名。
