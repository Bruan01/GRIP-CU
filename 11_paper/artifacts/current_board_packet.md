# Current board packet

## 主线决策

- 研究问题：在生成式知识图谱关系候选排序中，负样本如何同时具备结构相关性、
  模型边界性和可审计性？
- 工作题目：开放世界知识图谱关系候选排序的可审计结构反事实对比学习。
- 范围决策：CLEGR 是图级组合问答，答案候选与 Judge Accuracy 均不同于关系候选排序；排除于本文，不报告为 SC-CL 的负结果或迁移结果。
- incumbent：Original GRIP/B1 generation-only。
- mainline：Structure-Counterfactual-9 + candidate InfoNCE。
- controls：Random-9 + InfoNCE；ErrorMemory-9 + InfoNCE（辅助）。
- 不纳入：递归执行器、0.5B smoke、RCCGRIP pairwise margin 旧结果。

## 当前可提交的 claim

当前是“单种子实测 + 未完成协议”的二区期刊导向中期稿，尚未达到投稿成熟度。NELL23K 的 SC-CL seed=2026 具有原始
预测、冻结 bank、wiring、manifest 和 full candidate rerank，可报告为单次可复核结果；
FB15K237-2 的历史 InfoNCE-i0 只作为不同训练契约下的辅助参照。五种子、匹配随机对照、
机制消融和 FB 当前 SC-CL 结果仍不能写成已完成证据，具体边界见 `results_status.md`。

## 下一步状态

1. 以现有 NELL 实测结果完成中文期刊导向中期稿和 LaTeX 编译；当前 ICLR 模板仅作工作排版模板，最终按选定期刊要求重排。
2. 补齐 NELL 的 B1 full candidate、Random-9、频率匹配和归因消融。
3. 在 FB15K237-2 重新按当前 SC-CL 契约执行 bank freeze、wiring audit 和主对照。
4. 执行五种子与假阴性审计；只用原始输出和 manifest 扩大主张范围。

## 内部留档，不进入本文

- 用户报告一项 CLEGR 异机运行比原论文高 +0.415；该数据集和结果不进入当前 SC-CL 论文。异机方法与数值证据尚未同步，不用于论文结论。
