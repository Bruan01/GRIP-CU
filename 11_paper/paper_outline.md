# Paper Outline

更新日期：2026-10-28；主线：结构反事实对比学习（SC-CL）。

## paper_view

- 论文类型：面向开放世界关系候选排序的实证方法论文。
- 投稿目标：二区期刊；具体期刊和分区体系待确定，当前 ICLR 模板仅作工作稿排版。
- 工作题目：开放世界知识图谱关系候选排序的可审计结构反事实对比学习。
- 核心论点：生成式关系候选学习需要负例既靠近目标决策边界，又可追溯、可过滤；SC-CL 用训练图关系画像和冻结 B1 混淆信号构建候选，并通过 train-only bank、known-pair filter 与逐候选记录实现来源追溯。
- 核心洞见：开放世界条件下“难”与“真负”不是同一概念。候选构造的可审计性应与排序收益一起报告，且闭集候选指标不能替代自由生成指标。
- 研究缺口：随机候选可能信息量不足；单靠模型高分选 hard negatives 会把暂时混淆与潜在真关系混为一谈。本文聚焦生成式关系候选学习中候选来源、边界信号和过滤的组合设计。
- 方法：关系结构画像 + 冻结 B1 score violation + 均匀探索下限；对 train-known 关系过滤；冻结 K=9 bank；在原始 GRIP 候选级 InfoNCE 阶段使用，保留原生成损失与提示。
- 核心结果：NELL23K seed=2026 的全量候选重排 H@1/MRR=91.52%/95.27%；同候选前 200 题的 SC-CL/B1 MRR=94.96%/93.36%，H@1=91.00%/88.50%。直接解码结果独立完整报告。
- 范围：固定实体对上的关系候选排序。CLEGR-Facts/Reasoning 的图级 QA 不纳入本文；不把不适配的 CLEGR 结果写成 SC-CL 正面或负面证据，用户报告的 +0.415 仅留内部状态。
- 读者 takeaway：开放世界知识图谱中的候选学习应同时报告排序效用、负例来源、过滤规则和未知真值风险。

### 方法主张、待验证假设与证据门

| ID | 主张或假设 | 当前证据 | 投稿前证据门 / 证伪条件 |
|---|---|---|---|
| C1 | SC-CL 提供可复现、train-only 的关系候选构造与审计协议 | NELL seed=2026 的 bank、wiring、manifest 和 hash | 任一 split 泄漏、known-pair 冲突或 question-ID 不对齐即否定该 run 的协议有效性 |
| C2 | 待验证假设：SC-CL 候选信号优于同预算随机候选 | 当前尚无完整 Random-9 与 frequency-matched 配对比较 | matched controls 不优于随机或配对差异不稳定，则收窄/撤回收益主张 |
| C3 | 待验证假设：该方法在多种关系 KG 上稳定有效 | 当前尚未建立 | 五种子、跨图迁移和风险审计完成前不得声称稳定或泛化 |

### 叙事顺序

1. 问题：生成式关系候选学习缺少信息量与假阴性风险兼顾的负例构造。
2. 缺口：关系结构相关性、当前模型混淆度、train-known 过滤和来源审计未在本文设定中形成一个冻结协议。
3. 方法：结构画像与冻结 B1 violation 混合采样，保留均匀下限并冻结 bank。
4. 结果：先展示全量排序绝对值和同候选前 200 题的配对差异，再呈现逐题变化、直接解码、候选来源与成本。
5. 解释：讨论双信号和固定候选的设计含义；实际比较条件集中说明；结论强化方法与排序结果。

### 评估与分析计划

- 数据：主矩阵为 NELL23K、FB15K237-2、WN18RR、CoDEx-M；CoDEx-M 的官方 verified triple negatives 单独作风险审计，不映射成关系候选负例。canonical FB15K-237、YAGO3-10、CoDEx-S/L 暂缓。
- 基线：B1、Random-9、frequency-matched Random-9、Profile-only、Violation-only、ErrorMemory-9、SC-CL-9；外部 link-prediction 结果单独呈现。
- 指标：自由生成 EM/OOV/NLL；闭集 EM、MRR、Hits、margin；候选覆盖与 false-negative audit；资源成本。
- 分析：候选信号消融；关系频率捷径；K/rho/tau 敏感性；arity/degree/sparsity 切片；多种子 paired CI；开放世界正例冲突；leakage/wiring；资源—收益。
- 失败解释：若排序增益不超过频率匹配随机，撤回组合信号优势；若闭集改善伴随自由生成退化，只保留候选排序结论；若审计门失败，该 run 不进入主结果。

## evidence_view

- 成熟度：当前为单种子实测与未完成对照并存的研究中期稿，具体投稿证据门维持不变。

- 已观察：NELL23K seed=2026 的 SC-CL 训练、冻结 bank、wiring、manifest、full test rerank；B1 与 SC-CL 仅在前 200 个相同候选问题上严格配对。FB15K237-2 的 InfoNCE-i0 是历史不同契约结果。WN18RR 完成单种子 bank、训练和一次全测试直接解码，但尚未完成完整候选重排。
- 尚缺：NELL matched-budget B1/Random-9/frequency-matched controls、机制消融、五种子、完整 false-negative audit；FB 当前 SC-CL 主运行；WN18RR 的完整重排、对照、多种子和 B1 复现解释；CoDEx-M 的 Stage1/Stage2 task、adapter、bank 与模型运行。
- 排除：CLEGR-Facts/Reasoning 是不同任务定义；本地低预算 EM 和异机 +0.415 均不作为本文 SC-CL 证据。
- 投稿门：在上述核心公平对照、统计与风险审计达到预设门槛前，稿件定位为研究中期稿，不声称达到任何期刊分区的投稿成熟度。
