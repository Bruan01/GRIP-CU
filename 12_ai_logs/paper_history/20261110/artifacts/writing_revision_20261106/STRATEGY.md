# 2026-11-06 写作整合与后续分析策略（编辑前）

选定对象：`11_paper/main/main.tex`及其sections；教学材料与历史快照不改。
本次为已有实测结果整合，不增加文献主张或引用。write skill所列远程paper-contract/memory工具当前不可用，以本地outline、ledger、matrix和审计产物为准。

| 问题类别 | 当前问题 | 本次唯一处理动作 |
|---|---|---|
| manuscript-mapping | 最新NELL/CoDEx/FB完整直接解码及FB五方法控制未进入正式稿 | add result to main text：三数据集主表、FB五方法表、逐题配对变化 |
| unsupported writing | 均匀随机或单信号差异不足以支持超出频率分布的独有结构机制 | downgrade claim：摘要、引言、讨论与结论同步保留频率替代解释 |
| narrative | 高频率匹配分数的含义容易误读 | add result to main text：明确定义Full诱导的逐题频率配额，区分该条件对照和独立通用频率基线 |
| evidence gap | 净差7题可能掩盖频率层级间相反变化 | run analysis：只用train关系频次定义三分位层级，复用五方法预测；无新训练、HPO或种子 |
| metadata drift | JSON矩阵、section plan、README仍以历史NELL为唯一主结果 | repair paper contract：同步已完成/未完成状态、section/item/claim映射 |
| manuscript-mapping | 历史NELL重排、旧直接解码与成本来自不同模型 | move result to appendix：保留原数值和身份，正文提供明确桥接 |
| reproducibility | 现有画像公式与代码对度统计的log变换及分母常数不一致 | repair paper contract：按已冻结实现校正公式与统计定义，不修改实验算法 |

论文论证顺序：可追溯候选构造 → 三数据集单seed配置结果 → FB受控候选替换 → 频率配额对照收窄解释 → 预先按train定义的频率分层误差分析。

后续分析的研究问题：Full与配额匹配对照总体EM接近，是否存在按训练关系频率分层的收益/退化抵消？
分组：按训练词表内各关系的train边数，以排序后的第ceil(R/3)、ceil(2R/3)个频次为两条边界，同频次不拆分；分界只由训练图决定。测试目标关系频次用于归组；不按任何测试正确率调边界。
输出：逐题归组、逐关系结果、五方法分层micro EM与macro-relation EM、Full−配额匹配及对其他方法的纠错/退化/净差；每组n和关系数；来源hash、参数、运行日志与LaTeX表。
限制：在总体结果已观察后的描述性分析；关系频率不是训练样本数或实体稀疏度的同义词；不做显著性、跨seed或因果主张，不筛选有利分组。全部层级和负差异并列报告。
停止：一次固定CPU分析；不依据结果开新GPU实验。若无分层收益，继续收窄，不重切分组。
