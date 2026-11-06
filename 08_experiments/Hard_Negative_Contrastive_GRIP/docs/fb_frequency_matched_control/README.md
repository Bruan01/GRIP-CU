# FB关系频率匹配随机负例对照

本轮只新增一组Frequency-matched Random-9。目标是检验：在已有Full配置下，仅保持负例的训练频率桶分布，能否复现Full的性能。既有Full/Random/Profile/Violation均直接复用，不重训。

## 事前固定设计

- 频率仅为冻结train.txt中关系的边数；固定桶为`floor(log2(count))`，对应`[2^b, 2^(b+1)-1]`。不是根据测试效果选择桶数或边界。
- 对每个训练问题，保留原Full九个负例逐槽位的频率桶；同桶、同train-known过滤合法池内均匀无放回随机抽取。允许抽到原Full候选，禁止人为排除它们。
- 桶配额和槽位桶必须精确一致；不要求桶内具体边数相同。任何不足拒绝继续，无跨桶fallback。Full本身来自同合法池，因此可行性有直接来源；小池被迫选回原候选的槽位另行记录。
- 不按B1分数或画像相似度选择关系；这两项仅在候选固定后用于诊断。该对照不匹配B1困难度，也不单独隔离结构作用。
- FB15K237-2，seed2026，12,136条训练QA、3,109条候选资格、K=9；同Stage1、InfoNCE、LoRA和优化器。实际32步，保留原5-epoch/120步线性调度。第32步预先固定，只做一次10,000题完整直接解码测试；不做新完整验证、HPO或重复种子。
- 配置原先针对Full验证选定；464条train-pair QA修复及历史测试暴露保留。单次观测不等于多seed稳定性或各方法独立调参后的最优比较。

## 已完成CPU准备

全部3,109题Full候选顺序精确重放；新组3,109题槽位频率桶全部匹配，0次fallback。3,109题候选集合均改变，平均与Full共享0.6346/9个候选；27,981个抽样槽位中82个因桶池大小等于配额而被迫选定。QA文本、生成前缀、目标、候选资格和逐题损失权重不变。准备审计不是新模型性能。

## 执行与停止

入口为`scripts/run_fb_frequency_matched_control.py`，默认prepare不加载模型。prepare/check后使用独立冻结代码的run顺序执行训练、固定模型、一次测试和报告。状态每20秒落盘；进程出错保留现场并停止，不自动重试、重采bank或扩大预算。完成五方法表后停止，不继续其他训练。

本机单张RTX3090 24GiB；guardenv Python；Linux运行盘准备时剩余约778GiB。新增一组预计约6小时，实际时间以阶段日志为准。启动回执与实时目录见`RUN_RECORD.json`。

## 后续绘图和论文产物

Linux运行目录：`/home/kieran/grip-cu-runs/fb_frequency_matched_control_20261106_seed2026`。

| 路径 | 用途 |
|---|---|
| `contract.json`、`inputs/`、`reference/`、`code_snapshot/` | 冻结设计、数据、参照和代码SHA；借用输入源目录须一起保留 |
| `bank/relation_frequencies.csv` | 关系train频次和固定桶，用于复现分桶 |
| `bank/frequency_matched.jsonl`、`frequency_audit.jsonl` | 原始候选、逐题桶配额/池大小/强制槽位/抽样概率和过滤证据 |
| `bank/selected_candidates.csv` | Full与新组逐候选频次、结构相似度、B1违例度；用于频率/困难度分布图 |
| `bank/diagnostics.csv`、`frequency_bin_distribution.csv` | 逐题差异与聚合频率桶分布，显示匹配与桶内残差 |
| `training/trainer_listed/checkpoint-{8,16,24,32}/` | 完整adapter、optimizer、scheduler、RNG及trainer_state；不自动清理 |
| `selection.json`、`evaluation/test/` | 测试前固定权重身份；逐题原始response、正确标记和summary |
| `logs/`、`controller.log`、`events.jsonl`、`resources.jsonl`、`environment_*.json` | 命令、阶段时间、训练/测试日志、显存/功率与依赖环境 |
| `paper/training_history.csv` | 所有训练loss字段、步数和学习率，用于学习曲线 |
| `paper/metrics.csv`、`paired_five_methods.csv`、两组`paired_*.csv` | 五方法汇总、同题判定及纠错/退化/净变化 |
| `paper/table.tex`、`RESULTS.md`、`result.json`、`artifact_manifest.json` | 可写论文的表格、限定解释、结构化结果与全量文件哈希 |

结束后自动导出到`11_paper/artifacts/fb_frequency_matched_control_seed2026/`。原始逐候选大表及模型留在Linux，论文目录保存绘图所需轻量表与来源索引。

## 结果写回规则

关联`11_paper/paper_outline.md`的C2，实验条目`SC_CTRL_FREQMATCH`，实验节主文频率控制表，`paper_role=claim-carrying`。

- Full更高：训练频率桶匹配本身不足以复现Full表现；仍不能直接把差异归因于结构语义，因为B1反馈也改变。
- 持平/反超：频率可能解释相当部分收益，应收窄候选选择主张。不能改桶、重采bank或换seed挽救叙事。
- 不论方向，都报告与Full、Uniform Random的同题差异；不宣称精确频次已完全控制、统计显著或跨seed稳定。

下一步是完成本组固定评测、审计后写回论文；训练结束前不填写新EM。
