# WN18RR 第三数据集：运行准备（2026-11-02）

## 本轮范围与论文对照

继续预定的四分支比较：同条件纯生成B1、Soft-Mix 1.0、Calibrated 1.0、Calibrated 0.5。首轮seed=2026，重新生成WN训练任务、训练共享S1、用本轮B1计算训练关系混淆评分，各方法使用相同冻结负例。原始数据下载与预处理已完成；真实数据哈希、划分数量和实验计划dry-run通过，相关CPU测试22项通过。当前为运行准备，GPU任务尚未启动。

论文Table 2的GRIP-Qwen-7B在WN18RR上为64.30%，这是5次运行均值。首轮单seed只能报告单次结果，不能当作5次均值复现。默认沿用本项目已确定的S2固定3轮、无早停比较；论文为最多5轮、损失阈值0.15早停，因此本轮是核心设置对齐的固定预算比较，不称为严格论文协议复现。后续多种子及论文停止规则对照另作明确计划，不自动追加长训。

## 数据与评测

- 原始来源：官方GRIP处理器指定的 `WFRaain/TAG_datasets/WN18RR`，读取 `entity2text.txt`、`train.txt`、`valid.txt`、`test.txt`。
- 标准划分：train 86835、validation 3034、test 3134，11种关系。图和任务参考只来自train。
- 实体采用官方第二个tab字段，关系文字采用去掉起始下划线后以空格连接；生成 `rel2text.json` 用于评分和训练。
- 每题候选为全部11个训练关系文字的随机排列；完整评测3034条validation和3134条test，不能套用FB的10选1/10000题子采样。
- 使用官方word node题面；候选词表排序后以固定种子排列，记录文件哈希，不声称逐字复现官方依赖set顺序的历史候选排列。
- 保留全部非空源数据行，不无条件丢末行；预处理明确记录这一处理规则与官方处理器的差异。
- 验证集先完成四分支评测，在三个方法中按 `validation.official_em` 选定配置，平分取声明顺序较前者。冻结 `selection.json` 后才跑四分支test。自由生成、贪心解码、1000-token上限、官方答案解析精确匹配。

## 训练与负例

Qwen2.5-7B-Instruct，MLP-LoRA r24/alpha48，down/up/gate projection；S1最多1轮、损失阈值0.15；S2四分支均3轮、answer_only、继承S1优化器。batch=1、梯度累积512、学习率0.001、weight decay=0.0001、Adam epsilon=0.0001、max grad norm=1。任务配额Ns/Nr/Nc=6000/2000/8000，使用节点属性、均匀采样、rooted reasoning、纯文本context和训练实体节点。

三个方法共享冻结负例：6个均匀抽样加3个从Top9按混合概率抽样，rho=0.33。WN只有11个关系，按train过滤已知正关系后，有效负例池最多10个；不足9时使用全部有效负例，不重复抽样或塞回正例。因此WN的“困难子集”与全负例池可能接近，不能假定与FB/NELL相同的采样收益。Soft-Mix权重固定1；Calibrated为clip(1-0.5×Top9负例概率质量,0.25,1)，0.5配置再乘0.5。

评分按完整实体文字解析问答，将同名实体对应的训练关系取并集，避免空格、内含and或重名导致假负例漏过滤；歧义实体对不猜测。生成任务完成后仍须核查实际可匹配QA数量，零匹配会阻止对比实验继续。

## 2026-11-02 已有样本与重复文件清理

用户提供的 `wn18rr/tasks/grip_wn18rr_tasks.json` 是已生成的训练样本，包含163662条context和12334条QA，并非已训练模型。该文件记录验证集参考、未启用节点属性QA以及指令格式context，与本轮协议不同；保留原样供后续复用审核，不自动纳入当前训练。

经SHA-256逐一确认，`wn18rr/raw/` 的 `entity2text.txt`、`train.txt`、`valid.txt`、`test.txt` 与当前入口使用的 `data/raw_datasets/wn18rr/` 完全一致。按用户清理重复文件的要求，删除前者四份副本，保留后者及其下载报告、准备清单。共释放7445411字节（约7.10 MiB）。旧 `wn18rr/manifest.json` 保留为历史来源记录，其原始文件现在应从 `data/raw_datasets/wn18rr/` 获取。

旧 `wn18rr/processed/` 的验证/测试文件与新产物哈希不同，不能认定为重复，继续保留；旧任务、生成计划、`wn18rr.json`、模型及训练断点均未删除。此次清理不解决长训磁盘空间缺口。

## 准备、启动与恢复

从本实验目录执行（默认Python为项目grip-exp虚拟环境）：

```bash
# CPU下载、预处理和计划检查，不启动GPU任务
PREPARE_ONLY=1 bash configs/run_wn_three_configs.sh

# 已有数据时只校验数据/代码哈希和实验计划
DRY_RUN=1 bash configs/run_wn_three_configs.sh

# 空间满足后启动；再次执行自动恢复同目录
bash configs/run_wn_three_configs.sh
```

默认目录：`results/runs/wn18rr_three_configs_seed2026/`；默认tmux session：`wn-three-configs-2026`，连接命令 `tmux attach -t wn-three-configs-2026`。总日志为该目录 `run.log`，阶段日志为 `tasks.log`、`s1.log`、`b1.log`等。

所有长GPU任务经tmux；每10 optimizer steps保存checkpoint，保留2份；已完成阶段按命令及产物哈希复用，已有最终adapter跳过训练。任务生成、评分与预测使用现有续写机制。固定目录和运行锁避免并发重复；数据和代码哈希变化拒绝静默续用。

首次训练要求至少45 GiB空闲，各阶段要求至少10 GiB。准备时仅约13 GiB可用。2026-11-02经用户明确同意，执行pip cache purge（3406个缓存文件，约16.1 GiB），并删除已完成FB实验的五个中间断点目录（约25.5 GiB）：`fb15k237_three_configs_seed2026/trainer_s1`、`b1_seed2026/trainer_b1`及三个方法各自的`trainer_listed`，后三类路径均相对于该FB运行目录。清理后磁盘约54 GiB可用，已满足首次空间门槛。共享S1、S1优化器、四个最终adapter、任务、冻结负例、预测、汇总和日志保留；仅失去从已删除的FB中间checkpoint恢复的能力。未清理其他实验或Hugging Face/uv缓存。WN仍处于准备阶段，已有旧任务的复用审核尚未完成。原始数据、任务、模型和完整预测保留本地；提交脚本、测试、协议与小型准备清单。
