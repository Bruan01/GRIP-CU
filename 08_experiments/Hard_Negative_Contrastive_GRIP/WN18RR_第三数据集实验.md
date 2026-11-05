# WN18RR 第三数据集：首轮结果与论文对照

更新：2026-11-05。首轮 seed2026 队列状态为 COMPLETE，四分支训练、完整验证、验证选模和完整测试均已结束。下文保留 11 月 2 日准备与启动历史。

## 首轮完整结果（2026-11-05）

同骨干论文参考为 GRIP-Qwen2.5-7B 的 64.30%，来源为 [GRIP arXiv:2511.07457v2](https://arxiv.org/html/2511.07457v2) Table 2；§5.2 说明 WN 指标为五次运行均值。本轮为 seed2026 单次、固定三轮的同条件四分支比较。

| 配置 | 验证 official_em（3,034 题） | 测试 official_em（3,134 题） | 测试正确题数 | 较论文（百分点） | 较本地 B1（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 论文 GRIP-Qwen2.5-7B | — | 64.30% | — | 0 | — |
| B1 纯生成 | 53.2960% | 55.1372% | 1,728 | −9.1628 | 0 |
| **Soft-Mix 1.0（验证集选定）** | **88.9255%** | **88.8322%** | **2,784** | **+24.5322** | **+33.6950** |
| Calibrated 1.0 | 76.6645% | 77.1538% | 2,418 | +12.8538 | +22.0166 |
| Calibrated 0.5 | 63.4806% | 66.0498% | 2,070 | +1.7498 | +10.9126 |

三个对比训练配置的单次测试数值均超过同骨干论文参考。主方法保持为测试前由验证集选定的 Soft-Mix 1.0；相对本地 B1 净多答对 1,056 题。单次超过公开值已经达成，五种子稳定性和协议核验继续按 [冲刺计划](CCFB_冲刺计划.md) 第 3.5、6 节推进。

论文 Table 11 另报告 GRIP-Qwen3-8B 的 WN 为 68.48%，GRIP-Qwen3-14B 为 84.68%。Soft-Mix 本次相对后者高 4.1522 个百分点，这是不同骨干之间的数值参考；主比较仍使用同 Qwen2.5-7B 的 64.30%。

本轮 S2 固定 3 轮、无早停；论文 Table 13 为 WN 最多 5 轮、损失阈值 0.15 早停。候选顺序和源文件末行处理的差异见后文。论文五次均值、本轮单次结果、本地 B1 的复现差距分别保留，不把本轮称为严格同协议的五次复现。

### 结果证据与逐题核对

原始运行目录为 `results/runs/wn18rr_three_configs_seed2026/`，保留以下记录：

- `experiment_plan.json`：本轮输入、代码哈希与命令；四分支均固定 3 轮，LoRA r24/alpha48。
- `selection.json`：主方法为 `soft_mix_1`，`test_seen=false`；队列冻结验证选择后才执行本轮四组测试。
- `comparison.json`、`run_status.json`：四分支汇总和 COMPLETE 状态。
- `b1_seed2026/b1/summary.json` 及三个方法的 `listed/summary.json`：完整验证结果。
- `test/<分支>/summary.json`、`predictions_correct.jsonl`：完整测试汇总与逐题输出。分支名为 `b1_seed2026`、`soft_mix_1`、`calibrated_1`、`calibrated_05`。

2026-11-05 已重新核对四组逐题预测：每组均为 3,134 个唯一 ID，ID 集均为 `wn18rr:test:0` 至 `wn18rr:test:3133`，四组集合完全相同；从 `official_correct` 重算的准确率与各自汇总一致。完整预测和最终模型留在服务器，不随本文提交到 Git。

### 中断修复与对比损失范围

11 月 4 日恢复时复用已完成的任务、S1 和 B1，随后完成打分与正式 S2。本轮任务含 163,394 条 context、19,782 条 QA；全部 QA 保留生成损失，2,697 条关系 QA 使用候选对比损失。79 条可映射关系 QA 的完整实体身份无法唯一确认，因此以显式零权重清单项保留；其他未匹配 QA 按原规则只参与生成。

修复提交为 `4d627df`（打分资格）和 `c08ca31`（零权重清单接线）。原冻结计划、恢复依据、资格清单和全量接线证据分别保存在运行目录的 `experiment_plan.before_identity_fix_20261104.json`、`experiment_plan.before_exclusion_wiring_20261104.json`、`recovery_20261104.json`、`recovery_exclusion_wiring_20261104.json`、`confusion/qa_eligibility.json` 和 `wiring_identity_fix_20261104.json`。

WN 的有效池最多 10 个负例，当前每条可对比 QA 实际使用 9 个。Calibrated 1.0 在这些 QA 上的权重均值为 0.500390、标准差为 0.000748，接近常数 0.5；因此不能仅凭其与 Soft-Mix 的性能差异解释按题校准作用。匹配平均权重的常数对照已列入冲刺计划，尚未执行。

## 原始准备与启动记录（2026-11-02）

继续预定的四分支比较：同条件纯生成B1、Soft-Mix 1.0、Calibrated 1.0、Calibrated 0.5。首轮seed=2026，重新生成WN训练任务、训练共享S1、用本轮B1计算训练关系混淆评分，各方法使用相同冻结负例。原始数据下载与预处理已完成；真实数据哈希、划分数量和实验计划dry-run通过，相关CPU测试22项通过。2026-11-02北京时间14:35已通过项目入口启动tmux队列，初始阶段为tasks（GPU生成训练任务），完成后自动串行训练共享S1、B1和三个配置，再验证选模并测试。

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

用户提供的 `wn18rr/tasks/grip_wn18rr_tasks.json` 是已生成的训练样本，包含163662条context和12334条QA，并非已训练模型。该文件记录验证集参考、未启用节点属性QA以及指令格式context，与本轮协议不同。复用审核还发现context与summary合并保存，缺少逐条采样来源及本轮抽样开关证据；仅去除聊天模板不能确认采样协议一致。本轮复用已校验的原始数据和新预处理文件，重新生成训练任务；旧任务保留原样作为历史产物。

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

以下为2026-11-02启动时的空间处理历史：首次训练要求至少45 GiB空闲，各阶段要求至少10 GiB；当时准备阶段仅约13 GiB可用。经用户明确同意，执行pip cache purge（3406个缓存文件，约16.1 GiB），并删除已完成FB实验的五个中间断点目录（约25.5 GiB）：`fb15k237_three_configs_seed2026/trainer_s1`、`b1_seed2026/trainer_b1`及三个方法各自的`trainer_listed`，后三类路径均相对于该FB运行目录。清理后磁盘约54 GiB可用，满足当时首次空间门槛。共享S1、S1优化器、四个最终adapter、任务、冻结负例、预测、汇总和日志保留；仅失去从已删除的FB中间checkpoint恢复的能力。当次未清理其他实验或Hugging Face/uv缓存。随后WN从任务生成阶段启动。当前完成状态见本文开头；11月5日另一批旧checkpoint清理的完整记录见冲刺计划，WN本轮产物未受影响。
