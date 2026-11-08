# CCF B 冲刺：运行准备与续跑

更新于 2026-11-08，北京时间。服务器项目目录：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
```

WN 与 CoDEx-M 首轮均已完成。NELL seed2026 同 3 轮 B1 已完成。WN seed2026 独立核验和 no-adapter validation 诊断通过；seed2027 四分支正在运行，当前阶段为 B1。[CoDEx-M 首轮成绩](CODEXM_第四数据集实验.md) · [WN18RR 首轮与核验](WN18RR_第三数据集实验.md)

## 11 月 8 日：WN 独立核验与 seed2027 启动

按冲刺计划第 3.5 节检查后，WN 原始拆分无重复三元组，train/validation/test 间无同一金三元组；processed_train/val/test 与完整评测文件中的图边均逐条等于 86,835 条 train 边。3,034/3,134 个评测题的来源 ID、实体文字、关系答案、11-way 候选和题面检查均通过。四分支逐题预测经项目官方解析器重算与 summary 完全一致。原始实体对重合（同向 7/8，反向 1,073/1,095）与候选位置准确率差异已写入 [WN 实验记录](WN18RR_第三数据集实验.md)。

无 adapter 的 `Qwen/Qwen2.5-7B-Instruct` 在完整 validation 上为 441/3,034（14.5353% official_em）；B1 为 53.2960%，Soft-Mix 为 88.9255%。推理约 898 秒，模型本地离线加载，无 adapter、无 test 评估。命令收据、哈希、逐题预测和 summary 保存在服务器 `results/runs/wn18rr_three_configs_seed2026/diagnostics/base_qwen_validation/`。

WN seed2027 已使用 `configs/run_ccfb_followup.sh` 启动，tmux 为 `ccfb-wn18rr-2027`，目录 `results/runs/ccfb/wn18rr/seed2027/`。输入检查通过，无缺失依赖或代码来源；启动时磁盘可用 46.143 GiB，单批最低门槛 45 GiB，动态单批估算 42.138 GiB，GPU 空闲。整份剩余矩阵约需 325.962 GiB，因此 seed2027 完成后先复查资源，不直接并行启动其他批次。启动记录见 `results/runs/ccfb_preparation_20261104/followups/launch_check_wn18rr_seed2027.json`。

## 11 月 7 日 NELL 同 3 轮 B1 与 CoDEx-M 复验准备

NELL seed2026 B1 于 22:22:43 启动，tmux 为 `ccfb-nell23k-2026`。22:24 核对到 GPU 进程与真实训练日志，已完成 1/117 个优化器步。运行目录为 `results/runs/ccfb/nell23k/seed2026/`。当前模型加载原有任务和共享 S1，继承 S1 优化器，训练 3 轮。已有 3 轮方法分支的 4,951 条验证记录与本次验证输入逐题一致。测试报告完整 4,944 题的 official_em 和论文口径 4,943 题的 official_harness_em。

启动前实测 15.43 GiB 可用空间。原双断点预算为 16.95 GiB，本批次使用每 10 步保存一个滚动断点。保存新断点时计入两个同时存在的文件体积，最终导出时计入一个断点加 adapter，取两者较大值，再加 10% 余量和 10 GiB 保留空间，门槛为 14.67 GiB。与原计划逐项对照，训练命令唯一变化为 `--save_total_limit 2` 改为 `1`。启动预检保存在 `results/runs/ccfb_preparation_20261104/followups/nell23k_seed2026_compact_preflight.json`。

恢复时使用同一入口与断点模式，启动器会检查已冻结的命令及输入。

```bash
DATASET=nell23k SEED=2026 CCFB_NELL_COMPACT=1 TMUX_SESSION=ccfb-nell23k-2026 \
  bash configs/run_ccfb_followup.sh
```

CoDEx-M seeds 2027、2028 已复用首轮任务、S1、教师评分和 Soft-Mix 1.0 的冻结选择，生成 6 份方法负例清单。每份全量接线覆盖 19,798 条生成 QA 和 2,758 条有效对比 QA，保留 202 条零权重项。11 个后续批次与 30 份方法清单全部通过复核，计划和文件哈希保存在 `results/runs/ccfb_preparation_20261104/followups/report.json` 与各批次 `sampler_preparation.json`。

复核旧清单时发现 Python 路径的等价写法导致身份误报。路径比较现在解析 Python 所在目录，保留不同虚拟环境的身份，同时逐项比较训练参数与数据、源码哈希。15 项相关测试通过。四分支复验仍按至少 45 GiB 的启动门槛排期。

## 11 月 5 日：后续主矩阵已准备的代码与数据

本节测量截点为 17:27，北京时间；17:28 完成实际 GPU 防重叠验证。当前 CoDEx-M 继续运行在 `tasks`，没有额外启动 GPU 批次；其冻结源码哈希全部一致。完整测量、产物哈希和验证记录见 [后续准备记录](results/archive/ccfb_followup_preparation_20261105.json)。

| 后续批次 | S2 数量 | 输入与训练清单 |
| --- | ---: | --- |
| NELL seed2026 同 3 轮 B1 | 1 | 原任务、共享 S1、S1 优化器和完整评测输入核验通过，B1 无需方法负例清单 |
| NELL seeds2027、2028 | 8 | 6 份方法负例清单已生成并通过完整训练器资产解析 |
| FB seeds2027、2028 | 8 | 6 份方法负例清单已生成并通过完整训练器资产解析 |
| WN seeds2027—2030 | 16 | 12 份方法负例清单已生成并通过完整训练器资产解析 |
| CoDEx-M seeds2027、2028 | 8 | 原始/正式输入核验通过；任务、S1、教师评分和验证选模需等待当前首轮产出 |

以上是 11 个后续批次、41 次 S2；加当前 CoDEx-M 首轮的 4 次，主矩阵共剩余 45 次。9 个后续批次输入就绪，另 2 个等待当前 CoDEx-M 首轮。该状态描述代码和输入，实际启动还要满足 GPU、磁盘和计划中的科学核验条件。

24 份方法清单及 8 份基础采样清单位于 `results/runs/ccfb/<dataset>/seed<seed>/`，合计生成 64 个文件，约 131.55 MiB。每个批次的 `sampler_preparation.json` 保存源输入、运行源码和输出哈希。新入口仍会执行既定冻结阶段并校验相同内容。

完整 QA 接线结果：NELL 为 19,808 条 QA / 3,030 条有效对比 QA；FB 为 19,845 / 3,021；WN 为 19,782 / 2,697，另 79 条显式零权重排除项。三方法每条有效 QA 均使用相同的 9 个负例，仅权重不同；新 seed 改变负例采样与 S2 随机数，任务、S1 和教师评分保持固定。清单再次核验时哈希一致，没有重新采样。

新增入口：

- `scripts/prepare_ccfb_followups.py`：CPU 预检、实际清单准备、测量磁盘预算和后续启动检查。
- `configs/run_ccfb_followup.sh`：调用容量与空闲 GPU 检查，再进入原 tmux 调度入口。
- `tests/test_ccfb_followups.py`：检查存储计算、当前目录保护、保留原 45 GiB 门槛及 GPU 占用拒绝。

在服务器本目录刷新准备记录：

```bash
export PYTHONPATH="../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp:src:scripts"
../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python \
  scripts/prepare_ccfb_followups.py --prepare_negatives
```

已准备的清单会检查输入身份和输出哈希后复用；遇到已启动的批次会拒绝改写。更新报告保存在 `results/runs/ccfb_preparation_20261104/followups/report.json`，逐批次 CPU 计划在其 `plans/` 下。CoDEx-M 首轮 COMPLETE 后重跑上述准备命令，才会生成其后续两个 seed 的清单。

当前 CoDEx-M 完成、补足空间并通过相应核验后，下一项启动命令为：

```bash
DATASET=nell23k SEED=2026 bash configs/run_ccfb_followup.sh
```

新增入口对这一 B1 批次要求至少 **16.95 GiB** 可用空间，包含新模型、两个 checkpoint、10% 余量和 10 GiB 保留空间，比原入口 12 GiB 的静态门槛更充分。其余四分支批次保留原先至少 45 GiB 的门槛，并取测量预算与原门槛中的较大值。17:28 对真实忙碌 GPU 的启动检查已按预期拒绝 NELL 启动，当前 CoDEx-M PID 不变。新增检查与相关原调度回归合计 **14 项通过**；此处为 CPU 工程验证。

## 11 月 5 日：内存、显存和磁盘核算

| 资源 | 实测 | 对后续工作的判断 |
| --- | --- | --- |
| 系统 RAM | 总量 31.26 GiB，可用 28.16 GiB；另有约 8 GiB swap | 同机已完成 NELL/FB/WN 的同骨干训练，继续按单 GPU 串行运行；当前内存余量充足 |
| GPU 显存 | RTX 3090，24 GiB；当前约用 15.61 GiB、空闲 7.95 GiB | 当前任务生成正常；这一读数不是 S2 峰值，后续沿用已跑通的 batch=1 与梯度检查点配置 |
| 磁盘 | 准备后可用 47.47 GiB | 当前 CoDEx-M 首轮估算需 44.81 GiB，余量约 2.66 GiB；后续主矩阵不足 |

系统 RAM 的“可用”包含可回收文件缓存，不能只看 `MemFree` 判断不足。本次没有并行启动第二个 GPU 任务。

磁盘单位取现有独立产物的 `du -B1` 实测：r4 最终 adapter / 单 checkpoint 为 2.0680 / 2.1246 GiB；r24 为 2.2090 / 2.5476 GiB。剩余 45 次 S2 包含 r4 的 21 次和 r24 的 24 次：

| 占用或策略 | 预计需要的新增可用空间 | 与当前 47.47 GiB 的关系 |
| --- | ---: | --- |
| 45 个最终模型 | 96.44 GiB | 尚未包含 checkpoint 和新 S1 |
| 每个 S2 保留两个 checkpoint | 另需 211.52 GiB | 当前代码会保留这些文件 |
| CoDEx-M 新共享 S1 及两个断点 | 另需 6.37 GiB | 任务尚在生成，按新产物估算 |
| 主矩阵全部保留，加 10% 余量和 15 GiB 保留空间 | **360.77 GiB** | **缺约 313.30 GiB** |
| 若以后将已完成的中间断点备份到其他文件系统，本机保留最终模型、结果与活动断点 | **131.83 GiB** | **仍缺约 84.36 GiB**；此为条件方案，本次没有执行归档或删除 |

上述主矩阵不含[冲刺计划](CCFB_冲刺计划.md)第 7 节机制包。最初最多 8 次机制 S2（r4 两次、r24 六次）与一次独立 WN S1，按相同保留规则另计约 **70.51 GiB**（含 10% 余量）；主矩阵与机制包合计约 **431.28 GiB** 可用空间，当前缺约 **383.81 GiB**。这是存储估算，机制包尚未由本次准备物化或启动；可选 WN 预算搜索和进一步重复另计。FB 时间匹配 B1 的实际训练时长还需根据观测成本确定。

原计划 150 GiB 的目标只适用于逐阶段核对并将已完成断点备份到其他磁盘的主矩阵策略。按当前每个实验原地保留两个 checkpoint 的行为，改用约 361 GiB 主矩阵、约 432 GiB 含最小机制包的容量要求。当前 CoDEx-M 结束后，先复测空间再排 NELL；不能将当前首轮满足启动门槛当作后续全部工作已经有空间。

## 11 月 4 日准备与恢复快照

本次恢复现有 WN 队列，准备 CoDEx-M 正式数据和后续运行入口。新 GPU 实验仍按冲刺计划顺序执行；本次只启动了中断 WN 的恢复。

## 当前 WN 队列

```bash
tmux attach -t wn-three-configs-2026
```

任务、S1、B1 已完成，恢复时验证其原始文件哈希并复用，从 `score` 继续。随后沿原入口完成负例冻结、Soft-Mix 1.0、Calibrated 1.0/0.5、验证选模和四分支测试。

中断原因是 79 条生成的关系问题改写了实体名字，无法唯一对应完整实体文本。打分修复会显式记录并排除这些 QA 的候选对比损失，全部 19,782 条 QA 继续参与生成损失；安全可识别的 2,697 条 QA 接受训练图正关系过滤。没有猜测实体或放行未过滤负例。

该批次中的恢复证据：

- `results/runs/wn18rr_three_configs_seed2026/recovery_20261104.json`
- `experiment_plan.before_identity_fix_20261104.json` 与更新后的 `experiment_plan.json`
- `confusion/qa_eligibility.json`：源任务 SHA256、排除 ID、原因及范围
- `wiring_identity_fix_20261104.json`：三分支均经过实际训练器的完整资产解析；每个分支有 19,782 条 QA、2,697 条有对比负例、79 条显式零权重排除项
- `completed/`：复用阶段的命令和输出哈希

修复提交为 `4d627df`（打分资格）与 `c08ca31`（零权重清单接线）。第二次恢复记录在 `recovery_exclusion_wiring_20261104.json`；旧清单已保留，2,697 条有效负例的身份和权重不变。该变化修正了对比训练的可匹配范围，不更改已训练模型、生成 QA、S2 轮数或候选得分公式；发表时报告该覆盖率。

## CoDEx-M 正式数据

旧 `codexm/` 保持原样。新目录是 `data/raw_datasets/codexm/` 与 `data/codexm/`。离线重建：

```bash
PYTHON=../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python
"$PYTHON" scripts/download_codexm.py --import_dir codexm/raw --offline
"$PYTHON" scripts/prepare_codexm.py
```

外网可用的新环境可省去 `--import_dir codexm/raw --offline`，下载器使用固定公开版本并验证校验和。已有文件校验失败时拒绝覆盖。

准备结果：

| 输入 | 已核对规模与用途 |
| --- | --- |
| train.txt | 185,584 条边，仅用于训练图和训练参考 |
| processed_train.json | 训练图＋2,000 条 train-only 参考，供新协议任务生成 |
| processed_val.json / processed_test.json | 10,310 / 10,311 条完整官方格式问题 |
| recurrent_relation_prediction_full.json | 相同训练参考及完整验证/测试，保留来源 ID |
| recurrent_relation_prediction_smoke.json | 各划分前 96 条，与完整输入候选及 ID 一致 |
| rel2text.json / entity2text.txt | 51 关系文字及完整实体文字，接入已有解析和训练图过滤 |

图节点 17,050 个。原始三元组使用 [CoDEx 固定版本](https://github.com/tsafavi/codex/tree/3132e426c2a6b643b70bad679905a3a6270be440)，文字映射和题面规则核对 [GRIP 固定版本](https://github.com/JiaruiFeng/GRIP/blob/d423742ca41212c4a75f35bec11d7117b1921a1a/data/raw_datasets/codexm/process.py)。映射 JSON 的字段完全一致，原文件键序和缩进不同；原始字节哈希及规范 JSON 哈希均保留在 `source_manifest.json`。

评测每题从训练关系池中只排除金关系，抽 9 个负关系后与金关系混洗。验证和测试依次消耗同一个 `numpy.random.RandomState(2026)`，不抽子集。训练负例才按 train-only 图过滤已知正关系；不把旧目录额外过滤后的评测列表作为正式输入。原始三元组分类负例不进入本任务。

明确偏离项：关系文字排序取代官方无序集合；实体 ID 排序固定节点索引；保留没有末尾换行的最后一行；增加独立随机流生成训练参考。官方文字值不变，不能声称逐题候选与论文历史文件完全相同。精确规则、NumPy 版本和各输出 SHA256 见 `data/codexm/preparation_manifest.json`。

## 运行入口

固定矩阵位于 `configs/ccfb_sprint_matrix.json`。新入口所有 S2 均固定 3 轮、answer-only、继承 S1 AdamW、梯度累积 512；NELL/CoDEx-M r=4，FB/WN r=24。三个方法同 seed 共享负例身份，采样及训练都显式使用指定 seed。训练只评验证，选模冻结后再运行完整测试。

CPU 预检不加载模型、不创建 tmux：

```bash
DATASET=codexm SEED=2026 DRY_RUN=1 bash configs/run_ccfb_sprint.sh
DATASET=nell23k SEED=2026 DRY_RUN=1 bash configs/run_ccfb_sprint.sh
DATASET=fb15k237 SEED=2027 DRY_RUN=1 bash configs/run_ccfb_sprint.sh
```

预检 JSON 中 `missing_dependencies` 和 `missing_sources` 为空，且数据哈希检查通过，才表示已有输入可用；WN/CoDEx-M 后续 seeds 在首轮未完成时会明确列出依赖。

实际执行采用相同命令，去掉 `DRY_RUN=1`。只在前一 GPU 任务结束、存储满足要求后启动：

```bash
# 第四数据集首轮：新任务生成 → S1 → B1 → 打分 → 冻结负例 → 三方法 → 验证选模 → 测试
DATASET=codexm SEED=2026 bash configs/run_ccfb_sprint.sh

# NELL 公平基线：只补 B1，复用原任务和 S1，固定 3 轮
DATASET=nell23k SEED=2026 bash configs/run_ccfb_sprint.sh

# 固定任务 / S1 / 教师评分的配对 S2 复验示例
DATASET=fb15k237 SEED=2027 bash configs/run_ccfb_sprint.sh
```

NELL/FB/CoDEx-M 后续 seeds 为 2027、2028；WN 为 2027—2030。逐 seed、逐数据集串行执行；不复制已完成 seed2026。FB/WN 的 seed2026 应继续使用原入口。NELL seed2026 三个方法的历史结果保留，新入口只补公平 B1。

重复运行复用同一任务、S1 和冻结教师评分；S2 随机数和负例采样用当前 seed。NELL/FB 主方法固定为 Soft-Mix 1.0；WN/CoDEx-M 从首轮验证选模读取主方法，所有后续 seed 统一复验，不重新按 seed 挑选方法。该范围衡量固定 S1 下的 S2 波动；独立 S1 检查、额外机制对照仍按冲刺计划另批执行。

新首轮主任务 JSON 尚未生成，旧的 12,300 QA / 236,027 context 任务不直接复用。新入口会按节点属性 QA、纯文本 context、均匀采样、rooted reasoning、训练节点、保留平行边等当前协议，从新的 train-only 图和参考生成任务。

默认输出：CoDEx-M 首轮为 `results/runs/codexm_three_configs_seed2026/`；其余新批次为 `results/runs/ccfb/<dataset>/seed<seed>/`。重跑相同命令会核验被冻结的输入、源码、命令、已完成输出；训练器复用 checkpoint 的模型、优化器、调度器和 RNG 状态。最终 adapter 存在而验证未完成时只补评测。

NELL 同时报完整 4,944 题 official_em，以及按源 ID 排除原末行后的 4,943 题 official_harness_em；后者用于论文数字比较。验证按 4,951/4,950 题相应处理。

## 已执行验证与待运行范围

- 服务器原 Python 3.12 / NumPy 2.2.6 环境，74 项 CPU 回归通过：候选 RNG、train-only 来源、实体文字和过滤、生产输入校验、seed 参数、验证先于测试、选模范围、显式零权重排除及哈希恢复。调度测试中的模型产物是明确的假文件；另对 WN 三个真实分支完成全量训练器资产解析，没有据此宣称新方法 GPU 训练通过。
- 对实际全部 20,621 条 CoDEx-M 评测题独立重算原三元组 ID、官方题面和候选随机顺序；训练边及 2,000 条参考只来自 train；二次准备各输出文件字节哈希相同。
- 真实输入预检通过：CoDEx-M 首轮、NELL seed2026 B1、NELL/FB seed2027。WN/CoDEx-M 复验入口已准备，但必须等各自首轮完成、选模冻结。
- 新首轮和多种子 GPU 路径尚未实际执行。最小机制对照和独立 S1 的 GPU 实验仍为后续阶段。

对应检查：

```bash
PYTHONPATH=src:scripts:../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp \
  "$PYTHON" -m pytest -q tests/test_offline_scoring_preflight.py tests/test_task_file.py \
  tests/test_validation_hpo.py tests/test_shared_pool_samplers.py tests/test_shared_pool_sampler_smoke.py \
  tests/test_prepare_codexm.py tests/test_ccfb_sprint.py tests/test_wn_three_configs.py tests/test_offline_scoring.py
```

详细 CPU 预检计划保存在 `results/runs/ccfb_preparation_20261104/`。Git 只保存代码、轻量来源及准备清单，较大的原始/处理数据保留在服务器并由脚本重建。

## 当前存储限制

准备后约剩 35.8 GiB。新四分支批次要求至少 45 GiB，单独补 NELL B1 要求 12 GiB，每阶段低于 10 GiB 会停止并保留 checkpoint。新入口使用全局 GPU 锁、批次锁、共享起点锁及 GPU 空闲检查，避免与当前 WN 重叠。

WN 后还会继续消耗空间。CoDEx-M 启动前需要扩容或明确归档方案；多种子阶段按冲刺计划争取至少 150 GiB。此次没有删除历史 adapter、S1 优化器、任务、逐题预测或 checkpoint。
