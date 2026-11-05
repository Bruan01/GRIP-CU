# CCF B 冲刺：运行准备与续跑

更新时间：2026-11-04，北京时间。服务器项目目录：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
```

后续状态（2026-11-05）：WN 首轮已完成，CoDEx-M 首轮已于 15:53 启动。本文以下保留 11 月 4 日的准备与恢复快照；最新启动记录见 [CoDEx-M 第四数据集实验](CODEXM_第四数据集实验.md)，WN 完整成绩见 [WN18RR 第三数据集实验](WN18RR_第三数据集实验.md)。

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
