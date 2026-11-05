# CoDEx-M：第四数据集首轮实验

更新：2026-11-05 16:01，北京时间。本文记录实际启动快照，后续进度以服务器 `run_status.json` 和逐阶段日志为准。

## 实际启动与当前状态

- 启动时间：2026-11-05 15:53:00 北京时间（07:53:00 UTC）。
- tmux：`ccfb-codexm-2026`；seed：2026。
- 启动时仓库 HEAD：`c17e4386c0c196f1f55e0ba37ab92288d501a791`。
- 状态：`RUNNING / tasks`。Qwen2.5-7B-Instruct 已加载到 RTX 3090，正在生成新任务。
- 16:01 核对：首个 `summarization_s1` 子任务已完成并缓存 368/18,000 条，GPU 利用率 96%，占用 15,980 MiB。该计数只描述这一子任务，不是全实验完成比例。
- 尚无 S2 或评测成绩；B1 与三个方法均在后续队列中。

服务器项目目录：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
```

本轮输出目录：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/codexm_three_configs_seed2026
```

## 数据与协议

| 项目 | 本轮设置 |
| --- | --- |
| 骨干 | Qwen2.5-7B-Instruct，本地缓存、离线加载 |
| 训练图 | 185,584 条 train 边，17,050 个节点；51 个训练关系文字 |
| 任务参考 | 独立随机数抽取的 2,000 条 train-only 参考 |
| 完整验证 / 测试 | 10,310 / 10,311 题，每题 10 个候选 |
| LoRA | r=4、alpha=8，down_proj / up_proj / gate_proj |
| S2 预算 | 各分支固定 3 轮，无早停；answer-only，继承共享 S1 优化器 |
| S2 分支 | B1、Soft-Mix 1.0、Calibrated 1.0、Calibrated 0.5 |
| S2 训练参数 | batch=1，梯度累积 512；lr=0.001、weight_decay=0.0001、eps=0.0001、max_grad_norm=1 |
| 评测 | 自由生成 official_em，贪心解码，上限 1000 token |
| 选模 | 首轮在三个方法中按完整验证集选主方法，冻结选择后再测试四分支；B1 作同条件对照 |

任务从新的 `data/codexm/processed_train.json` 生成：节点属性 QA、纯文本 context、均匀采样、rooted reasoning、训练节点和保留平行边等开关与当前协议一致。旧 `codexm/tasks/` 的 12,300-QA / 236,027-context 任务不用于本轮。

评测候选按训练关系文字池排除金关系后抽取 9 个负候选，再排列 10 个候选；完整 validation 后接完整 test 使用同一 `RandomState(2026)`。训练负例过滤只使用训练图已知正例。候选标签排序、实体 ID 排序、保留末行和独立训练参考随机数属于已披露处理差异；这轮固定 3 轮预算也与论文最大轮数及损失早停规则不同。

来源与完整哈希见 [原始数据来源清单](data/raw_datasets/codexm/source_manifest.json) 和 [准备与校验清单](data/codexm/preparation_manifest.json)。CoDEx 数据固定于 `3132e426c2a6b643b70bad679905a3a6270be440`；GRIP 文字映射及处理器固定于 `d423742ca41212c4a75f35bec11d7117b1921a1a`。输入已通过全量 20,621 条评测题的来源、题面与候选 RNG 核验，二次准备字节哈希一致。

## 启动检查与自动队列

启动前 WN 为 COMPLETE，GPU 无计算进程，未发现重复 tmux 队列。可用空间约 47.61 GiB，通过新四分支批次 45 GiB 门槛；本地四个模型分片及 tokenizer 文件齐全。实际入口的 dry-run 校验输入哈希，返回 `missing_dependencies=[]`、`missing_sources=[]`，并确认 r4/alpha8、固定 3 轮及四分支。

本次 CPU 预检计划保存为：

```text
results/runs/ccfb_preparation_20261104/codexm_seed2026_launch_plan.json
```

本次启动命令（在服务器项目目录）：

```bash
DATASET=codexm SEED=2026 TMUX_SESSION=ccfb-codexm-2026 \
  bash configs/run_ccfb_sprint.sh
```

自动顺序：新任务生成 → 共享 S1 → B1 与验证 → 训练集混淆评分 → 冻结负例及完整接线核验 → 三个方法与验证 → 冻结主方法 → 四分支完整测试。Soft-Mix 保留 6 个均匀负例加 3 个混淆引导负例；Random-K 独立对照仍暂缓。

冻结计划为 `experiment_plan.json`，状态为 `run_status.json`；`completed/` 保存完成阶段的命令和输出哈希。当前任务日志为 `tasks.log`，逐条缓存位于 `tasks/cache/`。后续还需从真实生成 QA 登记有效覆盖率、排除项和权重分布；此前 74 项 CPU 检查不替代这些运行证据。

SSH 登录后查看：

```bash
tmux attach -t ccfb-codexm-2026
```

仅查看当前日志：

```bash
tail -f /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/codexm_three_configs_seed2026/tasks.log
```

若队列中断，先检查 `run_status.json`、对应日志、GPU 和磁盘，再在同一目录用相同启动命令恢复。入口校验冻结身份及已完成产物，按缓存或 checkpoint 接续；最终 adapter 已保存而验证未完成时只补评测。

## 数值目标与后续记录

同骨干论文参考为 CoDEx-M **94.42%**，沿用已核验的 [GRIP 论文 Table 3](https://arxiv.org/html/2511.07457v2)。首轮记录四分支完整验证 / 测试结果，主结果按验证集预选配置报告；同时计算相对论文和相对 B1 的百分点差值。当前尚无本轮结果。

后续配对 seeds 2027、2028 按 [CCF B 冲刺计划](CCFB_冲刺计划.md) 复用固定任务、S1 和教师评分，统一复验首轮冻结的主方法。三种子均值与配对增益完成前，只将首轮作为单次证据。汇总入口为 [结果与论文对照](RESULTS_20261030.md)。
