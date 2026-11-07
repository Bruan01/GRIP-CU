# CoDEx-M：第四数据集首轮实验

更新于 2026-11-07，北京时间。首轮四分支全部完成，验证集预选的 Soft-Mix 1.0 测试准确率为 96.2370%。

## 启动和完成时间

- 启动时间：2026-11-05 15:53:00 北京时间（07:53:00 UTC）。
- tmux：`ccfb-codexm-2026`；seed：2026。
- 启动时仓库 HEAD：`c17e4386c0c196f1f55e0ba37ab92288d501a791`。
- 11 月 5 日 16:01 核对到 Qwen2.5-7B-Instruct 在 RTX 3090 上运行，首个 `summarization_s1` 子任务已缓存 368/18,000 条。
- 11 月 6 日 08:09 完成任务生成，产出 238,634 条 context 和 19,798 条 QA，同日 19:12 完成共享 S1。
- 11 月 7 日 16:28 完成四组验证并冻结主方法，19:56:33 完成四组测试。状态为 `COMPLETE`，总耗时约 52 小时 4 分钟。

服务器项目目录：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
```

本轮输出目录：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/codexm_three_configs_seed2026
```

## 首轮结果

同骨干 GRIP-Qwen2.5-7B 的 CoDEx-M 测试报告值为 94.42%。[GRIP 论文 Table 3](https://arxiv.org/html/2511.07457v2)

| 配置 | 验证集 official_em | 测试集 official_em | 测试正确数 / 10,311 | 较 B1（百分点） | 较论文（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 论文 GRIP-Qwen2.5-7B | — | 94.42% | — | — | 0 |
| B1 纯生成 | 94.0737% | 94.2198% | 9,715 | 0 | −0.2002 |
| **Soft-Mix 1.0（验证集预选）** | **96.0815%** | **96.2370%** | **9,923** | **+2.0173** | **+1.8170** |
| Calibrated 1.0 | 96.0524% | 95.9267% | 9,891 | +1.7069 | +1.5067 |
| Calibrated 0.5 | 94.9952% | 94.9277% | 9,788 | +0.7080 | +0.5077 |

`selection.json` 在测试前按 10,310 道完整验证题选中 `soft_mix_1`，记录 `test_seen=false`。四组分别有 10,310 个唯一验证 ID 和 10,311 个唯一测试 ID，同一划分内的 ID 集合一致。逐题重算 `official_correct` 与各组汇总匹配。Soft-Mix 相对 B1 净多答对 208 题。

全部 19,798 条 QA 参与生成训练。混淆评分与候选对比覆盖 2,758 条可识别实体的关系 QA，另有 202 条关系 QA 在三个方法清单中记录为显式零权重项。三方法使用相同的负例身份。有效 QA 上 Calibrated 1.0 的 λ 为 0.503247–0.628281，Calibrated 0.5 的范围为 0.251624–0.314141。

本轮为 seed2026，使用固定 3 轮 S2。论文的最大训练轮数及早停规则不同。相对论文的差值用于核对数值目标，相对同条件 B1 的差值用于衡量本轮增益。后续使用预定 seeds 2027、2028 确认固定任务、S1 和教师下的 S2 稳定性。

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

冻结计划为 `experiment_plan.json`，状态为 `run_status.json`，`completed/` 保存完成阶段的命令和输出哈希。完整成绩见 `selection.json`、`comparison.json` 和 `test/<分支>/summary.json`，逐题预测见各评测目录的 `predictions_correct.jsonl`。任务日志和缓存保存在 `tasks.log` 与 `tasks/cache/`。

SSH 登录后查看本轮汇总：

```bash
cat /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/codexm_three_configs_seed2026/comparison.json
```

若队列中断，先检查 `run_status.json`、对应日志、GPU 和磁盘，再在同一目录用相同启动命令恢复。入口校验冻结身份及已完成产物，按缓存或 checkpoint 接续；最终 adapter 已保存而验证未完成时只补评测。

## 后续复验

配对 seeds 2027、2028 的负例清单已由首轮固定教师评分生成，并通过完整训练器资产解析。两组共享原任务、S1 和教师评分，统一复验首轮冻结的 Soft-Mix 1.0。GPU 复验按存储条件和计划排期执行。[CCF B 冲刺计划](CCFB_冲刺计划.md) · [结果与论文对照](RESULTS_20261030.md)
