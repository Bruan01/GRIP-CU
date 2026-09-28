# Exp1：GRIP 阶段干扰与 LoRA 融合实验

## 状态

**尚未开始训练。** 本目录是该实验唯一的工作区：输入副本、脚本、配置、日志、检查点、适配器、预测与分析都应存放在此目录树中。项目源代码和缓存只能作为只读来源；任何脚本都必须显式将输出路径指向本目录。

## 当前预检记录（2026-09-28 UTC）

- GPU 查询：`nvidia-smi` 返回 `Failed to initialize NVML: Driver/library version mismatch`（NVML 库版本 595.91）。当前无法确认 GPU 可用性，因而没有启动训练。
- 可用系统内存约 25 GiB；根分区可用约 95 GiB（使用率约 90%）。大规模模型/数据写入前需重新检查可用空间。
- Hugging Face 缓存存在 `Qwen2.5-0.5B-Instruct` 的目录，但其中有 `.incomplete` 权重分片；不能据此认定模型已经完整可加载。
- 当前复现代码未发现 `qwen-0.5b` 模型别名；需在本目录内的适配层支持真实模型路径/ID，不能误用 7B 映射。
- `13_base_method/grip-exp/grip-exp` 中未发现已处理的 NELL23K `processed_*.json` 或现成 `tasks.json`；当前可见的是 `data/raw_datasets/nell23k` 原始数据目录。训练输入须先在 exp1 内生成/拷贝并校验。
- 已有服务器训练 session：`shared-pool-coverage-adaptive-k-full-20260926`。不得打断该任务，也不得与其盲目争抢 GPU。
- 该基线 NELL23K shell 脚本设定 `save_strategy=no`，不能直接用于本研究的长训练恢复需求；本实验必须使用自有、可恢复配置，按 step 保存 checkpoint。

以上是一次性快照，启动训练前须在 `.inbox2/exp1/` 内重新记录 GPU、显存、磁盘与模型完整性。

## 本轮实验目标与执行闸门

先用 Qwen2.5-0.5B 做**冒烟**，验证数据流、Stage 1/Stage 2 训练、双 LoRA 保存/加载、固定融合与学习融合、评测和断点恢复。冒烟通过后再做 Qwen2.5-7B 小量实验；小量实验通过后才进入 7B 全量。小模型结果只作为实现/方向筛选证据，不等同于 7B 结论。

当前阻塞项是 GPU 驱动/NVML 不匹配、0.5B 缓存不完整、缺少已处理训练输入、以及基线脚本不满足 checkpoint/tmux 约束。解除前只做只读审计和实验脚手架，不启动模型训练或长评测。

## 预注册问题

1. 在本复现设置中，S2（QA/推理样本）是否会降低 S1（context/图事实记忆样本）对应的闭卷验证能力？
2. 若确有阶段干扰，两个阶段专属 LoRA 的组合是否比同一 LoRA 顺序训练更好？
3. 在相同两个适配器上，验证集学习逐层融合权重是否优于固定融合？
4. S1/S2 交替数据日程是否带来独立收益？交替与动态融合有无正向交互？

## 待实现的实验组

| ID | 数据顺序 | 参数 | 融合/用途 |
|---|---|---|---|
| B0 | S1 后 S2 | 一个 LoRA 连续更新 | 原复现对照，并保存 S1 后检查点以量化干扰 |
| B1 | 仅 S1 | 一个 LoRA | S1 能力参照 |
| B2 | S1 后 S2 | 两个独立 LoRA | 固定等权/固定和，测参数分离效果 |
| B3 | S1 后 S2 | 两个独立 LoRA | 冻结适配器，在验证集学习逐层 α |
| B4 | S1/S2 小批交替 | 对应阶段更新对应 LoRA | 固定融合，测交替训练效果 |
| B5 | S1/S2 小批交替 | 两个独立 LoRA | 周期性冻结 LoRA 并用验证集更新 α；仅在 B3/B4 有信号后运行 |

交替数据训练和适配器分离是不同因素，不把两者捆在一起作为唯一对照。至少完成 B0/B1，再决定是否继续其余组。

## 模型与融合定义

基础权重冻结，LoRA 更新写为 `ΔW[i,l] = B[i,l] @ A[i,l]`。固定组合应对**权重增量**进行定义，不能把低秩因子直接相加并假定等价：一般 `B1A1 + B2A2 != (B1+B2)(A1+A2)`。

B3 的主要形式：

`ΔW[l] = sigmoid(theta[l]) * ΔW1[l] + (1 - sigmoid(theta[l])) * ΔW2[l]`

先用每层一个 α；Stage 3 冻结基础模型和两个 LoRA，只更新 α。验证集和测试集严格隔离；测试集不得用于 α、阈值、早停或超参数选择。另记录固定 α=0.5、固定和以及全局单一 α 作为对照。

## 指标与判定

- 分别报告 S1 专项闭卷查询和 S2 专项 QA 的 EM/准确率；主指标定义需在运行前固定。
- 遗忘量：`S1_after_S2 - S1_after_S1`，同时报告配对 seed 差异和区间。
- 报告联合宏平均、每个阶段单项结果、每 seed 原始结果、训练步数、样本曝光数、GPU 小时、峰值显存、adapter/checkpoint 大小、评测延迟。
- 以随机种子为训练重复单位。先用小 pilot 估计种子间差异，再确定正式 seed 数和最小实际重要差异；不得事后挑选最佳 seed。
- M3 仅当 B3 与 B4 各自显示可信信号时运行；按预设因子交互检验互补性。
- 若 B0 未出现稳定且超过预设重要差异的 S1 下降，则不能声称新方法“解决遗忘”；应改写为多目标能力组合问题或终止遗忘分支。

## 目录约定

```text
.inbox2/exp1/
  README.md                 本文件：研究方案、状态和闸门
  resources.json            每次训练启动前的资源快照
  manifest/                 来源、哈希、环境、提交号、命令清单
  inputs/                   数据副本、固定 split、任务样本与数据校验
  configs/                  0.5B smoke、7B pilot、7B full 配置
  scripts/                  本实验自有脚本/入口和验证测试
  runs/<run_id>/             每个运行的 run.log、检查点、adapter、指标和预测
  analysis/                 预先定义的评估和统计脚本/报告
```

目录可以按需创建；不得把日志、缓存、临时文件、checkpoint 或分析产物写到 exp1 外。软件包/只读模型缓存本身可继续留在其原有缓存位置，但输入文件如被实验使用，应在 `inputs/` 留可追溯副本或明确登记只读来源和哈希；具体采用哪种方式需先确认许可与空间。

## 长任务运行规范

任何训练和长时间占 GPU 的评测必须通过 tmux 启动；启动脚本放在 `scripts/`，日志及输出放在 `runs/<run_id>/`。启动前检查同名 session，已存在则提示 attach 而不另开训练。每个训练按 step 保存 HF checkpoint（默认 `save_steps=10`、`save_total_limit=2`），重启从同一 run 目录自动续训；预测若支持 JSONL 续写则启用。终态适配器和检查点不能互相覆盖。

启动脚本应满足项目 `configs/tmux_guard.sh` 的等价保护要求，但不能将脚本/日志输出到 exp1 外。启动前需核实 guard 的输出路径行为；若不符合隔离要求，应在本目录内制作 wrapper/guard，或调整本目录专用配置，而不能直接调用会把产物写出目录的现成命令。

## 下一步

1. 解决 GPU 驱动与 NVML mismatch，并确认已有 GPU 使用情况；不得影响现有 tmux 任务。
2. 检查并完整获取 0.5B 权重，验证 tokenizer/model 加载；记录模型 ID、revision 和哈希。
3. 确认 NELL23K 数据的许可与来源，在 exp1 内准备隔离输入与固定评测划分。
4. 核实任务生成是否可复用缓存；任何任务生成都需要相应的可恢复执行路径，长时间 GPU 生成同样进 tmux。
5. 在 exp1 内实现最小 runner、检查点恢复、双适配器融合和自动化测试；先语法/单测，再短小烟测。
6. 0.5B 冒烟完整通过并通过质量门后，才规划 7B pilot；pilot 通过后再冻结全量配置。

- GPU currently used by existing tmux session `shared-pool-coverage-adaptive-k-full-20260926`; inspected Python PID 60981 and confirmed it is the Qwen-7B listed-contrastive run. Do not start another GPU job until this process exits.
- Existing cache model is incomplete (an `.incomplete` weight shard remains). An attempt to download the missing Qwen2.5-0.5B-Instruct snapshot to `exp1/models/` failed because this execution environment has no network route to `huggingface.co` (`Network is unreachable`). No training was started.
- The isolated smoke data was prepared in `inputs/nell23k_smoke_16_each.json` from a read-only copy of the existing generated GRIP task file. Source SHA256 and input counts are recorded in `manifest/input_manifest.json`.
- `scripts/train_mixed_lora_smoke.py` and `scripts/launch_smoke.sh` were added under exp1. Python syntax checks passed; a tiny CPU-only PEFT test confirmed that only the active named adapter receives gradients and that PEFT accepts a concatenated weighted adapter. This is API validation only, not GPU/model training.

## 0.5B Smoke Progress

The runner is an implementation check for interleaved S1/S2 batches with stage-specific LoRAs and a summed low-rank adapter. It is deliberately small (16 samples per stage, 8 optimizer steps by default). After a pass, it must be expanded into separately controlled B0/B1/B2 groups before any scientific claim. This smoke alone cannot establish forgetting or efficacy.

**Current launch blockers:** GPU occupied by another long Qwen-7B run; model weights incomplete; hub download unavailable from this shell. Wait for GPU release and make a complete model snapshot available under `exp1/models/Qwen2.5-0.5B-Instruct/` before invoking the launch script. The script refuses CPU fallback and refuses a busy GPU.

The environment used is the existing read-only uv environment at `08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/`; it was not rebuilt. All experiment-specific files remain under exp1.

## Run Log

| Time UTC | Run | Stage | Status | Notes |
|---|---|---|---|---|
| 2026-09-28 | preflight | environment | blocked | NVML mismatch; PyTorch CUDA works; RTX 3090 23.56 GiB; BF16 works |
| 2026-09-28 | input preparation | smoke input | passed | copied task source and selected 16 context + 16 QA samples in exp1 |
| 2026-09-28 | model preparation | Qwen2.5-0.5B | blocked | cache partial; Hub fetch failed with network unreachable |
| 2026-09-28 | PEFT API probe | adapter behavior | passed | CPU micro-model verified adapter-specific gradients and `cat` combination API |
| 2026-09-28 | GPU smoke | training | not started | existing Qwen-7B job owns GPU |
