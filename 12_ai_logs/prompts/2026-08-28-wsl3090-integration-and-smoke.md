# WSL2 RTX 3090 Codex Handoff Prompt

日期：2026-08-28
阶段：RecurrentGRIP v1 WSL 集成与单图 Smoke
远程仓库：`https://github.com/Bruan01/GRIP-CU.git`

将以下内容作为 WSL Codex 的任务 Prompt：

---

你运行在 Windows WSL2，硬件是 NVIDIA RTX 3090 24GB。当前仓库是 GRIP-CU。

本阶段目标：

1. 完成 RecurrentGRIP v1 的 WSL2 + RTX 3090 环境集成；
2. 运行全部单元测试；
3. 准备 CLEGR `StationShortestCount` 精确 hop split；
4. 完成 Qwen2.5-0.5B 单图 smoke；
5. 分析 smoke 输出并形成可复现记录；
6. smoke 通过后停止，不启动两小时 Pilot。

你需要实际执行命令、检查输出和定位问题，不要只提供操作建议。

## 1. 研究与代码边界

顶层仓库：

```text
GRIP-CU/
```

Original GRIP：

```text
13_base_method/grip-exp/
```

RecurrentGRIP v1：

```text
08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/
```

必须遵守：

- 不修改 Original GRIP；
- Original GRIP 应保持 clean，并固定在 commit `2835b440bfd2c4de36f0380ae19bc1c22e6cb459`；
- RecurrentGRIP 修改只能发生在 `08_experiments/RecurrentGRIP/`；
- 不提交 `.venv`、模型缓存、outputs、原始运行结果；
- 不覆盖已有 RUN_ID；
- 正式评估必须使用 `--evaluation_device cuda --require_cuda true`；
- 不把 CPU 结果作为正式实验结果；
- 不改变 v1 的方法语义。

v1 固定机制：

- 一个 decoder layer 作为 recurrent executor；
- 同一个 block 对象重复执行 K 次；
- recurrence 严格共享参数；
- graph-specific LoRA 只注入 executor layer；
- 推理阶段不访问图；
- generation 强制 `use_cache=False`；
- 不加入 adaptive halting、gate、frontier loss、adapter routing 和跨图 meta-training。

如果是 WSL 兼容问题或明确运行 bug，只做最小修复并记录。如果修改方法语义、数据协议或关键实验配置，使用 `08_experiments/RecurrentGRIP/create_version.sh` 创建新版本。

## 2. 阅读顺序

先检查并阅读项目内的 `AGENTS.md`，然后依次阅读：

1. `README.md`
2. `TODO_WSL3090.md`
3. `RESEARCH_STATUS.md`
4. `08_experiments/RecurrentGRIP/README.md`
5. `08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/README.md`
6. `08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/ENVIRONMENT_WSL3090.md`
7. `08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/PATCH_MANIFEST.md`
8. `08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/design/PILOT_PLAN.md`
9. `08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/grip-exp/docs/RECURRENT_GRIP.md`

## 3. 同步仓库

```bash
git rev-parse --show-toplevel
git status --short
git branch -vv
git remote -v
git submodule status
```

工作区 clean 时执行：

```bash
git switch main
git pull --ff-only origin main
git submodule update --init --recursive
git switch -c wsl3090-integration-20260828
```

如果分支已存在，切换到已有分支。

验证基线：

```bash
git -C 13_base_method/grip-exp status --short
git -C 13_base_method/grip-exp rev-parse HEAD
```

不符合预期时先调查，不要修改 Original GRIP。

## 4. 验证 WSL 和 GPU

执行并记录：

```bash
uname -a
cat /proc/version
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v timeout
command -v uv
python3 --version
df -h .
free -h
```

要求：

- WSL2；
- RTX 3090；
- 约 24GB 显存；
- `nvidia-smi` 正常；
- GNU `timeout` 可用；
- 项目优先位于 WSL Linux 文件系统，不长期从 `/mnt/c/...` 训练。

如果没有 uv，安装 uv 后刷新 PATH。Python 依赖只能安装到版本目录的 `.venv`。

## 5. 环境安装与测试

```bash
cd "$(git rev-parse --show-toplevel)/08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28"
bash configs/check_mac_static.sh
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

脚本名称中的 `mac` 不影响 WSL 静态检查；它只执行 AST、Bash 和 JSON 校验。

优先保留固定版本：

- Python 3.11
- PyTorch 2.7.1
- Transformers 4.56.1
- PEFT 0.17.1
- Accelerate 1.10.1
- torch-geometric 2.6.1

默认 PyTorch wheel 不能使用 CUDA 时，再尝试：

```bash
PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  bash configs/setup_wsl3090.sh
```

必须验证：

- `torch.cuda.is_available()` 为 true；
- GPU 为 RTX 3090；
- 显存大于 20 GiB；
- BF16 可用；
- unittest 全部通过；
- recurrent executor forward 通过；
- PEFT 只注入 executor layer；
- adapter 保存/重载通过；
- evaluation model 重载后位于 CUDA。

测试失败时复现最小失败，区分环境、依赖 API 与逻辑问题，做最小修复，然后重新运行失败测试和完整测试。不要删除测试或降低断言。

## 6. 准备 CLEGR

查找：

```bash
find "$(git rev-parse --show-toplevel)" "$HOME" \
  -type f -name 'processed_test.json' 2>/dev/null
```

找到真实 CLEGR 数据后运行：

```bash
RAW_INPUT=/实际路径/processed_test.json \
  bash configs/prepare_clegr.sh
```

如果不存在，检查 Original GRIP 的 README、CLEGR 处理代码和已有脚本，使用项目官方流程准备数据；不虚构数据。

生成文件：

```text
grip-exp/outputs/data/clegr_reasoning/recurrent_station_shortest.json
grip-exp/outputs/data/clegr_reasoning/recurrent_station_shortest.json.stats.json
```

验证：

- 仅 `StationShortestCount`；
- true hop 由 `edge_index` 无向 BFS 重算；
- train/validation 为 1–2 hop；
- test 为 3–4 hop；
- `answer = max(true_hop - 1, 0)`；
- split 无泄漏；
- 最多 16 张图；
- 每个 hop 最多 32 个问题；
- 输出 split/hop 统计并抽查至少 10 个 BFS/label 样本。

数据校验失败时不运行 smoke。

## 7. 单图 Smoke

只有环境、测试和数据验证全部通过后运行：

```bash
CLEGR_INPUT="$PWD/grip-exp/outputs/data/clegr_reasoning/recurrent_station_shortest.json" \
RUN_ID=wsl3090_smoke_20260828_01 \
  bash configs/run_qwen05b_smoke_wsl.sh
```

若 RUN_ID 已存在，递增为 `_02`、`_03`。

固定 smoke 配置：

- Qwen2.5-0.5B；
- 1 张图；
- train K=2；
- evaluation K=1,2；
- BF16；
- batch size 1；
- gradient accumulation 4；
- CUDA evaluation；
- 30 分钟 soft limit；
- 45 分钟 OS hard timeout。

检查 GPU 实际占用、无 CPU 回退、adapter 保存/加载、cache-free generation、预测和环境文件完整落盘。

## 8. 分析结果

```bash
RUN_DIR="$(cat results/LAST_SMOKE_RUN.txt)"
find "$RUN_DIR" -maxdepth 3 -type f | sort
RUN_DIR="$RUN_DIR" bash configs/analyze_pilot.sh
```

报告：

- GPU、CUDA、PyTorch、Transformers、PEFT；
- 总用时与峰值显存；
- train/validation/test 数量；
- K=1 与 K=2 准确率；
- correct 与 disabled adapter 差异；
- OOM、NaN、CUDA、adapter load 错误；
- evaluation model 是否位于 CUDA；
- 是否建议进入两小时 Pilot。

Smoke 只验证管线，不把单图指标写成论文结论。

## 9. 项目记忆

新增不可覆盖记录：

```text
08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/logs/
WSL_INTEGRATION_AND_SMOKE_2026-08-28.md
```

记录环境、commit、依赖、测试、数据统计、RUN_ID、配置、指标、错误、修复和结论。

更新：

- `TODO_WSL3090.md`
- `RESEARCH_STATUS.md`
- `08_experiments/RecurrentGRIP/README.md`
- 当前版本 `README.md`
- 代码变化时更新 `PATCH_MANIFEST.md`

状态需区分：

- macOS static PASS；
- WSL ML integration PASS/FAIL；
- RTX 3090 smoke PASS/FAIL；
- two-hour pilot PENDING。

## 10. 提交

```bash
git status --short
git diff
git -C 13_base_method/grip-exp status --short
git submodule status
```

只提交代码修复和 Markdown 摘要：

```bash
git add <相关文件>
git commit -m "test: validate RecurrentGRIP v1 on WSL2 RTX 3090"
git push -u origin wsl3090-integration-20260828
```

不要提交 `.venv`、模型、outputs、predictions、run.log 或完整 results。

## 11. 停止条件与汇报格式

完成 smoke、分析、验证记录和分支推送后停止。不要运行：

```bash
bash configs/run_qwen05b_pilot.sh
```

最终按以下格式汇报：

```text
## WSL 环境
- WSL：
- GPU：
- 显存：
- Driver：
- CUDA：
- PyTorch：
- BF16：

## 测试
- static：
- unittest：
- PEFT adapter scope：
- adapter reload CUDA placement：

## CLEGR
- 原始数据：
- 生成数据：
- 图数量：
- train/validation/test 分布：
- BFS/label 抽查：

## Smoke
- RUN_ID：
- RUN_DIR：
- 状态：
- 用时：
- 峰值显存：
- K=1：
- K=2：
- correct vs disabled：
- 错误：

## 代码
- 修改文件：
- commit：
- remote branch：
- Original GRIP clean：

## 结论
- 是否通过 smoke：
- 是否建议进入两小时 Pilot：
- 下一步命令：
```

---

## 使用说明

WSL Codex 应以这个 Prompt 为唯一阶段目标。返回 smoke 报告后，再决定是否启动两小时 Pilot，不提前扩展到动态停止或完整六数据集实验。
