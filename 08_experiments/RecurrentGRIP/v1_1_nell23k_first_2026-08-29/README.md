# RecurrentGRIP v1.1 — NELL23K-First Fixed-Depth Executor

## Experiment ID

`RG-E2-v1.1-nell23k-first-2026-08-29`

## Goal

先在 Original GRIP 已使用且已随代码仓库提供的 NELL23K 上完成 RecurrentGRIP
端到端验证，不让 CLEGR 下载和处理阻塞 WSL2 RTX 3090 smoke。图仍然先写入
executor layer 的 graph-specific LoRA，闭卷推理时由同一个共享 decoder block
重复执行 `K` 次。

## Research Boundary

```text
Z(0) = E(q)
Z(k+1) = F_{phi, DeltaW_G}(Z(k))
y = D(Z(K))
```

- `DeltaW_G` 只位于 recurrent executor layer；
- 所有 recurrence 使用同一个 decoder block 对象并严格共享参数；
- 推理 prompt 不包含原图；
- generation 强制 `use_cache=False`；
- NELL23K-first 只改变数据适配、运行顺序和结构距离分析，不引入 adaptive halting、gate、frontier loss、routing 或跨图 meta-training；
- `v1_fixed_depth_2026-08-28` 保持不变，继续保存 CLEGR-first 设计。

## Dataset Protocol

训练图和 QA 严格按 NELL23K 官方文件角色构造：

```text
train.txt -> 图结构 + recurrent QA train
valid.txt -> recurrent QA validation
test.txt  -> recurrent QA test
```

任务仍为 10-way relation prediction。候选关系从训练关系词表中确定性采样，答案
一定在候选集合内，默认 seed 为 `2026`。

对每个查询实体对，在 `train.txt` 构成的无向图上重新计算最短路径：

- 可达且距离大于等于 1：`true_hop = structural_distance`；
- 不可达或自查询：`true_hop = null`，仍保留在整体准确率中；
- hop/K 相关性只使用已知结构距离的样本；
- 该距离只用于结构诊断，不声称 relation label 必然由最短路径决定。

NELL23K 当前被 Original GRIP 处理为一个大图，因此 v1.1 控制组为：

```text
correct adapter
adapter disabled
```

只有一个图时，runner 会自动跳过 shuffled-adapter control。

## Directory

```text
v1_1_nell23k_first_2026-08-29/
├── README.md
├── SOURCE_BASELINE.md
├── PATCH_MANIFEST.md
├── configs/
├── design/
├── grip-exp/
├── logs/
└── results/
```

## macOS Static/Data Check

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
bash configs/check_mac_static.sh

MAX_TRAIN_QUESTIONS=64 \
MAX_VALIDATION_QUESTIONS=32 \
MAX_TEST_QUESTIONS=64 \
  bash configs/prepare_nell23k.sh
```

默认输出：

```text
grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json
grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json.stats.json
```

`outputs/` 被 Git 忽略，不会污染远程仓库。

## WSL2 RTX 3090 Smoke

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
bash configs/prepare_nell23k.sh

RUN_ID=wsl3090_nell23k_smoke_20260829_01 \
  bash configs/run_nell23k_smoke_wsl.sh

RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)" \
  bash configs/analyze_nell23k.sh
```

Smoke 固定：

```text
Qwen2.5-0.5B
train/validation/test questions = 64/32/64
K_train = 2
K_eval = 1,2
correct/none adapter controls
45-minute OS hard timeout
```

每次运行写入新的：

```text
results/runs/<RUN_ID>_nell23k_qwen05b_smoke/
```

已有目录会直接报错，绝不覆盖。

## Two-Hour Pilot

Smoke 和分析通过后，重新准备更大的固定输入：

```bash
MAX_TRAIN_QUESTIONS=512 \
MAX_VALIDATION_QUESTIONS=128 \
MAX_TEST_QUESTIONS=512 \
  bash configs/prepare_nell23k.sh

RUN_ID=wsl3090_nell23k_pilot_20260829_01 \
  bash configs/run_nell23k_pilot_wsl.sh
```

Pilot sweep：

```text
K = 1,2,3,4
wall-time soft limit = 120 minutes
OS hard timeout = 180 minutes
```

## Metrics

`configs/analyze_nell23k.sh` 生成：

```text
analysis/summary.json
analysis/k_adapter_accuracy.csv
analysis/split_k_adapter_accuracy.csv
analysis/hop_k_accuracy.csv
```

主要用于第一阶段决策的指标：

1. correct adapter 与 no-adapter 的准确率差异；
2. K=1/2 的整体准确率差异；
3. adapter 保存、重载和 CUDA 评估是否成功；
4. 单样本延迟和峰值显存；
5. 可达样本上按 structural distance 分桶的 K 曲线。

## Success Gate

满足以下条件后才启动两小时 NELL23K Pilot：

1. 全部 WSL 单元测试通过；
2. Qwen2.5-0.5B 训练和评估都在 RTX 3090；
3. adapter 只注入 recurrent executor layer；
4. K=1、K=2 都产生完整预测；
5. correct/none control 都产生结果；
6. 无 OOM、NaN、CUDA error 或 CPU 静默回退；
7. 运行产物、环境和数据统计完整留存。

NELL23K smoke/pilot 证明执行链可行后，再决定是否运行 CLEGR 的严格 K↔hop
机制实验。CLEGR 不再是当前 WSL 集成的前置条件。

## Current Status

- [x] 从 v1 创建独立不可覆盖快照；
- [x] 不修改 Original GRIP；
- [x] NELL23K train/valid/test 数据适配；
- [x] 确定性候选关系采样；
- [x] train-graph 结构距离计算；
- [x] unknown hop 输出与指标兼容；
- [x] NELL23K smoke/pilot/analyze 脚本；
- [x] macOS AST/Bash/JSON 与 NELL23K 数据测试通过（85 个 Python 文件，2026-08-29）；
- [ ] WSL2 RTX 3090 完整测试；
- [ ] Qwen2.5-0.5B NELL23K smoke；
- [ ] NELL23K 两小时 Pilot；
- [ ] Original GRIP 公平对照；
- [ ] CLEGR 严格机制确认。
