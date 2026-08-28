# RecurrentGRIP v1 — Fixed-Depth Shared Executor

## Experiment ID

`RG-E1-v1-fixed-depth-2026-08-28`

## Goal

检验图专属 LoRA 是否可以被解释为参数化图程序：图先写入 executor layer
内的 graph-specific LoRA，闭卷推理时由同一个共享 decoder block 重复执行
`K` 次，从而提升未见 3–4 hop 问题的准确率。

## Hypothesis

在只使用 1–2 hop `StationShortestCount` QA 训练后：

1. 3–4 hop 上 `K=3/4` 优于 `K=1`；
2. 最小成功 recurrence depth 与 true hop 正相关；
3. shuffled adapter 或 disabled adapter 会破坏性能。

## Method Boundary

```text
Z(0) = E(q)
Z(k+1) = F_{phi, DeltaW_G}(Z(k))
y = D(Z(K))
```

- `DeltaW_G`：只位于 recurrent executor layer 的图专属 LoRA；
- `F`：同一个 decoder block 对象，所有 recurrence 严格共享参数；
- 推理 prompt 不含图、路径、frontier 或 true hop；
- generation 强制 `use_cache=False`；
- v1 不包含 adaptive halting、update gate、frontier loss 和跨图 meta-training。

## Dataset / Split

- Dataset：CLEGR-Reasoning；
- Question type：仅 `StationShortestCount`；
- Train/validation：true hop 1–2；
- Test：true hop 3–4；
- true hop：在 `edge_index` 上重新执行无向 BFS；
- label check：必须满足 `answer = max(true_hop - 1, 0)`。

## Controls Included in This Snapshot

- correct graph adapter；
- cyclically shuffled graph adapter；
- adapter disabled；
- recurrence sweep `K=1,2,3,4,5`。

Original GRIP 和 GRIP+More-QA 继续使用各自独立 baseline runner，避免改变
Original GRIP 默认行为。

## Directory

```text
v1_fixed_depth_2026-08-28/
├── README.md
├── SOURCE_BASELINE.md
├── PATCH_MANIFEST.md
├── configs/
├── grip-exp/
├── logs/
└── results/
```

## Runtime Environment

- macOS：代码编辑、版本管理与无 Torch 静态检查；
- Windows WSL2 + RTX 3090 24GB：ML 单元测试、Qwen smoke 与正式 Pilot；
- 正式脚本强制 `require_cuda=true`，并显式把评估模型放到 CUDA，防止 adapter 重载后留在 CPU。

完整部署步骤见 [`ENVIRONMENT_WSL3090.md`](ENVIRONMENT_WSL3090.md)。

## Run

```bash
# macOS
bash configs/check_mac_static.sh

# WSL2 + RTX 3090
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
CLEGR_INPUT=/path/to/recurrent_station_shortest.json bash configs/run_qwen05b_smoke_wsl.sh
CLEGR_INPUT=/path/to/recurrent_station_shortest.json bash configs/run_qwen05b_pilot.sh
```

每次 smoke/pilot 自动写入新的 `results/runs/<RUN_ID>_*`，旧运行不会被覆盖。详细方法与数据步骤见 `grip-exp/docs/RECURRENT_GRIP.md`。

## Success Gate

至少满足以下两项才进入 v2：

1. 3–4 hop 比 Original GRIP 高至少 5 个绝对百分点；
2. `K=3/4` 明显优于 `K=1`；
3. 最优 `K` 与 true hop 正相关；
4. 1–2 hop 下降不超过 2 个百分点；
5. shuffled adapter 明显破坏性能或执行轨迹。

## Current Status

- [x] 独立代码快照；
- [x] CLEGR metadata 保留与精确 hop split；
- [x] fixed-depth recurrent executor；
- [x] correct/shuffled/none adapter controls；
- [x] 静态测试文件；
- [x] macOS / WSL2 RTX 3090 环境分工与独立脚本；
- [x] macOS AST/Bash/JSON 静态检查通过（83 个 Python 文件，2026-08-28）；
- [x] evaluation adapter 重载后的显式 CUDA placement；
- [ ] WSL2 安装 ML 依赖后的 PEFT adapter-scope 测试；
- [ ] Qwen2.5-0.5B RTX 3090 smoke test；
- [ ] 两小时 CLEGR pilot；
- [ ] Original GRIP 与 More-QA 对照结果。
