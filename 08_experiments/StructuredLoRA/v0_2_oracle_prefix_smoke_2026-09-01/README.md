# StructuredLoRA v0.2 — Oracle Prefix Smoke

## Experiment ID

```text
structured_lora_v0_2_oracle_prefix_smoke_2026-09-01
```

目录中的 `2026-09-01` 是预注册的 WSL run/version 日期；实现与静态审计在 2026-08-31 完成。

## Research question

在**完全知道真实 hop 深度**的情况下，把固定总 rank 的 LoRA 组织成有序累计深度子空间，是否比同总 rank 的普通 LoRA 更适合闭卷图路径推理？

本版本只验证结构本身，不训练 router。如果 perfect/oracle routing 都不能胜过普通 LoRA，项目按预注册 stop rule 终止，不继续堆 learned router。

## Causal hypothesis

普通 LoRA：

```text
ΔW(x) = B A x, rank(A)=8
```

StructuredLoRA 将总 rank 8 拆为四个 rank-2 组：

```text
ΔW(x,d) = Σ_g z_g(d) B_g A_g x
```

ordered prefix 使用：

```text
1-hop -> G1
2-hop -> G1 + G2
3-hop -> G1 + G2 + G3
4-hop -> G1 + G2 + G3 + G4
```

所有组统一使用 `alpha / total_rank` 缩放，因此 `static_split` 全开时与 monolithic rank-8 在总 rank、参数量和 LoRA 总缩放上匹配。

`ordered_prefix_local_credit` 的前向仍使用完整 prefix，但深度 d 的样本只把梯度传给新增组 Gd：

```text
forward(d) = G1 + ... + Gd
gradient target(d) = Gd
```

这避免深层样本反复重写浅层组，是本版本对 depth-local credit assignment 的操作化定义。

## Registered methods

| Method | Forward mask | Gradient mask | Purpose |
|---|---|---|---|
| `monolithic` | rank-8 全部 | rank-8 全部 | 主基线 |
| `static_split` | 四组全部开启 | 四组全部 | 检查“仅拆 rank” |
| `flat_oracle` | 只开对应深度组 | 同 forward | 普通 oracle experts |
| `ordered_prefix` | G1...Gd | 同 forward | 核心结构假设 |
| `ordered_prefix_local_credit` | G1...Gd | 只更新 Gd | 局部 credit assignment |
| `permuted_depth_prefix` | 错配后的 prefix 长度 | 同 forward | 深度标签因果对照 |
| `random_group_order` | 固定随机顺序的 prefix | 同 forward | 组身份置换对称性检查 |
| `non_nested_random_masks` | 同 active-rank 数但破坏嵌套 | 同 forward | 有意义的非嵌套结构对照 |

### 为什么额外加入 non-nested control

固定随机排列 `[G3,G1,G4,G2]` 后再取前 d 组，在数学上仍是嵌套 prefix，只是组编号被重命名；它不应被预注册为“必须下降”的负对照。真正检验有序累计结构的是：保持每个深度激活组数不变，但让 active sets 不再形成嵌套链。因此加入 `non_nested_random_masks`，并将 `random_group_order` 改为 symmetry sanity check。

## Fixed data

只使用 v0.1 的严格 exact-hop 数据：

```text
train:       716 = 179 × 4 depths
validation:  152 =  38 × 4 depths
test:        156 =  39 × 4 depths
```

这些任务满足 directed shortest path、无更短路径、relation chain 答案唯一并保存 provenance。NELL23K support-depth proxy 不作为本版本的监督标签。

## Fairness invariants

所有方法固定：

- Qwen2.5-0.5B-Instruct；
- 完全相同的 train/validation/test；
- `total_rank=8`；
- `groups=4, group_rank=2`；
- `alpha=16`，缩放分母始终是 total rank；
- `down_proj, up_proj, gate_proj`；
- 相同 optimizer steps、batch、tokenization、optimizer 和 seed；
- base model 全冻结；
- LoRA A 的八个行向量在每层初始化为正交行，LoRA B 为零；
- 不修改 `13_base_method/grip-exp/`。

## Metrics and mechanism probes

每个 method/seed 记录：

- overall accuracy；
- 1/2/3/4-hop accuracy；
- macro-hop accuracy；
- worst-hop accuracy；
- trainable parameters；
- tokens seen、训练时间、峰值 GPU memory；
- 四个 hop batch 的 LoRA gradient cosine；
- group/rank-slice knockout；
- 完整 validation/test predictions；
- 环境、Git commit、数据 SHA256。

## Registered gate

`ordered_prefix` 必须同时满足：

1. overall 高于 equal-rank `monolithic`；
2. overall 高于 `flat_oracle`；
3. 3/4-hop 平均至少提升 3 percentage points；
4. 1-hop 下降不超过 1 percentage point；
5. 高于 `permuted_depth_prefix`；
6. 高于 `non_nested_random_masks`。

一个 seed 只输出 preliminary decision。三个注册 seeds `[42,43,44]` 齐全后，才允许输出：

```text
GO_LEARNED_ROUTER
```

否则输出：

```text
STOP_STRUCTURED_LORA
```

## macOS static validation

本地不需要 torch：

```bash
cd 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01
bash configs/check_static.sh
```

预期：

```text
13 tests PASS
READY_FOR_WSL_GPU
8 registered methods
StructuredLoRA v0.2 static checks passed
```

## WSL2 + RTX 3090 smoke

```bash
cd 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01

# guardenv 已有依赖时直接运行
conda activate guardenv
bash configs/run_wsl_smoke.sh
```

指定本地模型目录：

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-0.5B-Instruct \
RUN_ID=wsl3090_oracle_prefix_smoke_01 \
bash configs/run_wsl_smoke.sh
```

完整三个 seeds：

```bash
RUN_ID=wsl3090_oracle_prefix_full_01 \
bash configs/run_wsl_full_seeds.sh
```

## Output layout

```text
results/runs/<RUN_ID>/
├── monolithic/seed_42/
│   ├── adapter_model.pt
│   ├── run_summary.json
│   ├── predictions_validation.jsonl
│   └── predictions_test.jsonl
├── ordered_prefix/seed_42/
├── ...
├── suite_summary.json
├── suite_metrics.csv
└── REPORT.md
```

每次实验使用新的 `RUN_ID`，不覆盖历史版本。

## Current state

```text
Implementation: complete
Static audit:   pass
WSL GPU run:    pending
Learned router: blocked by oracle-prefix gate
```
