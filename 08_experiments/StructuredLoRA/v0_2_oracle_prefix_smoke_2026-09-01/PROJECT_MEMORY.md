# Project Memory — StructuredLoRA v0.2

更新日期：2026-08-31

## 日期说明

`2026-09-01` 是预注册的 WSL run/version 日期；当前代码准备与提交发生在 2026-08-31。

## 不变量

- 本版本独立存在于 `08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/`。
- 不修改、复制覆盖或向 `13_base_method/grip-exp/` 写入实验代码。
- 数据只读取 v0.1 exact-hop 三个 split。
- v0.2 只研究 oracle routing，不实现 learned router。
- 所有方法 total rank 固定为 8，target modules 固定为 `down_proj/up_proj/gate_proj`。
- 结果必须按 `RUN_ID/method/seed` 留存。

## 方法的新颖性边界

核心不是 rank split、expert routing 或正交初始化，而是：

```text
ordered cumulative depth subspaces
+ ordinal/nested activation
+ depth-local credit assignment
```

## 重要设计修正

固定 random group order 只是参数组重命名，仍保持 nested prefix，理论上不应被要求下降。它现在作为 symmetry control；真正的负对照是 `non_nested_random_masks`。

## 当前证据

macOS 静态验证覆盖：

- exact-hop split 数量和无泄漏；
- 8 个方法的 mask 定义；
- ordered prefix 嵌套性；
- local credit 前向/梯度 mask；
- equal-rank 配置；
- dry-run 命令矩阵；
- Python 语法。

WSL runtime preflight 额外验证：

- CUDA 可见；
- monolithic 与 structured trainable parameter 数相等；
- depth=3 local credit 只给 G3 梯度；
- 普通 prefix 给 G1/G2/G3 梯度。

## 下一动作

在 WSL 3090 上运行：

```bash
conda activate guardenv
cd 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01
RUN_ID=wsl3090_oracle_prefix_smoke_01 bash configs/run_wsl_smoke.sh
```

将完整 `results/runs/<RUN_ID>/` 提交远程，再根据 `REPORT.md` 决定停止还是扩展三个 seeds。

## Smoke run outcome (2026-09-01)

使用 conda `guardenv` 在 WSL2 RTX 3090 上完成注册的 v0.2 smoke：

- `RUN_ID=wsl3090_oracle_prefix_smoke_01`
- PyTorch `2.12.0+cu130`，Transformers `4.57.3`
- CUDA 可用，GPU 为 `NVIDIA GeForce RTX 3090`，compute capability `8.6`
- 8/8 methods 完成，单注册 seed `42`；数据为 train/validation/test = `716/152/156`
- 所有方法 trainable parameters 均为 `3,317,760`，满足 equal-rank 公平性
- smoke decision：`PRELIMINARY_STOP`

Test accuracy / depth accuracy：

| method | overall | d1 | d2 | d3 | d4 | macro-hop | worst-hop |
|---|---:|---:|---:|---:|---:|---:|---:|
| `monolithic` | 0.3462 | 0.1538 | 0.2821 | 0.5128 | 0.4359 | 0.3462 | 0.1538 |
| `static_split` | 0.3782 | 0.1795 | 0.3077 | 0.5385 | 0.4872 | 0.3782 | 0.1795 |
| `flat_oracle` | 0.2949 | 0.1538 | 0.1795 | 0.4103 | 0.4359 | 0.2949 | 0.1538 |
| `ordered_prefix` | 0.3205 | 0.1538 | 0.2051 | 0.4615 | 0.4615 | 0.3205 | 0.1538 |
| `ordered_prefix_local_credit` | 0.2949 | 0.1282 | 0.1795 | 0.4103 | 0.4615 | 0.2949 | 0.1282 |
| `permuted_depth_prefix` | 0.3462 | 0.1795 | 0.2308 | 0.5128 | 0.4615 | 0.3462 | 0.1795 |
| `random_group_order` | 0.3205 | 0.1538 | 0.2051 | 0.4359 | 0.4872 | 0.3205 | 0.1538 |
| `non_nested_random_masks` | 0.2949 | 0.1026 | 0.1538 | 0.4359 | 0.4872 | 0.2949 | 0.1026 |

Gate audit：`ordered_prefix` 未超过 `monolithic`（-2.56 pp）、未超过
`permuted_depth_prefix`（-2.56 pp），3/4-hop 平均相对 monolithic 为 -1.28 pp，
因此未满足注册 gate。它超过 `flat_oracle` 和 `non_nested_random_masks`，且 1-hop 未下降，
但不足以输出 `PRELIMINARY_GO`。按预注册规则不实现 learned router，也不运行 full seeds。

机制探针已生成于各 method 的 `run_summary.json`：包括 depth gradient cosine、gradient norm
和 group knockout。`monolithic/static_split/ordered_prefix/ordered_prefix_local_credit` 均有
knockout 结果；各 method 的 adapter 权重只保留本机并被 Git 忽略。

可复现结果目录：

```text
results/runs/wsl3090_oracle_prefix_smoke_01/
```

提交结果时只提交 `REPORT.md`、`suite_summary.json`、`suite_metrics.csv`、各 method 的
metadata/predictions/config 等文本结果，不提交任何 `adapter_model.pt`。
