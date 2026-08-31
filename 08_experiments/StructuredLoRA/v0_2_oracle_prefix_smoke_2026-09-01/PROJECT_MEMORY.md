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
