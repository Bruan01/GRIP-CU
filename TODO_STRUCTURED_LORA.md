# StructuredLoRA 下一步 TODO

更新日期：2026-08-31

## 当前任务

在 WSL2 + RTX 3090 上创建并运行：

```text
08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/
```

完整交接 Prompt：

- [`08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/NEXT_STEP_WSL_PROMPT.md`](08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/NEXT_STEP_WSL_PROMPT.md)

## 运行前

```bash
git pull --ff-only
cd 08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31
bash configs/check_mac_static.sh
```

应看到：

```text
8 tests PASS
34,216 support-depth labels
1,024 exact-hop tasks
GO_ORACLE_PREFIX
```

## v0.2 最小比较

```text
monolithic rank=8
static split 4x2 all-on
flat oracle expert 4x2
ordered oracle prefix 4x2
ordered prefix + depth-local credit
permuted depth labels
random group order
```

## 禁止事项

- 不修改 `13_base_method/grip-exp/`；
- 不覆盖 v0.1；
- 不实现 learned router；
- 不只比较 S-LoRA rank=8/12 和 GRIP rank=4；
- 不把 NELL23K support depth 写成 ground-truth query hop；
- 不把 `>4_or_unreachable` 写成全局不可达；
- 不在 oracle prefix 失败后继续堆模块。
