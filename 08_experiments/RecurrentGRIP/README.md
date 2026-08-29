# RecurrentGRIP Experiment Registry

本目录按不可覆盖的版本快照保存 RecurrentGRIP 实验。`13_base_method/grip-exp`
只作为 Original GRIP 基线来源，不直接加入实验方法改动。

## 版本命名

```text
vMAJOR[_MINOR]_<mechanism>_YYYY-MM-DD/
```

版本复制脚本会排除 `.venv/`、`model_cache/`、`outputs/`、旧 `results/`、旧 `logs/` 和 Python bytecode，避免把 macOS/WSL 平台环境或旧实验结果带入新版本。

每个版本至少包含：

- `README.md`：假设、边界、运行状态和结论；
- `SOURCE_BASELINE.md`：来源代码及 commit；
- `PATCH_MANIFEST.md`：相对 Original GRIP 的文件变更；
- `grip-exp/`：该版本可运行代码快照；
- `configs/`：固定参数；
- `results/`：原始输出和汇总；
- `logs/`：运行日志。

## 已登记版本

| 版本 | 日期 | 机制 | 状态 |
|---|---|---|---|
| `v1_fixed_depth_2026-08-28` | 2026-08-28 | CLEGR-first 单层共享参数固定深度递归执行 | 已冻结；保留为严格 hop 机制版本 |
| `v1_1_nell23k_first_2026-08-29` | 2026-08-29 | NELL23K-first fixed-depth executor 与结构距离分析 | macOS 验证中；待 WSL2 RTX 3090 smoke |
