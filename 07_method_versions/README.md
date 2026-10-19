# Method Versions

更新日期：2026-10-19

当前方法版本只围绕 Original GRIP 的候选关系学习维护：

- `v0_baseline/`：Original GRIP/B1；
- `v1_random9_infonce/`：Random-9 + InfoNCE；
- `v2_errormemory9_infonce/`：ErrorMemory-9 + InfoNCE。

实际实现和测试统一位于 `08_experiments/Hard_Negative_Contrastive_GRIP/`。每个版本必须记录候选协议、训练预算、数据集、seed、adapter provenance 和证伪结果；不再创建 RecurrentGRIP 或递归 smoke 版本。
