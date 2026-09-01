# 08 Experiments

Method 回答“算法是什么”。
Experiment 回答“怎么证明它有效”。

## 子目录
- 01_main_results/
- 02_ablation/
- 03_hyperparameter/
- 04_efficiency/
- 05_robustness/
- 06_error_analysis/
- 07_significance/

## 每次实验至少记录
- Experiment ID
- Date
- Git Commit
- Dataset
- Config
- Seed
- GPU
- Result
- Conclusion
- Next Action

## 原则
不要只记录成功实验。
失败实验同样重要。

## 版本化方法实验

- `PriorityDistillGRIP/`：当前候选主线；v0.1 用 NELL23K strict exact-hop 测试训练期 oracle path priority 能否在 graph-free 推理时超过 answer-only 与等 token 对照，等待 WSL 3090 seed-42 smoke。
- `StructuredLoRA/`：历史结构化秩实验；v0.2 oracle-prefix smoke 为 `PRELIMINARY_STOP`，停止 learned router，完整证据继续保留。
- `RecurrentGRIP/`：历史递归执行实验快照与诊断结果，继续保留。
- `FactorGRIP/`：历史 candidate-energy/factorization probe，继续保留。
- Original GRIP 保持在 `13_base_method/grip-exp/`，实验方法不直接修改基线仓库。
- 方法代码或配置发生改变时创建新版本目录，不覆盖已运行版本。
- 当前采用 macOS 开发与 CPU 数据审计、Windows WSL2 + RTX 3090 训练和 GPU 推理的双环境流程；平台环境不写入 Original GRIP。
