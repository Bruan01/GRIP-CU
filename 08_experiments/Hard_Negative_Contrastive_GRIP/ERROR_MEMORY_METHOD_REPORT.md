# ErrorMemory-9 方法落地报告

日期：2026-10-19
分支：`codex/boundary-negative-recurrentgrip`
父线：`origin/main`
并行对照：`origin/recurrentgrip-nell23k-20260904`
基线：原始 GRIP，代码位于 `13_base_method/grip-exp`

## 方法定位

本分支只研究“候选关系学习阶段的负样本质量”。原始 GRIP 的 graph-context 训练、生成目标、LoRA 训练方式和评估格式保持不变。新增约束为候选关系 InfoNCE；新增采样器为 ErrorMemory-9。

ErrorMemory-9 将三类信号合并到一个冻结的候选决策集：

```text
B1 train-only generation errors
        + B1 candidate score violation
        + uniform probability floor
        -> fixed K=9 candidate bank
        -> original GRIP candidate InfoNCE
```

候选池剔除训练图中同一实体对的已知真实关系。默认参数为 `K=9`、`rho=1/3`、`alpha=1`、`beta=1`、`tau=0.25`。冻结阶段只读 train split；valid/test 不进入错误记忆、分数池或过滤事实。

## 与基线的关系

基线 arm 是原始 GRIP/B1 generation-only；Random-9 + InfoNCE 是计算匹配控制；ErrorMemory-9 + InfoNCE 是主方法。主结果应同时报告自由生成 EM、闭集候选 MRR/Hits、wrong-in-list、OOV、候选有效率和采样审计，而不能只报告一个 EM。

## 代码入口

- `error_memory_sampling.py`：计算 gold-conditioned 错误记忆、score violation 和混合采样概率；
- `listed_training.py`：保持 generation loss，叠加候选 InfoNCE；
- `mine_rollout_hard_negatives.py`：冻结 B1 训练集生成错误；
- `score_relation_candidates_offline.py`：冻结 B1 训练集候选分数；
- `mine_error_memory_negatives.py`：CPU 冻结 bank；
- `verify_error_memory_wiring.py`：验证来源、question ID、训练过滤和 K=9；
- `train_listed_contrastive.py`：从原始 GRIP baseline 子模块加载模型、训练候选阶段并评估标准 graph record；
- `run_error_memory_wsl.sh`：WSL 分阶段入口和 SwanLab 记录。

## 证据边界

当前只完成代码和入口校验，没有运行实验。因此只能把方法写成“把冻结的 GRIP 训练错误和分数违例转化为固定大小的候选决策集”，不能写成已经提升性能。训练前需要保存实际 branch HEAD、baseline 子模块 commit、数据/adapter hash、环境、seed、GPU、原始输出和复现命令。
