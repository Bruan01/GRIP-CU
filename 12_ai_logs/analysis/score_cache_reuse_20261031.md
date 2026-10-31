# 三数据集候选评分缓存核查

核查时间：2026-10-31（北京时间）。只读核查；未停止当前评分、修改实验契约或启动新训练。

## NELL

旧完整训练评分仍在：
`08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/nell23k_structure_counterfactual_3090_20261021_02_artifacts/b1_train_scores/`。
`run_stats.json` 记录3253题、644094个候选、198个关系、`checks_passed=true`。
`candidate_scores.jsonl` 文件存在，大小552842928字节。

这份旧评分的教师为 `13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter`；本次流式重算其权重SHA256：
`559f0d509fb429edd27bbb9d5e63e2894fcee3d413a2107be8c50666f88350c5`。

当前新队列使用与真实Stage1配套的B1：
`/mnt/e/GRIP-CU/artifacts/stage2_memory_retention/nell_r1/control_20261018_C1/adapter`。
本次重算权重SHA256：
`a52fd9fd416a22153206850df38f30d08b9ca01c2f68d808406b76cf29e7e8ce`，与其checkpoint_manifest记录一致。
两个权重文件均为2204583480字节，但内容不同。

同一新B1以前已有训练评分：
`/mnt/e/GRIP-CU/artifacts/hard_negative_boundary_nell/local_pair/score_smoke_10_20261019/`；其metadata与run_stats显示仅10题，不能覆盖本次2974题。
另一历史NELL `nell_relation_scores_20` 是20题test评分，不是全量train评分。

**重算原因是当前实验契约选择了新的配套B1教师，而不是旧评分丢失。**
只将3253条筛选成2974条，本身无需GPU重算。旧B1评分仍可作为另一种“冻结旧教师挖负例”的方案使用，前提是明确记录教师来源与实验变化；它不能冒充新B1的评分。

## FB 与 CoDEx-M

在仓库与 `/mnt/e/GRIP-CU` 的候选评分文件检索范围内：

- FB历史 `evaluation/*_candidate_score.jsonl` 及 `outputs/confusion_debug/i0_full` 保存了历史测试评估评分；后者metadata明确 `split=test`、2639题。不能将其冒充当前训练QA评分用于train-only挖负例。当前FB配套B1是本次恢复后新训练的权重、任务是已记录的pair-repaired训练副本。
- 尚未找到CoDEx-M当前配套B1的完整训练QA评分。这个结论限于上述检索范围，不表示所有外部磁盘都不存在。

## 复用边界

超参数搜索不要求每个trial重新做全关系评分。当前控制器每个数据集在prepare阶段生成一次评分、冻结一次bank，后续trial共享同一个bank。
控制器目前按目标目录的run_stats/metadata/wiring完成标记跳过已完成步骤，未实现跨任意历史目录的兼容缓存发现。
跨目录复用应核对教师权重身份、任务/QA、train划分、关系词表和映射、提示与评分规则以及覆盖度；只改变QA子集时可以筛选现有完整评分。

本次核查没有改变正在运行的评分或搜索策略；三数据集涨点目标仍未完成。
