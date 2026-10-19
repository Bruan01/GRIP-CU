# GRIP-CU Research Status

更新日期：2026-10-19

## 当前主线

当前只研究原始 GRIP 的候选关系学习改进，不再运行或维护 RecurrentGRIP、递归执行器和 0.5B smoke 旧流程。

固定比较为：

- **Original GRIP/B1**：原始 graph-context/QA generation-only 基线；
- **Random-9 + InfoNCE**：从训练关系词表随机采样 9 个合法负样本；
- **ErrorMemory-9 + InfoNCE**：由冻结 B1 的训练集生成错误、候选分数违例和均匀概率下限共同构造 9 个负样本。

方法只改 Stage 2 候选关系学习：

```text
original GRIP generation loss
    + candidate InfoNCE(gold relation, 9 negatives)
```

Stage 1 graph-context 训练、LoRA 结构、prompt 和自由生成评估保持不变。

## 数据集

首轮固定使用：

- `NELL23K`：当前原始 GRIP 复现锚点；
- `FB15K237-2`：仓库中原始 GRIP 的 FB15K-237 文本关系版本。

本地数据统一放在 `04_benchmark/data/`，该目录已加入 `.gitignore`，不提交大文件和 task JSON。

## 当前假设

在相同模型、Stage 1 adapter、Stage 2 预算、候选数 `K=9`、seed 和生成目标下，B1 自身暴露的训练集边界错误应比纯随机负样本更接近候选决策边界。

可证伪预测：

1. ErrorMemory-9 的 closed-set relation MRR/Hits 优于 Random-9；
2. free-generation EM 不低于 Original GRIP/B1；
3. 若只改善候选排序而不改善自由生成，只能声称候选辨别改善；
4. 若不优于 Random-9，则保留完整失败记录，不宣称 hard negative 必然有效。

## 当前状态

- [x] 原始 GRIP/B1、Random-9、ErrorMemory-9 方法目录和测试；
- [x] 删除 RecurrentGRIP smoke 旧实验树及无关候选方法目录；
- [x] NELL23K、FB15K237-2 数据整理到 `04_benchmark/data/`；
- [x] `/mnt/e/GRIP-CU` Qwen2.5-7B 基座、NELL adapter 和 FB15K237-2 历史资产核对；
- [x] 两个数据集的 read-only preflight；
- [ ] 恢复或重新训练可用的 FB15K237-2 B1/S1 adapter 权重；
- [ ] B1 train-only rollout 和全关系候选 score；
- [ ] ErrorMemory-9 bank freeze、wiring 审计；
- [ ] Original GRIP/B1、Random-9、ErrorMemory-9 配对训练；
- [ ] 结果分析和论文结论。

## 证据边界

训练前必须记录 branch HEAD、Original GRIP 子模块 commit、数据和 adapter hash、模型缓存、seed、GPU、训练预算以及复现命令。valid/test 不能进入 ErrorMemory、候选分数或已知事实过滤。实验完成前不声称性能提升、OOV 修复或通用 KG 负采样理论。
