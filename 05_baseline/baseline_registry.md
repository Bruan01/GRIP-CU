# Baseline Registry

更新日期：2026-08-28

| ID | Baseline | 目的 | 参数匹配 | FLOPs 匹配 | 优先级 |
|---|---|---|---:|---:|---:|
| B0 | Base LLM, no graph adapter | 检查预训练知识泄漏 | 否 | 否 | 必须 |
| B1 | Original GRIP | 直接基础方法 | 是 | 基准 | 必须 |
| B2 | GRIP + More Reasoning QA | 排除更多训练数据因素 | 是 | 是 | 必须 |
| B3 | GRIP-Deep Unshared | 排除模型深度与参数量因素 | 是 | 可配 | 必须 |
| B4 | GRIP ComputeMatch | 排除更多测试计算因素 | 是 | 是 | 必须 |
| B5 | Self-Refine GRIP | 区分 token-space 修正与 hidden recurrence | 是 | 是 | 必须 |
| B6 | Recurrent LM without Graph LoRA | 检查执行器是否记住测试图 | 接近 | 是 | 必须 |
| B7 | RecurrentGRIP with shuffled adapter | 检查图 adapter 因果作用 | 是 | 是 | 必须 |
| B8 | Full Graph Context LLM | 显式图输入强对照 | 否 | 报告 | 正式实验 |
| B9 | Retrieved Subgraph LLM / KG-RAG | 外部检索强对照 | 否 | 报告 | 正式实验 |
| B10 | Oracle Path | 完美证据上界 | 否 | 报告 | 分析 |

## Original GRIP 复现状态

当前本地代码已经具备：

- per-graph train-then-infer；
- Qwen2.5-7B LoRA；
- NELL23K staged runner；
- CLEGR 和 Scene Graph shell 配置；
- EM/F1/Hit/LLM judge 评价。

正式开发新方法前必须完成：

1. 固定代码版本；
2. 记录环境与 GPU；
3. 核验 NELL23K 当前 35% pilot 与论文报告结果差异；
4. 至少跑通一个 CLEGR 小规模 baseline；
5. 保存每个实验的配置、日志和原始预测。

## 公平性要求

- B1 与 RecurrentGRIP 使用同一批训练样本；
- B2 与 RecurrentGRIP 使用同等 reasoning QA；
- B3 匹配 trainable parameter count；
- B4/B5 匹配推理 FLOPs 或 wall-clock budget；
- B6/B7 用于验证 graph-specific memory 的必要性；
- 不以弱文本 RAG 作为唯一检索对照。
