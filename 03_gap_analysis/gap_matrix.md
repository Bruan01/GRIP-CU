# Gap Matrix

更新日期：2026-10-19

| 已有能力 | 当前缺口 | 本项目测试 |
|---|---|---|
| GRIP 能完成 graph-context/QA 生成 | 候选负样本未利用 B1 的错误结构 | ErrorMemory-9 |
| 随机负样本易复现 | 可能远离候选决策边界 | Random-9 对照 |
| score-based hard negative 可获得高分候选 | 可能放大偶然错误或真事实 | train-only violation + known-fact filter |
| InfoNCE 可用于候选关系判别 | 未知其在 Original GRIP Stage 2 的净收益 | B1 / Random-9 / ErrorMemory-9 配对比较 |

## 证据边界

该 gap 只涉及候选关系学习质量，不涉及递归执行器、动态停止、显式图检索或通用 KG 负采样理论。
