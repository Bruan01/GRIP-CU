# Hard-Negative Contrastive GRIP — 进展评估与 H2 验证报告

日期: 2026-09-03
范围: `GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP`
结论先行: 项目处于「设计 + 候选审计」阶段，无训练结果；增量创新点真实但薄；核心前提 H2 经零训练验证**被证伪**，建议停止结构负样本方向。

---

## 1. 项目现状

### 1.1 已完成（约 20–30%）

| 模块 | 状态 | 说明 |
|---|---|---|
| 文献调研 `RESEARCH_REVIEW.md` | ✅ | 覆盖 GRIP / SANS / SimKGC / 生成式硬负样本 / RotatE / ProGCL，明确 novelty 边界 |
| 实验设计 `EXPERIMENT_PLAN.md` | ✅ | H1–H5 假设、B0–B10 消融、success gate 完整 |
| 集成契约 `IMPLEMENTATION.md` | ✅ | Stage A–E 对接 `RecurrentGRIP` 快照 |
| 核心库 `src/hard_negative_grip/` | ✅ | 候选生成 / InfoNCE+margin loss / adapter 对比 loss / 续写打分 / MRR·Hits@K |
| 单元测试 `tests/test_core.py` | ✅ | 7 passed |
| 候选审计 `results_nell23k_audit.json` | ✅ | 160 题、0 泄漏、四家族计数：uniform 640 / tail_range 485 / path 523 / random 640 |

### 1.2 未完成（Stage B–E，论文的命门）

- ❌ 没有任何一次训练/评测跑过（`results/` 目录此前为空）。
- ❌ B0–B10 十一个变体一个都没跑。
- ❌ Stage B（续写打分）、C（自定义 loss 接入 Trainer）、D（adapter 对比）、E（评测）均只有文档，未接入代码。

### 1.3 关键前置问题（基线）

`RecurrentGRIP` 的 pilot 结果显示，用图特定 adapter 的命中率**低于**不用 adapter 的 base model：

```
k=1|adapter=correct  accuracy = 0.0453   (4.5%)
k=1|adapter=none     accuracy = 0.2266   (22.7%)
```

即当前 NELL23K 快照里图 adapter 是拖后腿的。这直接动摇 H4（图知识由 adapter 承载）这一机制假设。

---

## 2. 增量论文评估

### 2.1 创新点（组合式，单点不新）

1. **关系级结构感知负样本**（最像「方法」）：
   - `tail_range_relation`：尾实体集合与正关系重叠的「类型易混淆」错误关系。
   - `path_relation`：query head 附近 k-hop 内出现过的关系（SANS 邻居思想搬到关系预测 + LLM）。
2. **候选级对比 loss**（无创新，SimKGC 已做，只能当载体）。
3. **adapter 身份对比**（最有区分度）：正确 adapter vs 关闭/打乱 adapter 的分数 margin，回答「图知识在 LoRA 还是 base 先验」。

### 2.2 定位

- 能投的层级：**Findings / workshop / 短文的「analysis + extension」**，不是主会 full paper。
- 最薄弱的点：三个负样本家族全是确定性启发式，审稿人一句话「SANS + 关系翻转」即可拍平；H4 机制声明需要多图（CLEGR）shuffled 控制，NELL23K 单图做不了。

### 2.3 结论

创新点真实但**不构成独立卖点**，且必须建立在「结构负样本确实更难」这一前提上——该前提在第 3 节被证伪。

---

## 3. H2 验证实验（零训练打分）

### 3.1 方法

- 加载 pilot 训好的 NELL23K adapter（Qwen2.5-0.5B，LoRA r=4，layer 12，depth 2）。
- 对 audit 的 160 题，逐候选关系算 **normalized continuation log-likelihood**（前缀完全一致，只换关系）。
- 两个条件：`--control correct`（用图 adapter）与 `--control none`（关 adapter，base model）。
- 判定：负分数越高 / margin 越小 = 越难（越易与正答案混淆）。

### 3.2 结果

| 家族 | correct 负分数 | correct margin | none 负分数 | none margin |
|---|---|---|---|---|
| uniform | -4.035 | 2.509 | -4.646 | 3.430 |
| tail_range | -3.981 | 2.430 | -4.725 | 3.491 |
| path_local | -4.106 | 2.591 | -4.758 | 3.552 |
| random | -4.489 | 2.964 | -5.300 | 4.085 |

配对检验「结构家族是否比 uniform 更难」：

| 条件 | tail_range 更难 | path 更难 |
|---|---|---|
| correct adapter | 50.7% | 42.5% |
| base model（none） | 47.4% | 43.2% |

### 3.3 根因判定

1. **不是 adapter 没学好**：base model 本身就认为 tail_range / path 不比 uniform 难（47% / 43%），说明「类型易混淆」「路径局部」这两个结构信号在 NELL23K 关系预测里不构成更强混淆性。
2. **adapter 只做微弱扰动**：tail_range 从 47.4% → 50.7%（噪声内），path 无变化。
3. **旁证**：adapter 把所有负样本分数都比 base model 抬高（uniform +0.61、tail_range +0.74、path +0.65、random +0.81），判别力下降，与 pilot「correct 4.5% < none 22.7%」一致。

### 3.4 结论

**H2 被明确证伪，根因是负样本设计本身不成立，而非训练问题。** 继续 B2–B10 训练大概率无 H1 收益，不建议投入 GPU。

---

## 4. 建议

1. **停止结构负样本方向**，把本次阴性结果作为诚实实验记录保留（已记录在 `results/`）。
2. 若仍想继续，唯一有据可依的路径：
   - **重定义负样本**：先诊断 tail_range / path 为何不更难，再决定改定义或放弃；
   - **转路线 B**：adapter 身份对比做机制分析（不依赖负样本硬度），但需先解决「adapter 为何比 base model 差」。
3. 下一个可执行的低成本诊断：**为什么 adapter 判别力低于 base model**（例如检查训练 loss 曲线、LoRA 尺度、graph context 是否被正确内化）。

---

## 5. 产物清单

| 文件 | 内容 |
|---|---|
| `scripts/score_h2_gate.py` | 零训练打分脚本，支持 `--control correct/none` |
| `results/h2_gate_results.json` | correct adapter 逐题逐家族分数 |
| `results/h2_gate_results_noadapter.json` | base model 逐题逐家族分数 |
| `results/h2_gate_verdict.md` | H2 对照与判读 |
| 本文件 | 整体进展与评估报告 |
