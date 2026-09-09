# H2 Gate 结果（零训练打分）

日期: 2026-09-09（重跑）；2026-09-03（旧跑，已作废为证伪依据）
脚本: `scripts/score_h2_gate.py`
题目: candidate audit 的 160 题（train 64 / validation 32 / test 64）
指标: 每个候选关系的 normalized continuation log-likelihood（teacher-forced，前缀完全一致）

## 这次改了什么

旧跑用的是失败的 attention-LoRA pilot adapter（correct EM 4.5% < none 22.7%），且打分前缀没有套 GRIP 的 `"Given the context graph titled ..."` 模板。那次「H2 被证伪」因此不干净。

本次默认 `--recipe storage`：

- Adapter: `quick01_storage_quick/adapter`（MLP LoRA 全层，correct EM 47.8% > none 22.5%）
- 打分前缀与 GRIP 训练/评测一致
- `--control none` 关掉同一份 adapter，作为 base-model 对照

旧结果仍保留在 `h2_gate_results.json` / `h2_gate_results_noadapter.json`，不再作为判定依据。

## 结论：path_local 仍被证伪；tail_range 仅有弱信号，不够开训练

（负分数越高 / margin 越小 = 越难）

| 家族 | correct 负分数 | correct margin | none 负分数 | none margin |
|---|---|---|---|---|
| uniform | -1.658 | 1.373 | -4.588 | 3.777 |
| tail_range | -1.625 | 1.330 | -4.573 | 3.727 |
| path_local | -1.727 | 1.453 | -4.708 | 3.899 |
| random | -1.990 | 1.706 | -5.164 | 4.353 |

### 配对检验：结构家族是否比 uniform 更难

| 条件 | tail_range 比 uniform 更难 | path 比 uniform 更难 |
|---|---|---|
| correct adapter（MLP storage） | **59.9%**（91/152） | **43.2%**（63/146） |
| base model（none） | **47.4%**（72/152） | **43.8%**（64/146） |

按 split（correct adapter）：

| split | tail_range | path |
|---|---|---|
| train | 60.3%（35/58） | 43.4%（23/53） |
| validation | 59.4%（19/32） | 40.6%（13/32） |
| test | 59.7%（37/62） | 44.3%（27/61） |

二项检验（相对 50%）：overall tail_range z=2.43，双侧 p≈0.015；**test-only** 37/62，z=1.52，双侧 p≈0.13。path 在所有 split 都稳定低于 50%。

对照旧跑（坏 adapter，仅作历史）：tail_range 50.7% / path 42.5%。换能存图的 adapter 后，path 完全没变；tail_range 从抛硬币抬到约 60%。

### 根因判定

1. **path_local 这个家族可以关掉。** 能存图的 adapter 和 base model 都认为它比 uniform 更容易。邻域 k-hop 关系在 NELL23K 关系预测里不是混淆轴。
2. **tail_range 有方向正确、幅度很弱的信号。** 相对 base 的 47.4% 抬到 59.9%，且三个 split 一致；但 test 单独不显著，Hits@1 只从 94.4% 掉到 92.1%。这不够支撑「结构负样本明显更难」。
3. **adapter 确实把图写进去了**（与旧跑相反）：正样本分数从 none 的 -0.811 升到 -0.285，负样本分数整体也升高约 +2.9。判别力的绝对 Hits@1 两边都极高（≈92–95%），说明「在一小撮候选上排续写分数」本身太容易，不能外推到生成 EM。
4. **题目文本自带 10-way 候选**，audit 里另造的负关系不一定出现在这 10 个里。这会系统性压低未列出负样本的分数，让所有家族的 Hits@1 虚高。相对比较（谁比谁难）仍可用，绝对 Hits@1 不能当训练收益预期。

## 最终判断

用能存图的 adapter 重测之后，**H2 没有被救活到可以开 B2–B10 训练的程度**：

- path_local：明确失败，不要再当硬负样本家族。
- tail_range：弱阳性，先诊断「尾实体集合重叠是否真的对应类型混淆」，再决定是否重定义；不要直接投入对比训练。
- 旧结论里「base model 也证伪所以设计本身不成立」那一句不成立——base model 看不见图结构，不能用来否证结构负样本。真正干净的否证对象只剩 path_local；tail_range 仍是未决的弱信号。

不建议现在跑 B2–B10。下一步若继续，只做零训练诊断：用 aligned 10-way 的 `listed_relation` 对 uniform 打分，不再把 path_local / tail_range 当主硬负样本。

## 附录：换到 MLP storage 方法后，困难样本有没有打中真实错误

日期: 2026-09-09
对照: `quick01_storage_quick` 的真实生成（correct adapter EM 47.8%），不是续写分数。
重叠题: audit 160 题里有 96 题落在 quick01 的 val/test（train 64 题 quick01 没有生成记录）。

这 96 题上 quick01 对 37、错 59。

| 检查 | 结果 |
|---|---|
| 模型答错时，预测关系落在 tail_range 家族 | **2/59 = 3.4%** |
| 落在 path_local | **1/59 = 1.7%** |
| 落在 uniform | **0/59** |
| 落在 random | **2/59 = 3.4%** |
| 题目 10-way 干扰项被 tail_range 覆盖 | 16/864 = **1.9%**（16/96 题有交集） |
| 被 path_local 覆盖 | 15/864 = **1.7%** |
| 错题里预测仍在 10-way 列表内 | 29/59 |
| 错题里预测跑出列表（另造关系） | 30/59 |

全体 640 题同样：334 错里 52% 选了列表内干扰项，48% 生成了列表外关系（常见是前缀混淆，如 `animalistypeofanimal` → `animalistypedegree`）。

H2 续写硬度按生成对错切开（storage adapter）：答对的题 tail_range 比 uniform 更难 57%；答错的题 61%。path 在答错题上正好 50%。困难样本并没有集中出现在模型真正失败的题上。

**判断：换到能存图的同一套 MLP storage 方法之后，当前构造的结构负样本仍然无效。** 原因不是 adapter 没学好，而是负样本不在模型的决策集合里——模型在 10-way 列表上选，或另造近形关系；audit 另造的 tail_range / path 几乎从不等于这两种错误。续写分数上的「稍难一点」不能外推成训练信号。

## 附录 2：对齐官方 10-way 之后如何改负样本家族

日期: 2026-09-09

- 官方 seed=2026 列表与 RecurrentGRIP 列表 **0 题完全相同**（pilot Jaccard 0.074）。
- 用模型当时看到的 10-way 做 `listed_relation`：**174/174** 列表内错误被覆盖。
- 用官方列表去套已经跑完的 quick01：**9/174**。所以对齐是给以后的评测/训练用，不能挽救旧 dump 的覆盖率。
- `surface` 打中 17/70 词表内列表外错误。
- 先验 `hallucinated` 打中 **0/90** 真 OOV。真 OOV 走约束解码或 rollout 挖掘，不走结构负样本。

对齐后的审计：`results_nell23k_audit_aligned.json`（listed 1440 = 160×9，0 泄漏）。覆盖率：`results/error_coverage_decision_set.json`。

## 原始数据

- storage correct adapter: `results/h2_gate_results_storage.json`
- storage base model: `results/h2_gate_results_storage_noadapter.json`
- 旧 pilot（坏 adapter，仅对照）: `results/h2_gate_results.json`、`results/h2_gate_results_noadapter.json`
