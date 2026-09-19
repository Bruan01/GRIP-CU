# Hard-Negative Contrastive GRIP — 进展评估与 H2 验证报告

日期: 2026-09-19（补实验 H smoke）；2026-09-16（补 7B 全量评测）；2026-09-10（0.5B smoke）；2026-09-09（H2 用 MLP storage adapter 重测）；2026-09-03（初稿）
范围: `GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP`
对外汇报稿: `汇报_进展与实验结果.md`
结论先行: 结构负样本不在决策集合里，不要开 path/tail_range 的 B2–B10。listed 比 uniform 更难只是选项印在题目上，不是创新。0.5B smoke 未过 H1（EM 43.8% vs 51.0%，−7.3 pp）。Qwen2.5-7B 在官方 NELL23K 全量测试上过门：自由生成 EM 84.91%→89.12%（+4.21 pp），闭集 86.93%→92.05%（+5.12 pp）。那是 370 QA 词表 listed（实验 G）。同源 198 训练图负样本（实验 H）smoke 上仍高于冻结 B1，但只有 +2.08 pp，且增益全在验证集；论文主结果在更大切片出来前仍是 G。本质是 Stage 2 加 InfoNCE，不是新负采样；H5 算力对照仍缺。

---

## 1. 项目现状

### 1.1 已完成（方法门禁已过；H4 / H5 未做）

| 模块 | 状态 | 说明 |
|---|---|---|
| 文献调研 `RESEARCH_REVIEW.md` | ✅ | 覆盖 GRIP / SANS / SimKGC / 生成式硬负样本 / RotatE / ProGCL，明确 novelty 边界 |
| 实验设计 `EXPERIMENT_PLAN.md` | ✅ | H1–H5 假设、B0–B10 消融、success gate 完整 |
| 集成契约 `IMPLEMENTATION.md` | ✅ | Stage A–E 对接 `RecurrentGRIP` 快照 |
| 核心库 `src/hard_negative_grip/` | ✅ | 候选生成 / InfoNCE+margin loss / adapter 对比 loss / 续写打分 / MRR·Hits@K |
| 单元测试 `tests/test_core.py` | ✅ | 10 passed |
| 候选审计 `results_nell23k_audit_aligned.json` | ✅ | 对齐官方 10-way 后 160 题、0 泄漏；listed 1440（每题 9 个）/ hallucinated 640 |
| 决策集合覆盖 `results/error_coverage_decision_set.json` | ✅ | listed 覆盖 174/174 列表内错误；官方列表只能套中 quick01 的 9/174；OOV 0/90 |
| H2 零训练打分（storage adapter） | ✅ | 2026-09-09 重跑；详见 `results/h2_gate_verdict.md` |

### 1.2 训练与评测进度

- ✅ Stage B 打分脚本已接到 quick01 MLP adapter（`scripts/score_h2_gate.py --recipe storage`）。
- ✅ Stage C 的 listed Trainer 已接入。
- ✅ 2026-09-09：0.5B smoke，共享 Stage 1，再 fork B1 vs listed（`results/runs/20260909_listed_vs_b1_smoke_listed_vs_b1_smoke`）。生成 EM 未过门。
- ✅ 2026-09-13～14：Qwen2.5-7B 在 `grip_nell23k_tasks.json` 上同样分叉（`results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/`）。关系类 QA 从训练词表抽 9 个负样本（370 种；实验 F/G）。
- ✅ 2026-09-15：同一份 7B adapter 在 smoke 96 / pilot 640 / 官方全量 9895 题上做自由生成 + 闭集打分。全量主结果在 `results/runs/20260915_140500_qwen7b_full_decode/`。
- ✅ 2026-09-18：实验 H 把 listed 负样本换成官方 198 训练图关系（`process.py` 规则）。复用 20260913 Stage 1 / 冻结 B1，只重训 listed。240/240 步，约 13.5 小时，终态 `listed/adapter/`。smoke 生成 EM 88.54% vs 冻结 B1 86.46%（+2.08 pp）；test 打平，涨点全在 val。目录 `results/runs/20260918_qwen7b_train_graph_negatives/`。
- ⏳ 2026-09-19：listed-only pilot 640 解码进行中（B1 复用 20260915，不重解）。生成已出：86.72% vs B1 84.53%（+2.19 pp）；test +1.56 pp，不再是 smoke 上的打平。闭集还在跑。目录 `results/runs/20260919_qwen7b_pilot_listed_only/`。
- ❌ 0.5B 的 pilot **训练**未跑；7B 只是在 pilot 切片上解码。
- ❌ B2–B10 结构家族按 H2 证据不应开。Stage D（adapter 身份对比）未做。H5 算力匹配未做。

### 1.3 关键前置问题（基线，已修正）

失败的 RecurrentGRIP attention-LoRA pilot 确实是 `correct 4.5% < none 22.7%`，但那是配置错误。后续 `quick01_storage_quick`（MLP LoRA 全层）已经把这个前提翻过来：

```
correct adapter  47.8%
no adapter       22.5%
```

「图写进参数」在本环境成立。H4 不再被这条基线直接否掉；H2 必须在这份能存图的 adapter 上重测（第 3 节）。

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

结构负样本这条方法线**不构成独立卖点**：path_local 失败，tail_range 弱。当前能写的贡献是诊断（错误在决策集合，不在图结构负样本）加上一条薄扩展（生成 loss + 候选 InfoNCE）。7B 全量数字让这条扩展过了 H1 门，但审稿人仍可收成「加了排序损失」。C 会 / Findings 够格；A/B 需要 H5、第二数据集或 adapter 机制实验。详见汇报稿 §0、§4.6、§4.7。

---

## 3. H2 验证实验（零训练打分）

### 3.1 方法（2026-09-09 重跑）

- 加载 `quick01_storage_quick` 的 MLP LoRA adapter（Qwen2.5-0.5B，r=4/alpha=8，全层 `down/up/gate_proj`）。
- 对 audit 的 160 题，逐候选关系算 **normalized continuation log-likelihood**（前缀与 GRIP 训练/评测一致，只换关系）。
- 两个条件：`--control correct` 与 `--control none`。
- 判定：负分数越高 / margin 越小 = 越难。

2026-09-03 旧跑用失败的 attention-LoRA adapter + 不完整 prompt，结果保留但不作为判定依据。

### 3.2 结果（storage adapter）

| 家族 | correct 负分数 | correct margin | none 负分数 | none margin |
|---|---|---|---|---|
| uniform | -1.658 | 1.373 | -4.588 | 3.777 |
| tail_range | -1.625 | 1.330 | -4.573 | 3.727 |
| path_local | -1.727 | 1.453 | -4.708 | 3.899 |
| random | -1.990 | 1.706 | -5.164 | 4.353 |

配对检验「结构家族是否比 uniform 更难」：

| 条件 | tail_range 更难 | path 更难 |
|---|---|---|
| correct adapter（MLP storage） | **59.9%**（91/152） | **43.2%**（63/146） |
| base model（none） | **47.4%**（72/152） | **43.8%**（64/146） |

test-only：tail_range 37/62 = 59.7%（z=1.52，p≈0.13）；path 27/61 = 44.3%。

### 3.3 根因判定

1. **path_local 可以关掉**：能存图的 adapter 和 base model 都认为它比 uniform 更容易。
2. **tail_range 方向对、幅度弱**：相对 base 的 47.4% 抬到 59.9%，三个 split 一致，但 test 单独不显著，Hits@1 只从 94.4% 掉到 92.1%。
3. **adapter 这次确实把图写进去了**：正样本分数从 none 的 -0.811 升到 -0.285。旧跑「adapter 判别力下降」不再适用。
4. 题目文本自带 10-way 候选，audit 负关系不一定出现在列表里，绝对 Hits@1 虚高；相对比较仍可用。

### 3.4 结论

**H2 没有被救活到可以开训练的程度。** path_local 明确失败；tail_range 是未决的弱信号，不值得现在跑 B2–B10。旧结论里「base model 也证伪所以设计本身不成立」不成立——base model 看不见图结构。

---

## 3.5 对齐官方 10-way 之后，决策集合怎么修（2026-09-09）

官方协议：`process_raw_data.py --datasets nell23k --seed 2026`，只给 valid/test 抽 10-way。RecurrentGRIP 用另一套 per-subset RNG，同一 `question_id` 的列表几乎完全不同。

| 文件 | eval 题数 | 与官方 10-way 完全相同 | 平均 Jaccard |
|---|---|---|---|
| smoke 96 val/test | 96 | **0** | 0.081 |
| pilot 640 val/test | 640 | **0** | 0.074 |

对齐产物：`data/nell23k/recurrent_relation_prediction.aligned.json`（及 `_pilot`）。Train 题官方没有列表，保持 RecurrentGRIP 10-way。审计 `results_nell23k_audit_aligned.json`：listed **1440**（每题满 9 个干扰项）、hallucinated 640、0 泄漏。

quick01 640 题错误切成三类后，各家族打中**模型真实错答**的比例：

| 家族 | 列表内误选 174 | 列表外但在词表 70 | 真 OOV 90 |
|---|---|---|---|
| 评测题自己的 10-way / `listed` | **174/174** | 0/70 | 0/90 |
| 官方 10-way | 9/174 | 8/70 | 0/90 |
| surface | 14/174 | **17/70** | 0/90 |
| hallucinated（先验拼接） | 0/174 | 0/70 | **0/90** |
| tail_range | 11/174 | 7/70 | 0/90 |
| path_local | 12/174 | 18/70 | 0/90 |
| uniform / random | 0 | 0 | 0 |

怎么修：

1. **主负样本改成 `listed_relation`**，而且必须是评测时那份 10-way 的全部 9 个干扰项。这才能覆盖约 52% 的生成错误（列表内误选）。
2. **对齐官方列表是评测协议卫生**，不是给已经跑完的 quick01 补覆盖。quick01 用的是 RecurrentGRIP 列表；拿官方列表当负样本只能打中 9/174 次当时的列表内错误。以后 H2 / 训练 / 生成评测都走 aligned 文件。
3. **`surface_relation` 作为第二家族**，覆盖「生成了一个真实关系但不在 10-way 里」（17/70）。
4. **不要指望先验幻觉字符串召回真 OOV。** 90 个真 OOV 里编辑距离 ≤2 的词表近邻只有 8 个，stem+suffix 精确命中 0。这是解码问题：要么生成时约束在 10-way 上，要么用模型自己的 rollout 错误当负样本。结构家族和拼字符串都打不中。
5. path / tail_range 不再当主硬负样本。

**仍然不要开结构负样本的 B2–B10。** aligned listed vs uniform 已打完：adapter 159/160、base 160/160 认为 listed 更难。这是预期内的卫生检查，不是创新点。

---

## 4. listed vs B1 训练与评测

细节和完整表格见 `汇报_进展与实验结果.md` §4.5–4.7。这里只保留判定。

| 实验 | 模型 | 评测 | 生成 EM | 闭集 EM | 判定 |
|---|---|---|---|---|---|
| 2026-09-09 smoke 训练 | 0.5B | 96 题 | 43.8% vs B1 51.0%（−7.3 pp） | 未做 | H1 未过门；列表内 26→1，OOV 21→53 |
| 2026-09-13 7B 训练后 smoke 解码 | 7B | 96 题 | 90.63% vs 86.46%（+4.17 pp） | 91.67% vs 87.50% | 方向反转，样本太小 |
| 2026-09-15 pilot 解码 | 7B | 640 题 | 90.31% vs 84.53%（+5.78 pp） | 92.50% vs 86.25% | 同向 |
| **2026-09-15 官方全量** | **7B** | **测试 4944** | **89.12% vs 84.91%（+4.21 pp）** | **92.05% vs 86.93%（+5.12 pp）** | **H1 过门（370 词表 listed）** |
| 同上 | 7B | 全量 9895 | 88.82% vs 85.02%（+3.80 pp） | 91.71% vs 86.90% | 验证/测试同向；OOV 322→425，列表内 1160→681 |
| **2026-09-18 实验 H smoke** | **7B** | **96 题** | **88.54% vs 冻结 B1 86.46%（+2.08 pp）** | 未做 | 门禁仍过，但弱于 370 listed；test 64 题打平（55/64），val +6.25 pp。列表内 10→4，OOV 3→7 |

7B 实验 F/G 训练负样本来自论文 QA 词表抽样，不是官方测试 10-way，故不是测试泄露，但也不是严格的 listed 同分布训练。实验 H 改成官方 198 训练图关系后，smoke 上 listed 仍高于同一份冻结 B1，但净增益从 +4.17 pp 掉到 +2.08 pp，且全部来自验证集 2 题。同源负样本更会压列表内混淆（10→4，370 版是 7），OOV 变差（3→7）。96 题不能当方法结论。listed 多出来的候选前向 H5 仍未闭合。

---

## 5. 建议

1. **不要开 path/tail_range 的 B2–B10。**
2. aligned listed 卫生检查不能当论文贡献。
3. 不要把 0.5B 64 题掉点写成论文阴性，也不要把 7B 涨点说成结构硬负样本成功。
4. **实验 H smoke 不能当方法结论。** listed > B1 仍成立，但只多对 2 题、全在 val；论文主结果仍是实验 G 的 370 词表全量。下一步是 listed-only 的 pilot 640 / 全量解码（B1 用 20260915 冻结预测，不重解）。
5. 对外数字以汇报稿 §0 和 §4.7 为准；H 的限制写在 §4.8。

---

## 6. 产物清单

| 文件 | 内容 |
|---|---|
| `scripts/align_official_nell23k_lists.py` | 把 val/test 10-way 重写成官方 `process.py` seed=2026 列表 |
| `scripts/audit_error_coverage.py` | 负样本家族 vs 真实生成错误（列表内 / 词表内列表外 / OOV） |
| `data/nell23k/*.aligned.json` | 对齐后的 smoke / pilot / full 题目 |
| `results_nell23k_audit_aligned.json` | 对齐后的候选审计（listed 每题 9 个） |
| `results/error_coverage_decision_set.json` | quick01 334 个错误上的家族覆盖率 |
| `scripts/score_h2_gate.py` | 零训练打分脚本，`--recipe storage/pilot`，`--control correct/none` |
| `results/h2_gate_results_storage_aligned.json` | 官方 10-way 对齐后 storage adapter 分数（listed vs uniform） |
| `results/h2_gate_results_storage_aligned_noadapter.json` | 同一套 aligned 题目的 base-model 分数 |
| `results/h2_gate_results_storage_noadapter.json` | 关掉同一 adapter 的 base-model 分数 |
| `results/h2_gate_results.json` | 旧 pilot adapter 分数（仅对照） |
| `results/h2_gate_results_noadapter.json` | 旧 base-model 分数（仅对照） |
| `results/h2_gate_verdict.md` | 重跑对照与判读 |
| `results/runs/20260909_listed_vs_b1_smoke_listed_vs_b1_smoke/` | 0.5B smoke：共享 S1 + B1 vs listed |
| `results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/` | 7B 训练：论文任务文件上的 S1 + B1 / listed adapter |
| `results/runs/20260915_101030_qwen7b_pilot_decode/` | 7B adapter 在 640 题切片上的生成 + 闭集 |
| `results/runs/20260915_140500_qwen7b_full_decode/` | 7B 官方 9895 题主结果，`comparison_closed_set.json` |
| `汇报_进展与实验结果.md` | 对外汇报稿（含 7B 全量数字与实验 H smoke） |
| `results/runs/20260917_qwen7b_train_graph_negatives/` | 实验 H 第一次 listed 重训：76/240 中断，无 adapter |
| `results/runs/20260918_qwen7b_train_graph_negatives/` | **实验 H**：同源 198 训练图负样本，终态 listed adapter + smoke vs 冻结 B1 |
| `configs/run_listed_only_decode.sh` | listed-only 更大切片解码；B1 复用 20260915 预测 |
| 本文件 | 整体进展与评估报告 |
