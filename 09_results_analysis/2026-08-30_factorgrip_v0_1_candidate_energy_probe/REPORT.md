# FactorGRIP v0.1 Candidate-Energy Probe：独立重算、完整性审计与下一步决策

更新日期：2026-08-30  
远程 HEAD：`2994d43 feat: finalize FactorGRIP candidate-energy probe`  
源 run：`wsl3090_nell23k_candidate_energy_20260830_01`

## 1. 结论先行

**本轮不支持把 Candidate-Energy / ScoreGRIP 继续发展为独立刷榜创新点；保留它作为 FactorGRIP 后续实验的固定解码器和诊断基线。继续 FactorGRIP 主线，但下一步必须转向参数存储与结构化分解，而不是继续堆解码技巧。**

原因有两层：

1. **解码器确实修复了大量输出失败。** 在 correct adapter 的 test 上，score decoder 从 `11/64=17.19%` 提升到 `22/64=34.38%`，增加 `17.19 pp`，配对 McNemar `p=0.00342`，bootstrap 95% CI 为 `[+7.81,+28.13] pp`。
2. **但提升不是 GRIP 参数记忆带来的。** no-adapter + score 达到 `24/64=37.50%`，反而高于 correct-adapter + score 的 `22/64=34.38%`。correct−none 为 `-3.13 pp`，95% CI `[-10.94,+4.69] pp`，McNemar `p=0.6875`。validation 方向更差：`37.50%` 对 `53.13%`，correct−none=`-15.63 pp`。

因此，当前最稳妥的机制解释是：**候选解码修复的是自由生成与候选选择之间的 decoder mismatch；现有 K1 graph adapter 没有显示出相对基础模型的可用存储/检索增益。**

这次实验并没有真正实现或否定 FactorGRIP。它否定的是“只换候选解码器就能形成主方法”的路线，并进一步说明 FactorGRIP 必须直接处理 storage/interference。

## 2. 拉取与运行产物核验

本地分支：

```text
wsl/nell23k-smoke-20260829
```

拉取后状态：

```text
HEAD = origin/wsl/nell23k-smoke-20260829 = 2994d43
```

运行产物完整：

| 项目 | 数量/状态 |
|---|---:|
| validation questions | 32 |
| test questions | 64 |
| adapter controls | correct / none / wrong_depth |
| decoder types | free / constrained / score |
| predictions | 864 |
| candidate-score records | 288 |
| 每题候选数 | 固定 10 |
| 重复 prediction key | 0 |
| 重复 score key | 0 |
| target 位于候选集 | 864/864 |
| candidate score 有限值 | 全部通过 |
| OOM / NaN / Traceback / timeout | 未发现 |
| run status | complete |
| 内部计时 | 402.85 s |
| 外层 GPU wall time | 429.95 s |

输入源文件仍存在于 RecurrentGRIP v1.1.1 快照中：

```text
08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/
  grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json
```

SHA256：

```text
e9b4f037150c3b58ca3421647fe53b63a15876a9a5a9ed755df7907c99e1c478
```

独立检查确认，结果中的 question id、target 和十候选与该输入源一致；没有用模型输出伪造 ground truth，也没有归一化分数冒充 accuracy。

## 3. 独立重算的核心结果

### 3.1 Test 主结果

| Adapter | Free | Constrained | Score |
|---|---:|---:|---:|
| correct K1 | 11/64 = 17.19% | 22/64 = 34.38% | 22/64 = 34.38% |
| none | 16/64 = 25.00% | 24/64 = 37.50% | 24/64 = 37.50% |
| wrong-depth K2→K1 | 4/64 = 6.25% | 24/64 = 37.50% | 23/64 = 35.94% |

关键配对比较：

| 比较 | 差值 | McNemar p | Bootstrap 95% CI |
|---|---:|---:|---:|
| correct score − correct free | +17.19 pp | 0.00342 | [+7.81,+28.13] pp |
| correct constrained − correct free | +17.19 pp | 0.00342 | [+7.81,+28.13] pp |
| correct score − none score | -3.13 pp | 0.6875 | [-10.94,+4.69] pp |
| correct constrained − none constrained | -3.13 pp | 0.6250 | [-9.38,+3.13] pp |
| correct free − none free | -7.81 pp | 0.3018 | [-18.75,+3.13] pp |
| correct score − wrong-depth score | -1.56 pp | 1.0000 | [-7.81,+4.69] pp |

Score decoder 在十选一随机基线之上显著：

- correct score：`22/64`，相对 10% 随机选择的单侧精确二项概率约 `1.20e-7`；
- none score：`24/64`，对应约 `4.49e-9`。

这恰好说明：**问题和候选文本本身已给基础模型提供较强语义判别信号，而不是证明 graph adapter 存储成功。**

### 3.2 Validation 方向

| Adapter | Free | Constrained | Score |
|---|---:|---:|---:|
| correct K1 | 12.50% | 43.75% | 37.50% |
| none | 25.00% | 53.13% | 53.13% |
| wrong-depth K2→K1 | 6.25% | 50.00% | 46.88% |

correct score − none score 为 `-15.63 pp`。validation 与 test 对 adapter 效应的方向一致：都是 correct 不如 none。因此“correct adapter 至少 +5 pp”的预注册门槛明确失败。

## 4. 解码器到底修复了什么

### 4.1 主要修复是离开候选集合

Test 上 free generation 的原始字符串不在十候选中的数量：

| Adapter | Out-of-set | 比例 |
|---|---:|---:|
| correct | 40/64 | 62.50% |
| none | 38/64 | 59.38% |
| wrong-depth | 57/64 | 89.06% |

constrained 和 score 都强制输出候选，out-of-set 为 0。correct 条件下，score 相对 free 有 12 题从错变对、1 题从对变错，净增 11 题。

因此可以支持的窄主张是：

> 在该 NELL23K 96-question diagnostic subset 上，候选约束或候选序列似然能够显著减少自由生成的输出流形错误。

不能扩张为：

> GRIP adapter 中已经存有正确事实，只是原 decoder 没有读出来。

如果后一说法成立，correct adapter 应稳定超过 none，并且错误/打乱 adapter 应明显下降；当前两项都没有出现。

### 4.2 Score 与 constrained 几乎等价

Test 上：

- correct：两者均为 34.38%；
- none：两者均为 37.50%；
- wrong-depth：constrained 37.50%，score 35.94%。

这意味着当前数据没有证明完整 sequence-likelihood ranking 比简单 trie-constrained greedy generation 更优。Candidate-Energy 可以保留为稳定评估工具，但目前缺乏独立方法价值。

## 5. 归一化与资源代价

validation 选择了 `raw_sum`，独立重算确认 raw sum 在 correct-adapter validation 上也优于 length normalization：

| Split / Adapter | Raw sum | Length normalized |
|---|---:|---:|
| validation / correct | 37.50% | 31.25% |
| validation / none | 53.13% | 34.38% |
| test / correct | 34.38% | 20.31% |
| test / none | 37.50% | 32.81% |

因此选择 raw sum 并非 test 泄漏，且即使只用 correct-validation 选择也不会改变配置。

但是 raw sum 天然受候选 token 长度影响，当前结果还不能把其提升全部解释为更准确的知识能量。正式实验应同时报告：

- raw sum；
- mean log-probability；
- length penalty `alpha` 在 validation 上选择后的冻结结果；
- target/candidate token-length 分层；
- 候选顺序重排后的性能和 prediction flip rate。

资源方面，score 的单题平均计时较低，但显存峰值显著更高：

| 条件 | 平均延迟 | 最大 allocated peak memory |
|---|---:|---:|
| correct free test | 0.432 s | 约 1.01 GB |
| correct constrained test | 0.607 s | 约 1.01 GB |
| correct score test | 0.078 s | 约 4.55 GB |

score 将十候选组成 batch，一次前向吞吐高，但峰值 allocated memory 约为 generation 的 4.5 倍。这里记录的是 PyTorch allocated peak，不是 `nvidia-smi` 总显存占用。

## 6. 完整性审计中的关键问题

本地审计状态：**WARN**。数据和主 accuracy 可以使用，但部分鲁棒性/校准产物不能用于论文主张。

### 6.1 Candidate-order permutation 检查无效

runner 实际执行的是：

```python
scores = score_candidates_batched(...)
permuted = list(reversed(score_dicts))
argmax(permuted)
```

它只是反转已经计算完成的分数对象，没有：

1. 重排 question 文本中的十候选；
2. 重新 tokenize；
3. 重新前向评分。

所以 `288/288 order_permutation_stable=true` 是按构造恒真的，不能视为鲁棒性实验。更进一步，该检查使用的是 `norm_logprob`，而最终选择的 decoder 使用 `raw_sum`，即使不是恒真检查，也没有验证实际 decoder。

结果中已经出现位置偏差迹象：test/no-adapter raw scorer 对候选位置 0 预测 11 次，对位置 9 预测 0 次；当 target 位于位置 9 时为 `0/6`。这不是因果结论，但足以要求真实重排实验。

### 6.2 缺少 memory-specific negative control

`wrong_depth` adapter 仍然是在同一 NELL23K graph 上训练的 K2 adapter，只是以 K1 执行。它检查 depth mismatch，不是：

- wrong-graph adapter；
- label-shuffled adapter；
- relation-permuted adapter；
- 参数块打乱 adapter。

因此它不能独立证明预测是否来自正确图记忆。当前更有力的负证据来自 correct 直接不如 none。

### 6.3 Calibration 文件与实际 decoder 不一致

提交的 `calibration.json` 始终用 `norm_logprob` 计算 top-1/top-2 margin，但实际 score decoder 选择的是 `raw_sum`。因此原 calibration 不是最终 decoder 的校准结果。本分析重新输出了 raw-sum margin，但真正的 ECE/Brier/selective-risk 分析仍应在修复版完成。

### 6.4 运行代码 provenance 不够精确

运行环境记录的 Git commit 是父提交 `2ee0c86`，而 probe 代码和结果最终在 `2994d43` 提交。代码快照已随结果保存，因此可以审查；但不能用环境中的 commit 单独精确恢复运行时工作树。下一次应先提交代码，再运行，并在 config 中保存：

- commit hash；
- `git diff --quiet` / dirty 状态；
- 输入数据 SHA256；
- adapter model SHA256 或 manifest hash。

### 6.5 一个不影响 accuracy 的元数据不一致

`response_in_candidates` 使用原始字符串成员判断，而 accuracy 使用去标点后的 exact match。有一题输出 `concept_...`，target 为 `concept:...`，accuracy 判对但 metadata 判 out-of-set。正式报告应统一 normalization。

## 7. Go / Partial / Stop 判断

| 预注册条件 | 结果 |
|---|---|
| correct−none test ≥ +5 pp | **Fail：-3.13 pp** |
| score−free test ≥ +10 pp | **Pass：+17.19 pp** |
| validation/test 方向一致 | Pass，但 adapter 效应一致为负 |
| wrong/shuffled 低于 correct | **Fail：wrong-depth 略高于 correct；且缺少真正 shuffled control** |
| candidate-order permutation 稳定 | **未测；原检查无效** |
| 总体 Go | **False** |

最终分类：

```text
STOP_AS_STANDALONE_DECODER_IDEA
KEEP_AS_BASELINE_COMPONENT
CONTINUE_FACTORGRIP_STORAGE_MAINLINE
```

## 8. 对“刷榜论文还是机制论文”的判断

当前结果既不是刷榜结果，也还不足以成为完整机制论文：

- 只有 NELL23K；
- 只有 Qwen2.5-0.5B；
- 单 seed；
- 64 个 test questions；
- test 子集已经被多轮诊断使用；
- 没有 Original GRIP official-scale 正复现；
- 没有 FactorGRIP 实现；
- 没有等参数、等训练 token、等推理预算比较。

但它完成了一个有价值的**方法筛选门**：排除了“把候选打分包装成主创新”的低潜力路线，并把后续资源集中到 graph-memory representation 上。

适合论文的进入方式仍是：

> 机制问题发现 storage–retrieval interference → 提出结构化参数存储/路由方法 → 在等预算下获得多数据集性能提升 → 用干预实验解释提升来自何处。

也就是说，论文最终应是**有机制动机的方法论文**，而不是只写机制分析，也不是只堆工程训练技巧。

## 9. 下一步工作优先级

### P0：FactorGRIP v0.1.1 审计修复版（只做一次，控制投入）

新建独立版本：

```text
08_experiments/FactorGRIP/v0_1_1_candidate_energy_audit_2026-08-30/
```

必须修复：

1. 对每题生成至少 5 个 deterministic candidate permutations；
2. 每次真正重建 question prompt、重新 tokenize、重新前向；
3. 报告 accuracy mean/std、prediction flip rate、target-position accuracy；
4. normalization 只在 correct-adapter validation 上选择并冻结；
5. calibration 使用最终选中的 score；
6. 增加至少一个 memory-specific control：relation-label permutation、wrong-graph adapter 或 adapter-weight block permutation；
7. 先提交 clean commit 再运行，记录输入和 adapter hash。

该版本目的只是让诊断结论可引用，不再尝试通过更多 decoder 超参数把结果调成正向。

### P1：Original GRIP 正复现门

在实现 FactorGRIP 大规模版本前，先确认 monolithic Original GRIP 在其正式协议下能够稳定超过 no-adapter。否则后续方法是在一个未复现的弱基线上优化，难以形成可信刷榜论文。

最低要求：

- NELL23K 扩大到正式 train/validation/test；
- 不再使用已反复查看的 64-question diagnostic test 作为最终 test；
- Original GRIP free 与 candidate-score 两套评估都报告；
- 至少 3 seeds；
- correct、none、wrong-graph/label-shuffled controls；
- 固定 validation 选择，完整 test 一次性执行。

### P2：FactorGRIP v0.2 equal-budget storage factorization

通过正复现门后，新建：

```text
08_experiments/FactorGRIP/v0_2_equal_rank_graph_factorization_2026-08-30/
```

核心比较：

- monolithic GRIP；
- relation-family/community FactorGRIP；
- random partition；
- uniform routing；
- learned sparse routing；
- oracle routing upper bound；
- wrong/permuted routing；
- equal total LoRA rank；
- equal train QA/token；
- 固定 candidate decoder。

真正的 Go gate 应围绕 FactorGRIP−monolithic，而不是 score−free：

1. validation/test 同方向；
2. test 至少 `+5 pp`；
3. paired CI 下界接近或高于 0；
4. random partition 和 permuted routing 明显更差；
5. 参数、训练 token、推理候选和显存预算公平。

### P3：多数据集刷榜主表

NELL23K smoke 出现稳定正信号后再扩展：

1. NELL23K；
2. WN18RR；
3. CoDEx-Medium；
4. FB15K237。

Candidate-Energy 在主表中只作为统一 decoder/消融项，不作为论文标题中的独立贡献。

## 10. 可写与不可写的主张

### 当前可写

- 在该 diagnostic subset 上，候选约束和候选序列评分减少了自由生成的候选外输出；
- decoder mismatch 是当前低分的一个组成部分；
- 现有 correct K1 adapter 没有显示出相对 no-adapter 的正增益；
- 下一方法需要处理 parameter storage/interference，而不是继续只优化输出格式。

### 当前不可写

- Candidate-Energy 解码成功读取了 GRIP 参数记忆；
- FactorGRIP 已被验证；
- correct adapter 优于基础模型；
- candidate-order 鲁棒；
- NELL23K SOTA；
- 结论可推广到 GRIP 的全部数据集或更大模型。

## 11. 本次新增分析产物

```text
09_results_analysis/2026-08-30_factorgrip_v0_1_candidate_energy_probe/
├── REPORT.md
├── analyze_factorgrip_probe.py
├── analysis_manifest.json
├── integrity_audit.json
├── decision_summary.json
├── condition_metrics.csv
├── normalization_metrics.csv
├── paired_comparisons.csv
├── candidate_position_diagnostics.csv
└── failure_examples.jsonl
```

所有核心 accuracy 均从 `predictions.jsonl` 独立重算；raw/normalized score 均从 `candidate_scores.jsonl` 重算。统计分析使用纯 Python、20,000 次 paired bootstrap 和 exact two-sided McNemar。
