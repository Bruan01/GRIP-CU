# PriorityDistill-GRIP v0.1.6 Results Analysis

- **分析日期**：2026-09-02（Asia/Shanghai）
- **环境**：WSL2 + RTX 3090 24GB + conda `guardenv`
- **模型**：本地 `Qwen/Qwen2.5-0.5B-Instruct`
- **数据**：NELL23K exact-hop，train/validation/test = 716/152/156
- **实验目录**：`v0_1_6_joint_trace_answer_2026-09-02`
- **运行目录**：`results/runs/wsl3090_v016_joint_20260902_01`

## 一句话结论

这次不是“完全没收敛”。模型在训练集上已经学会了不少东西，但主要表现为**记忆训练样本 + 学会输出格式**；到了没有图证据的新组合上，答案泛化仍然不稳定。更大的生成长度修复了一个严重的评估问题，但没有把方法变成可靠的组合推理器。

## 1. 先区分两个实验结果

原始运行的 `max_new_tokens=24` 对结构化 trace 输出太短：

- train 中 605/716 条目标超过 24 token；
- validation 中 129/152 条超过 24 token；
- test 中 130/156 条超过 24 token。

所以原始 graph-free test `3/156 = 1.92%` 不能直接当作最终方法能力。它混合了真实推理错误和“输出还没写完就被截断”两种错误。

随后使用同一批 checkpoint、同一模型和同一数据，只把生成预算改为 `max_new_tokens=80`，重新做了 corrected post-hoc evaluation；没有重新训练。

## 2. corrected checkpoint curve

| checkpoint | validation answer | test answer | test trace | test joint |
|---|---:|---:|---:|---:|
| stage1 | 16.45% | 22.44% | 29.49% | 5.13% |
| stage2_epoch1 | 22.37% | 23.08% | 29.49% | 4.49% |
| stage2_epoch2 | 20.39% | 28.21% | 28.85% | 5.77% |
| stage2_epoch3 | 25.66% | 28.85% | 30.77% | 7.05% |
| stage2_epoch4 | 21.05% | 32.05% | 31.41% | 8.97% |
| **stage2_epoch5** | **27.63%** | **29.49%** | **31.41%** | **7.05%** |
| stage2_epoch6 | 26.32% | 30.77% | 36.54% | 11.54% |
| stage2_epoch7 | 25.66% | 35.90% | 33.97% | 10.90% |
| stage2_epoch8 | 26.97% | 39.74% | 40.38% | 15.38% |

按照预先规定的“只看 validation 选 checkpoint”规则，正式 checkpoint 是 `stage2_epoch5`，所以正式 graph-free test 是 **46/156 = 29.49%**。`stage2_epoch8` 的 39.74% 只能作为训练曲线诊断，不能因为 test 高就倒过来选它；它的 validation 只有 26.97%，低于 epoch5 的 27.63%。

这个曲线告诉我们两点：

1. 生成长度修复很重要：从原始 1.92% 到 corrected curve 的 20%--40% 区间，说明 24 token 确实严重污染了评估。
2. 训练不是稳定单调变好：validation 在 epoch 5 达到峰值后波动，test 却在 epoch 8 继续升高。152/156 个样本的测试规模较小，不能用最后 test 峰值替代 validation 选择。

## 3. stage2_epoch8 checkpoint audit：训练集和“给证据”对照

使用 corrected `max_new_tokens=80` 对 `stage2_epoch8` 做了 post-hoc audit：

| 条件 | 数据 | answer accuracy |
|---|---|---:|
| graph-free | train | **572/716 = 79.89%** |
| graph-free | test（curve） | **62/156 = 39.74%** |
| gold trace，terminal masked | validation | 54/152 = 35.53% |
| gold trace，terminal masked | test | 57/156 = 36.54% |
| gold trace，terminal visible | validation | 78/152 = 51.32% |
| gold trace，terminal visible | test | 82/156 = 52.56% |

说明：audit 额外跑了 train 和 gold-trace 条件；graph-free validation/test 采用 corrected checkpoint curve 的统一输出。因此 stage2_epoch8 的 graph-free test 正式记为 **62/156 = 39.74%**。

训练集远高于测试集，说明有明显的记忆/组合泛化鸿沟。terminal 可见时比 terminal masked 高约 16 个百分点，说明实体终点恢复和结构化生成仍是主要瓶颈之一。`gold_trace_unmasked` 是 oracle-only sanity check，不是 graph-free 结果。

## 4. 候选打分诊断

之前的 teacher-forced candidate ranking 不要求模型自由生成完整实体，而是比较 gold answer 与三个 train distractor 的条件 log-likelihood：

- graph-free rank-1：101/156 = 64.74%；
- masked gold-trace rank-1：98/156 = 62.82%；
- oracle-evidence rank-1：88/156 = 56.41%。

这说明模型不是完全没有答案信号。自由生成 exact match 很低的部分原因，是模型经常只生成实体公共前缀，例如：

```text
预测：concept_sportsleague
真实：concept_sportsleague_nba
```

所以现在至少要把两个问题分开：

1. 模型是否在候选集合中偏好正确实体？——有一定能力；
2. 模型能否从开放词表中完整生成正确实体并同时满足 trace 格式？——目前较弱。

## 5. 回答“是不是没收敛”

不能只回答“没收敛”。训练 loss 从约 0.93 降到约 0.06，说明 teacher-forced 训练目标确实被优化；corrected train answer 79.89% 也说明模型在训练分布上已经学到很多。

更准确的判断是：

- **优化层面**：收敛到一个能拟合训练目标的解；
- **泛化层面**：没有稳定学会新实体组合上的图推理；
- **生成层面**：结构化目标较长，且实体 ID 共享前缀，容易输出不完整字符串；
- **选择层面**：validation 峰值与 test 峰值不一致，说明单次小 split 上的 checkpoint 选择有波动。

因此继续无条件增加 epoch、LoRA rank 或学习率，不是当前最有信息量的动作。

## 6. 当前对 idea 的判断

可以继续，但要缩小论断：

> v0.1.6 证明了文本 trace 联合监督能够让模型学会一部分 trace/answer 行为，也保留了非随机的候选答案信号；但它没有证明仅靠这种 LoRA + 文本 trace 目标就能稳定完成 graph-free 的组合知识内化。

不能扩大成“所有 LoRA + trace 都不行”，也不能仅凭 29%--40% 的自由生成准确率就断言必须立即换成完整显式图模块。当前更像是三个瓶颈叠加：

1. 训练/测试的组合泛化鸿沟；
2. 开放词表实体 ID 的完整生成问题；
3. trace 和 final answer 的联合格式过于脆弱。

## 7. Candidate-set 受限解码实验（已完成）

为区分“模型不会选答案”和“模型会选但自由生成写不完整”，新增了后验诊断脚本：

```text
scripts/evaluate_constrained_answers.py
priority_distill/constrained.py
```

每个 validation/test 问题使用 1 个 gold answer 加 3 个**同深度、训练集来源、答案去重后的 distractors**。生成时通过与 Transformers 4.57.3 兼容的 `CandidateTrieLogitsProcessor`，只允许生成候选实体字符串。该结果是候选集合诊断，不是完整实体词表准确率。

为避免旧候选池中同一个 answer 由多条训练样本重复出现，`priority_distill/candidates.py` 现在在选择 distractors 时强制答案去重；受限解码结果已在该修正后重新运行。

| checkpoint | validation constrained | test constrained | validation free | test free |
|---|---:|---:|---:|---:|
| **stage2_epoch5（validation-selected）** | **73/152 = 48.03%** | **78/156 = 50.00%** | 42/152 = 27.63% | 46/156 = 29.49% |
| stage2_epoch8（diagnostic） | 69/152 = 45.39% | 84/156 = 53.85% | 41/152 = 26.97% | 62/156 = 39.74% |

受限解码相对自由生成提高：

- 正式 checkpoint stage2_epoch5：test 从 29.49% 提高到 50.00%，+20.51 个百分点；
- stage2_epoch8：test 从 39.74% 提高到 53.85%，+14.11 个百分点。

这验证了“实体完整生成/格式错误”确实是重要瓶颈，但不能把 50.00% 当作完整词表 graph-free 准确率，因为候选集合中人为包含了 gold answer。

同时，stage2_epoch5 的 teacher-forced candidate scorer（同样是 gold + 3 个训练 distractors）在 test 上达到 **99/156 = 63.46%**，高于 token-by-token 的受限 greedy decoding **78/156 = 50.00%**。这说明还存在第二个解码问题：逐 token 贪心不等价于比较完整候选序列概率。下一步应实现 sequence-level candidate scoring/beam search，而不是只继续增加训练轮数。需要注意，direct answer-only baseline 在完全相同的候选诊断中为 **87/156 = 55.77% constrained**，所以当前 joint 方法的 **50.00%** 还没有超过 direct baseline。

## 8. 更新后的判断

当前证据比之前更清楚：

1. 自由生成较低，部分原因是输出不完整；
2. 候选集合受限后，准确率显著超过 25% 的随机基线；
3. 但即使在候选集合中，stage2_epoch5 也只有 50.00% constrained greedy、63.46% teacher-forced scorer，说明模型仍没有稳定解决组合关系；
4. gold trace 不会自动带来更高候选选择准确率，说明中间 trace 监督尚未可靠转化成可泛化的关系组合能力。更直接地说，在公平的 candidate-set constrained test 上，joint 为 **78/156 = 50.00%**，direct answer-only 为 **87/156 = 55.77%**；因此 v0.1.6 当前没有显示相对 answer-only 的增益。

因此当前最稳妥的结论是：

> 主要问题不是单纯“训练没收敛”，而是“部分答案信号 + 不稳定的实体解码 + 泛化不足”共同造成的。LoRA + 文本 trace 方向仍可继续验证，但下一步要先改成 sequence-level candidate scorer，并用 seen-composition / novel-composition split 检验是否真的学会组合规律。

## 9. 下一步

sequence-level 候选选择、严格 composition/entity split 和 direct baseline 的首轮公平对照已经完成。当前按信息增益排序：

1. **把 `score_deployment_candidates.py` 的 exact candidate selection 作为正式诊断协议**，并在 validation 上选择 checkpoint、test 上只报告一次；
2. **补充更大测试集或多 seed**，确认 direct 优于 joint 的差异不是 156 条测试样本的偶然波动；
3. 若需要开放词表部署，再实现真正的 constrained beam search，并与目前的 exact candidate scorer、trie-constrained greedy 分开报告；
4. 若多 seed 后 joint 仍不能超过 direct，停止继续堆当前 trace 目标，转向 learned router 或显式神经关系组合器；
5. 不要仅通过增加 epoch、LoRA rank 或学习率继续堆当前方案。

## 结果文件

- corrected curve：`results/runs/wsl3090_v016_joint_20260902_01/corrected_eval/corrected_checkpoint_curve.json`
- corrected selected metrics：`results/runs/wsl3090_v016_joint_20260902_01/corrected_eval/corrected_selected_checkpoint_metrics.json`
- corrected audit：`results/runs/wsl3090_v016_joint_20260902_01/checkpoint_audit_corrected/checkpoint_audit.json`
- candidate ranking（结构化 teacher-forced）：`results/runs/wsl3090_v016_joint_20260902_01/candidate_ranking_audit/`
- deployment-prompt candidate ranking：`results/runs/wsl3090_v016_joint_20260902_01/deployment_candidate_ranking_seq_stage2_epoch5/`
- strict generalization split：`results/runs/wsl3090_v016_joint_20260902_01/constrained_entity_stage2_epoch5/generalization_splits.json`
## 10. Deployment-prompt sequence-level scorer（2026-09-03 补充）

前面的 `scripts/score_answer_candidates.py` 是一个**结构化 teacher-forced 诊断**：它把 gold intermediate trace 作为 answer 前缀，因此不能直接当作 graph-free 部署结果，也不适合单独用来比较 direct 与 joint。为避免把“看到了正确中间节点”误认为模型已经会推理，新增：

```text
scripts/score_deployment_candidates.py
```

该脚本对 direct 和 joint 使用完全相同的 `build_evaluation_prompt`：不提供 gold trace、path 或 evidence；只对 gold + 3 个 train-only distractor 做完整候选序列打分。结果为：

| 方法 | deployment-prompt sequence rank-1 | constrained greedy |
|---|---:|---:|
| direct answer-only | **81/156 = 51.92%** | **87/156 = 55.77%** |
| v0.1.6 joint trace + answer | **77/156 = 49.36%** | **78/156 = 50.00%** |

在同一个 deployment prompt 下，joint 没有超过 direct，反而低 2.56 个百分点。配对统计为：joint 胜 17 条、direct 胜 21 条、两者都对 60 条、两者都错 58 条。严格 composition split 的 sequence rank-1 为：

- direct：seen `31/67 = 46.27%`，novel `50/89 = 56.18%`；
- joint：seen `33/67 = 49.25%`，novel `44/89 = 49.44%`。

这使结论更稳：candidate-set 中确实存在可利用的答案信号，但 v0.1.6 的 joint trace supervision 没有在公平的 graph-free candidate selection 上带来增益。

## 11. Direct answer-only 公平对照（2026-09-03 补充）

使用 v0.1.5 direct answer-only checkpoint，在相同的 gold + 3 个答案去重、同深度、train-only distractor 候选池上评估。这里优先采用上一节定义的 **deployment-prompt** 协议：

| 方法 | deployment-prompt sequence rank-1 | constrained greedy |
|---|---:|---:|
| direct answer-only | **81/156 = 51.92%** | **87/156 = 55.77%** |
| v0.1.6 joint trace + answer | **77/156 = 49.36%** | **78/156 = 50.00%** |

因此两种 graph-free candidate-set 解码都没有显示 joint 优于 direct。旧的结构化 teacher-forced 表格（direct `55/156 = 35.26%`、joint `100/156 = 64.10%`）仍保留在历史结果目录，但它给不同 checkpoint 使用了不一致的结构化目标前缀，只能作为“trace 条件下的模型分数诊断”，不能作为公平 direct-vs-joint 主结果。

严格 composition split 的 constrained 结果：

- direct：seen `30/67 = 44.78%`，novel `57/89 = 64.04%`；
- joint：seen `32/67 = 47.76%`，novel `46/89 = 51.69%`。

这组结果不支持“joint 已经带来组合泛化提升”：joint 在 seen composition 略高，但在 novel composition 低约 12.35 个百分点。由于 test 只有 156 条，且这是候选集诊断，不能据此作强统计结论；但它足以决定下一步必须做严格 split 和解码对照，而不是继续直接加训练轮数。

严格 split 结果文件：

```text
results/runs/wsl3090_v015_direct_20260901_01/constrained_entity_stage2_epoch4/composition_split.json
results/runs/wsl3090_v016_joint_20260902_01/constrained_entity_stage2_epoch5/composition_split.json
results/runs/wsl3090_v015_direct_20260901_01/candidate_ranking_seq_stage2_epoch4/composition_split.json
results/runs/wsl3090_v016_joint_20260902_01/candidate_ranking_seq_stage2_epoch5/composition_split.json
```


## 12. Fair multi-seed follow-up（2026-09-03）

为检查 v0.1.6 joint 与 direct 的差异是否只是单个 seed 的偶然现象，已完成公平协议下 seed 43/44 的训练。两种协议使用相同的 NELL23K exact-hop 数据（716/152/156）、Qwen2.5-0.5B、本地 LoRA 配置、候选池、`max_new_tokens=80` 和每阶段 token 预算；只改变 `training.seed`。每个 run 都按 graph-free validation 选择 checkpoint，再读取 test，不能用 test 反选 epoch。

### 12.1 validation-controlled answer accuracy

| protocol | seed | selected checkpoint | validation | test |
|---|---:|---|---:|---:|
| direct answer-only | 43 | stage2_epoch5 | 49/152 = 32.24% | 52/156 = 33.33% |
| direct answer-only | 44 | stage2_epoch4 | 51/152 = 33.55% | 47/156 = 30.13% |
| graph-free trace + answer | 43 | stage2_epoch7 | 47/152 = 30.92% | 54/156 = 34.62% |
| graph-free trace + answer | 44 | stage2_epoch6 | 52/152 = 34.21% | 52/156 = 33.33% |

两 seed 汇总（样本标准差）：

| protocol | validation mean ± std | test mean ± std |
|---|---:|---:|
| direct answer-only | **32.89% ± 0.93 pp** | **31.73% ± 2.27 pp** |
| graph-free trace + answer | **32.57% ± 2.33 pp** | **33.97% ± 0.91 pp** |

joint 的两 seed test 均值比 direct 高 **2.24 个百分点**，但 validation 均值反而低 **0.32 个百分点**；在只有两个新增 seed、每个 test 仅 156 条的情况下，这不足以证明稳定增益，也没有达到项目预设的 three-seed `GO_LEARNED_PRIORITIZER` 门槛。不同 seed 选择了 epoch 4–7，也再次显示当前训练曲线存在明显 checkpoint 波动。

### 12.2 当前诊断完整性

四个 seed-sweep run 的训练、validation/test prediction、budget audit 和 selected metrics 均已落盘，并生成：

```text
results/fair_seed_sweep_20260903.json
```

但候选集 post-hoc 诊断目前只在 seed 43 的 direct/joint run 下生成；seed 44 目录暂时只有核心训练结果，没有 `candidate_diagnostics_<selected_checkpoint>/`。因此不能把候选集 rank-1 或 constrained greedy 宣称为四 run 的多 seed 结果。当前可以下的最强结论是：

1. 在两个新增 seed 的 validation-controlled 自由 graph-free answer accuracy 上，joint 与 direct 基本持平，均值差异很小；
2. 当前证据不支持“trace supervision 已稳定带来泛化提升”；
3. 在补齐 seed 44 candidate diagnostics 前，不应继续解释候选集解码或进入 learned prioritizer；
4. 下一步优先补齐 seed 44 的 constrained/deployment diagnostics，或直接按项目 gate 追加第三个公平 seed，再决定停止当前 trace 目标还是转向显式组合器。
