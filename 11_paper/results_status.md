# Results status：必须保留在论文工作区

更新日期：2026-11-03

2026-10-30 执行范围补记：本轮按用户要求仅做 NELL/FB/CoDEx-M 的单 seed=2026
Stage-2 超参数搜索；复用并固定各自的 Stage-1 权重，多 seed 暂缓。
2026-10-30 时新队列尚未产生完整测试结果；以下旧行仍作为历史实验记录保留。

2026-11-02 新增：NELL 单 seed=2026 的当前 Stage2 搜索已完成一次完整测试。
验证选定 trial1/step32 后停止该数据集搜索，再评估固定模型；4,943题正确4,340题，
自由生成 EM=87.800931%。原论文v2表3的 GRIP-qwen-7b 参考为87.74%，
本地观测数值高0.060931个百分点。匹配同题、同候选提示的当前B1为4,181/4,943，
EM=84.584261%；净增加159题。它不是下方历史87.457%的另一份B1。
证据与只读模型归档在 `artifacts/nell23k_single_seed_test_20261102.json`。
归档权重位于Linux保留目录，通过硬链接保留，不额外复制2.2GB权重块。
这是单次本地协议观测；此前NELL测试已有探索性暴露、任务生成使用过验证few-shot，
且方法包含Stage2调参（包括与控制不同的Adam epsilon），不能写成纯损失项的隔离收益、
多种子结论或已完成全部论文协议对齐。该时点CoDEx与FB正式方法测试尚未完成。

2026-11-03 新增核验：CoDEx-M / seed2026 已完成10,311题完整测试，
trial1/step32自由生成正确9,958题，EM=96.576472%；原论文表3参考94.42%，
本地观测数值高2.156472个百分点。同问题、候选顺序和答案的匹配B1正确9,727题，
EM=94.336146%，方法净增231题（+2.240326个百分点）；paired rescue=301、
regression=70。20,622条预测正确标记已重算、问题ID与提示对齐已核验。
验证选定后冻结模型再测试，完整验证EM=96.469447%，当前搜索保持关闭。
证据见 `artifacts/codexm_single_seed_test_20261103.json`；获选权重以只读硬链接
保留在Linux论文归档目录，不增加权重数据块。当前Stage2采用5-epoch调度并选中第32步，
与论文CoDEx的10-epoch设置不同；任务生成与提示协议仍需审计，不能声称严格论文复现、
多种子结论或纯损失项隔离收益。当前FB train-only bank及wiring已完成；Stage2第8步后
曾遇CUDA异常，随后恢复又被WSL路径别名误判阻断。路径判断已修复，正从完整第8步
检查点恢复；FB尚无本轮完整方法测试结果。NELL新续训继续等待该队列结束。

## 当前结论

当前论文是“单种子实测 + 未完成实验协议”的中期稿，不再把所有数字视为 preset。

### 已完成且可用于正文的证据

1. **NELL23K / SC-CL-9 / seed=2026**
   - 训练 run：`08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/nell23k_structure_counterfactual_3090_20261021_06_train`
   - Stage 2：10 epochs、240 steps、12,148,720 input tokens；训练完成，adapter、预测和 summary 齐全。
   - train-only bank：3,253 行，每行 K=9，`rows_failed=0`；wiring 报告 `train_only=true`、`same_questions_as_task=true`、manifest hash 已记录。
   - 自由生成：4,943 条测试 QA，EM=87.032%（`listed/predictions_correct.jsonl`）。
   - 官方候选 teacher-forced 重排：4,943 条，EM/H@1=91.523%、MRR=95.270%、H@3=99.069%、H@10=100%。相对同一模型自由生成的 EM 差异为 +4.491 个百分点；这不是自由生成提升。
   - B1 与 SC-CL 的同候选列表配对重排目前仅有前 200 条：B1 EM/MRR=88.50%/93.36%，SC-CL EM/MRR=91.00%/94.96%；配对 rescue=9、regression=4、both-correct=173、both-wrong=14。

2. **NELL23K / B1**
   - 原始 B1 自由生成覆盖 4,943 条，EM=87.457%，可作为生成锚点。
   - B1 的同候选列表闭集 full 结果尚未完成，因此不能把 SC-CL full MRR 与 B1 full MRR 直接排名。

3. **FB15K237-2 / 历史候选 InfoNCE-i0**
   - 本地 processed test 与历史产物 hash 相同：`c77dde51aafc7b1ce3a5011ade914343c2689f8b38b12774ea4840ec1c9068ed`。
   - B1 candidate score：10,000 条，EM/H@1=79.91%、MRR=88.007%、H@3=95.80%、H@10=100%。自由生成 EM=73.23%。
   - i0 candidate score：10,000 条，EM/H@1=91.62%、MRR=95.412%、H@3=99.20%、H@10=100%。自由生成 EM=88.69%。
   - i0 来源为历史 `fb15k237_i0_paired_candidates.json`，LoRA rank=24、Stage 2 epochs=5、lambda=0.1 等均与当前 SC-CL 契约不同；该结果只作为历史候选 InfoNCE 辅助证据，不是 SC-CL 的 FB 主结果。

4. **WN18RR / SC-CL 单种子探索性直接解码**
   - train-only bank：2,709 条关系问答，每条 K=9，`rows_failed=0`；wiring 记录 `train_only=true` 且与任务问题对齐。
   - 在完整 valid 的 3,034 条问题上选定 `lambda_candidate=0.25`，随后完成一次 3,134 条全测试的直接解码评估，EM=86.886%。训练达到 5 epochs / 125 steps。
   - 同一测试文件上的本地 B1 直接解码 EM=32.674%；与原论文 WN18RR 的五次运行均值 64.30% 有较大差距，复现口径尚需解释。当前没有该获选模型的完整候选重排、匹配随机对照与多种子，故只作附录探索性记录，不进入主效果或跨图收益结论。

## 尚未完成、不能写成实测结论

- FB15K237-2 当前 B1/Random-9/SC-CL-9 matched-budget 主比较（train-only bank与wiring已完成）；
- NELL/FB 的 Random-9、frequency-matched Random-9、Profile-only、Violation-only、ErrorMemory-9；
- 五个训练种子 2026--2030 的均值、标准差、bootstrap CI 和显著性；
- NELL 的 K/rho/tau 敏感性、假阴性 held-out overlap、切片和完整资源剖析；
- WN18RR 获选模型的完整候选重排、匹配控制、多种子与本地 B1 复现解释；CoDEx-M 的Random-9等消融、多种子和论文协议对齐；canonical FB15K-237、YAGO3-10 和 CoDEx-S/L 暂缓。

### CoDEx-M 数据资产及状态沿革

- 官方 CoDEx-M commit `3132e426c2a6b643b70bad679905a3a6270be440` 已固定，17,050 entities、51 relations、185,584/10,310/10,311 train/valid/test 正例完整转换；valid/test 关系 QA 均为全量。
- 10 选 1 候选由 train 关系标签确定性构造，过滤相同实体对的 train-known 关系；CoDEx 官方 tail-corruption 硬负例单独保留，不作为关系候选真值。
- 当前冻结候选审计：train-known 冲突为 0；val 92,790 个负候选中有 8 个与持出正例重叠、92,782 个开放世界未判定；test 92,799 个负候选中有 6 个持出正例重叠、92,793 个未判定。未判定不计为假例。
- 上述为早期只完成数据转换时的记录；后续已完成任务构造、Stage1/B1恢复、SC-CL候选库与Stage2搜索，2026-11-03核验完整方法测试，见本页新增记录；不再标记为 `pending_inference`。

## CLEGR：排除在当前论文范围之外

- 本文研究对象是固定实体对上的关系候选排序；现有 SC-CL 的关系画像、B1 违例分数、关系词表候选库和 MRR/Hits 均针对这一任务定义。
- CLEGR-Facts/Reasoning 是图级问答，包含路径、拓扑、聚合等问题和异质答案空间。把关系候选 SC-CL 直接迁移过去会改变反事实定义、候选空间和评估目标，因此 CLEGR 不进入当前论文的实验矩阵、主表或跨数据集结论。
- 用户报告的异机 CLEGR 运行相对原论文高 `0.415`，作为内部旁支信息保留；当前工作区没有该运行的原始产物，且此结果不用于当前论文的正面或负面结论。现有本地低预算评估（6,460 条、EM=27.15%）也不属于 SC-CL 证据。
- 若未来另立 CLEGR 研究，应单独定义并实现图编辑反事实与确定性答案验算，遵循 GRIP 的 CLEGR 协议；本地 `paper_protocol.py` 将 `clegr_reasoning` 的主指标设为 `llm_as_judge`，judge 为 Qwen2.5-32B-Instruct，测试数为 400。该协议不纳入本文。

## 正式替换规则

只有当每个方法、数据集和 seed 同时具有原始 prediction、闭集 score、自由生成输出、
配置、数据/adapter hash、日志和 question-ID 对齐报告时，才可从“待运行”改为“observed”。
历史 i0 不得被重命名为 SC-CL；候选重排 EM/MRR 不得写成自由生成性能提升。
