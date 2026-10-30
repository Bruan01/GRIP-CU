# 结构反事实对比学习：精简实验 TODO

更新时间：2026-10-28（按用户要求删减扩展实验）  
对应论文：`11_paper/main/main.tex`  
方法：SC-CL，训练图关系画像 + 冻结模型违例度 + train-known 过滤 + 固定候选库。

当前只保留五组：**主对照、双信号消融、候选机制与审计、一个第二数据集、成本与输出行为汇总。** 原 853 行清单及其历史工程任务已整体备份，本文为当前执行清单。已完成产物保留，未启动或停止任何训练作业。

**论文需要回答的四个问题**

| 问题 | 保留的证据 | 论文位置 |
|---|---|---|
| SC-CL 是否比普通候选训练及频率匹配候选更有效？ | NELL 全量同契约主对照、三训练种子 | 主表 T1 |
| 结构与模型混淆分别贡献什么？ | Profile-only、Violation-only，与主对照复用比较 | 消融表 T2 |
| 双信号实际选了什么，候选来源与真值状态怎样记录？ | 候选分布、过滤审计、两个描述性切片 | 机制图 F1、审计表 T3 |
| 另一关系图上是否呈现相同趋势，需要什么资源？ | 一个第二数据集；现有输出与分阶段成本 | 扩展表 T4、成本/输出辅表 |

## 0. 执行前核验：合并为一个检查项

- [ ] PAPER-EXP-00：固定真实运行版本及比较契约，完成下列核验后复用旧模型和 bank；不为此另建实验组或重复跑整套环境测试。

| 核验内容 | 当前需要解决的具体问题 |
|---|---|
| 公式与代码 | 当前 `counterfactuals.py` 五维均 log1p、距离分母为 `1+abs(a)+abs(b)`，度按不同实体求均值；论文现有表述有差异。按实际运行版本统一，保留旧 bank 与结果身份。 |
| 初始化与 teacher | NELL run_config 的 s1_adapter / b1_adapter 指向同一路径，需核对来源与权重身份；训练初始化、离线评分模型和比较 B1 分别记录。 |
| 温度语义 | 固定 InfoNCE、采样与 violation 的实际温度/读取分支；本轮不扫描温度。 |
| 评价与对齐 | 固定十选一候选重排；逐题核对 ID、target、候选列表/顺序及 prompt hash。现有 evaluator 使用题目给定选项，不属于 label-blind 检索。 |
| 指标 | 从 candidate_scores 核对 MRR/H@1/H@3、tie 规则；需要 margin 时计算 gold-minus-hardest。当前 score_margin 是 top1-minus-top2，不能混用。 |

统一条件记录在每次运行的 config/manifest 中。正式 Python：`/home/kieran/miniconda3/envs/guardenv/bin/python`；PYTHONPATH 包含 `13_base_method/grip-exp`、实验模块 `src` 和 `scripts`。

## 1. PAPER-EXP-01：NELL 主对照与三种子

**职责：检验候选选样本身的价值。**

| 模型 | 训练安排 | 比较作用 |
|---|---|---|
| 现有 B1 | 复用 adapter，补完整 4,943 题重排；不新增 G0 训练 | 原 GRIP 参照，注明历史预算 |
| Random-9 | 新训练，合法候选池中均匀无放回采样 | 候选级 InfoNCE 的同预算随机控制 |
| Frequency-matched-Random-9 | 新训练，匹配 SC-CL 候选的训练关系频率桶 | 控制关系频率因素 |
| SC-CL-9 | seed=2026 优先复用，核验通过后补其他种子 | 当前方法 |

- [ ] PAPER-EXP-01-A：补现有 B1 全量重排，保留其与 SC-CL 的历史预算差异；不以该比较单独归因候选构造收益。
- [ ] PAPER-EXP-01-B：补 Random-9 与频率匹配随机的固定 bank 和可执行训练入口。匹配桶仅从 train 定义，逐题匹配候选频率配额，记录配额不足的回退规则及实际匹配误差。
- [ ] PAPER-EXP-01-C：三个候选训练臂共享 Stage 1、task、LoRA、优化器、微批顺序、训练步、K=9、lambda=1、候选温度=1 和 train-known 过滤；各方法用自身 bank，同方法跨训练种子固定 bank。
- [ ] PAPER-EXP-01-D：主三臂固定训练种子 2026、2027、2028，完整报告逐 seed 结果及 mean±sample-std；本轮不扩展五种子。
- [ ] PAPER-EXP-01-E：同题输出 H@1、MRR、H@3 与 paired delta。MRR 为主指标，直接解码结果随最终 checkpoint 一并保存，不作为额外调参任务。

NELL 参考配置：LoRA rank/alpha=4/8，Stage 2 为 10 epochs、240 steps，batch=1、梯度累积=512。具体以核验后的主运行配置为准。

**统计**：SC-CL–Random 和 SC-CL–频率匹配随机为预定主比较；报告配对区间，问题级 bootstrap 不替代训练种子间的波动。多种子区间同时保留 seed 与问题的配对关系，单种子子集不写成整体稳定性结论。模型选择和参数固定使用验证集，不按测试差异挑选 seed 或设置。

**输出**：T1 主表及逐题对齐文件。三个候选臂的比较承担方法收益论证；现有 B1 作独立参照。新增 G0-matched、独立 InfoNCE 开关训练均已取消，因此不再单独主张对候选损失自身作严格因果归因。

## 2. PAPER-EXP-02：双信号消融

**职责：说明结构画像与冻结模型混淆各自的作用。**

- [ ] PAPER-EXP-02-A：在 NELL seed=2026 补 Profile-only（alpha=1,beta=0）与 Violation-only（alpha=0,beta=1）；保留 rho=1/3、K=9 和同一训练契约。
- [ ] PAPER-EXP-02-B：与 EXP-01 已有 Random/SC-CL 共用评价，报告 H@1、MRR 和相对差异；保留逐题四类正确性变化。

**输出**：T2 四行消融表。当前单种子消融用于机制描述，不写“跨种子稳定互补”或“超加性”。不追加交互检验、熵匹配单信号、画像置换、Frequency-only+Violation 等训练臂。

## 3. PAPER-EXP-03：合并候选分布、开放世界审计与少量切片

**职责：用已有产物说明候选设计的内容与作用场景。优先 CPU 聚合，不新训模型。**

- [ ] PAPER-EXP-03-A：对 Random、频率匹配随机、Profile、Violation、SC-CL 的已选候选，按问题等权汇总结构相似度、B1 violation、score gap、关系频率及关系覆盖；画一张结构相似度×violation 分布图。频率匹配误差合并在此报告。
- [ ] PAPER-EXP-03-B：独立核对训练来源、question ID、候选互异与已知关系过滤。区分 train-known 冲突、同实体对的 valid/test 正例重叠和开放世界未判定，分别报告候选数/问题数及分母。
- [ ] PAPER-EXP-03-C：只保留两个描述性切片：训练图关系频率、固定 B1 在相同候选上的混淆程度。预先固定分桶，报告每层 n 与 Random/频率随机/SC-CL 的 MRR、H@1 差异；低样本层只作描述。
- [ ] PAPER-EXP-03-D：按固定规则抽取 2–3 个可复算候选实例，展示结构分数、违例度和过滤记录；存在已知冲突类别才展示该类实例。

**输出**：F1 一张机制组合图、T3 一张来源/真值审计表及简短实例。原 PAPER-EXP-05/06 的必要部分并入本组，不再逐项开实验。

持出事实只用于已冻结候选的事后审计，不回流采样、过滤、参数或模型选择。未判定不当作假事实；CoDEx 实体替换 hard negatives 不给关系替换候选提供真值。取消关闭过滤的模型训练、隐藏正例压力实验、全部 degree/arity/稀疏度切片及结构特征变体。

## 4. 第二数据集：只补一个，归入主效果表

**职责：在另一关系空间复核候选方法，复用已有任务与模型资产。**

- [ ] PAPER-SECOND-01：先核对 WN18RR 已有 SC-CL 获选模型和本地 B1 的初始化、task、解码长度及预算；核对 SC-CL bank、teacher 和验证集选参记录。
- [ ] PAPER-SECOND-02：若来源与比较契约可统一，优先用 WN18RR 补 SC-CL 与 Random 的同预算重排、直接解码，目标三个训练 seed；已有兼容 SC-CL seed=2026 可复用。
- [ ] PAPER-SECOND-03：若 WN 基线/模型来源无法统一，则改用已完成 task 的 CoDEx-M，按同样两臂设计验证。选择依据是资产和协议可用性，先固定数据集再看新增测试差异；本轮只执行一个第二数据集。

WN 已有 2,709 行候选库，lambda=0.25 获选模型在 3,134 题上的直接解码 EM 为 86.89%；本地 B1 复现差异仍需核验。WN 配置在数据集内匹配，不强行套用 NELL 的 LoRA/训练轮数。已用验证集选择过的设置保留选择记录，新增对照共享固定设置；若重新搜索参数，两臂获得相同验证搜索预算。

CoDEx-M 的 `tasks/generation_plan.json` 已为 complete，有 236,027 条 context_samples 和 12,300 条 qa_samples，先盘点现有模型产物。FB 历史 i0 保留独立参考身份，暂不为本轮恢复 FB 全套训练；canonical FB、YAGO3-10、CoDEx-S/L 和 CLEGR 均移出执行计划。

**输出**：T4 一个第二数据集的主对照，注明实际 seed 数和模型身份；第二图缺频率匹配随机时，不将频率控制结论外推至该图。

## 5. PAPER-EXP-09：成本与直接解码汇总，随主实验产出

**职责：交代实际运行条件与模型输出行为，复用 EXP-01/02 的模型和预测。**

- [ ] PAPER-EXP-09-A：汇总离线 B1 评分、bank 构造、Stage 2、直接解码、重排的实测耗时、token/forward、已记录峰值显存，明确共享预处理的摊销方式。不为成本分析另跑训练。
- [ ] PAPER-EXP-09-B：最终 checkpoint 在同题上报告直接解码 EM 和重排 H@1/MRR；列表外与训练词表外输出分开统计。保留不占优结果，不把 SC-CL 自身两种答案选择规则的分数差写成训练收益。

NELL 已有评分 19,027 秒、训练约 80,870 秒、重排 3,567 秒的记录。现有 SC-CL/B1 直接解码 EM 为 87.03%/87.46%，SC-CL 重排 H@1 为 91.52%，三者保持原有指标身份。

**输出**：成本表与直接解码辅表。原 PAPER-EXP-10 合并于此；取消中间 checkpoint 验证、收敛曲线、预算前沿、完整显存曲线和额外 NLL 评估。

## 6. 执行量与顺序

| 批次 | 内容 | NELL 新训练数量 |
|---|---|---|
| CPU 与评估准备 | EXP-00；复用产物做 EXP-03；补 B1 full rerank 的运行配置 | 0 |
| 第一批 | Random、频率匹配随机、Profile-only、Violation-only 各 seed=2026 | 4 |
| 第二批 | Random、频率匹配随机、SC-CL 各补 2027/2028 | 6 |
| 一个第二数据集 | Random/SC-CL 配对，复用兼容模型 | 单独计量 |

NELL 共 11 个 arm×seed：主三臂×3 seed 加两个单信号×1 seed；复用原 SC-CL seed=2026 后新增 10 次训练。以历史 80,870 秒/次作为保守等效锚点，第一批约 89.9 GPU 小时，两批合计约 224.6 小时；不含评分、评估、排队和第二数据集，实际耗时另行记录。旧模型契约不兼容时再增加一次 SC-CL 训练，不隐去这项成本。

按“版本/公式核验 → B1 full 评估 → Random/频率匹配入口 → 第一批训练 → 主三臂三种子 → 一个第二数据集 → CPU 图表汇总”推进。固定三种子计划，不以首轮显著性决定是否补种子。

RTX 3090 24 GiB 按串行作业安排，运行前查看当前占用和磁盘；不重训 teacher、不复制基座权重、不自动清理已有 checkpoint。未请求在本轮启动 GPU 作业。

## 7. 已删减项目与合并关系

| 原项目 | 本轮处理 |
|---|---|
| 新训练 G0-matched / InfoNCE 开关 | 删除；复用现有 B1 作参照，候选方法主比较保持同预算 |
| PAPER-EXP-04 画像打乱、频率单特征、熵匹配训练 | 删除；频率匹配随机保留在主比较 |
| PAPER-EXP-05 独立组 | 来源/持出审计并入 EXP-03；关闭过滤训练和隐藏正例实验删除 |
| PAPER-EXP-06 全套切片 | 仅保留频率与 B1 混淆两类 CPU 描述分析，并入 EXP-03 |
| PAPER-EXP-07 顺序扰动、新干扰项、全词表、top-L 检索 | 整组删除；固定现有十选一条件排序协议 |
| PAPER-EXP-08 K/rho/tau/lambda 敏感性 | 整组删除；已有 K3/9/15 bank 保留，不因此启动额外训练 |
| PAPER-EXP-09 收敛曲线与预算前沿 | 删除；只保留运行中自然产生的成本记录 |
| PAPER-EXP-10 独立实验组 | 两种答案选择规则的最终结果并入主实验/EXP-09 |
| 五种子、多数据集铺开、外部 KGE/图对比基线、ErrorMemory | 从当前执行计划删除；现有历史产物仍归档 |
| 风险感知 v2、软标签/风险权重分支 | 移出当前论文补实验任务，不自动启动算法重设计 |
| 重复的旧工程清单与历史路线门 | 从当前文件移除，完整内容保存在精简前备份 |

## 8. 最小实现与完成标准

- [ ] DELIVER-01：补齐语义明确的 Random/频率匹配随机入口，复用 `build_structure_counterfactual_bank.py`、`train_listed_contrastive.py` 与现有 manifest/wiring；不另建重复训练框架。
- [ ] DELIVER-02：增强 `compare_candidate_rerank_runs.py` 的候选/prompt 对齐和同规则指标聚合；不使用跨论文不同答案规则的增益字段。
- [ ] DELIVER-03：每次运行保留 config、模型/teacher/bank/task/eval 哈希、seed、逐题预测/分数、来源审计和自然产生的资源记录。实验脚本改动只检查受影响路径，不另列完整项目测试/环境重建任务。
- [ ] DELIVER-04：实际完成后写回 `11_paper/paper_experiment_matrix.md/json`、`results_status.md` 和 `artifacts/evidence_ledger.md`。旧矩阵中的扩展项统一标为本轮取消/归档，不因仍在旧矩阵中出现而重新启动。

本轮不修改训练代码和结果状态，不把删减实验解释为已有结果通过验证。论文按“主效果—双信号—候选机制—一个第二数据集—资源条件”组织，每张表图只承担对应职责。

精简前完整清单：`11_paper/artifacts/experiment_todo_before_prune_20261028.MD`；更早版本：`11_paper/artifacts/experiment_todo_before_20261028.MD`。
