# NELL / WN 候选来源对照：单种子、固定预算

## 研究问题与证据边界

父结果为 FB 完整对照：Full 与 Full 逐题频率桶配额匹配随机在直接 EM 和固定列表排序上均接近。研究问题是该观察是否跨数据集成立，以及小关系集合中接近穷举的负例选择是否限制采样策略差异。关联 `C2`、`SC_CTRL_SOURCES_NELL23K`、`SC_CTRL_SOURCES_WN18RR`，对应 `11_paper/paper_outline.md` 和实验矩阵。

NELL 为机制解释的跨图对照（claim-carrying）；WN 为历史候选池条件下的小候选池探索边界（supporting / appendix）。不能把跨图差异单独归因于候选池大小，也不能把 Full 配额匹配当作独立的一般频率方法。历史 Full 获选配置和测试暴露继续披露；WN 本地 B1 32.67% 与原文参考差异仍未解释，本轮不修复 B1，也不以该差异证明采样机制。

正文按研究问题组织重点；预定三行方法和有解释关联的正、零、负差都保留在总表/附录或可追溯报告中。不得按分数方向挑数据集、改指标、删对照、延长训练或改变论文机制结论。可选的全文无关探索可以明确标记为 reference-only，但不能隐藏与声称结论直接冲突的结果。

## 预定设计

| 数据集 | Full 模型 | 新增控制 | 优化步 / 学习率调度步 | QA / eligible QA | 完整测试 | 原测试选项 |
|---|---|---|---|---|---|---|
| NELL23K | 复用原验证获选 step32 | Uniform-9、Full-quota-matched-9 | 32 / 240 | 12,014 / 2,974 | 4,943 | 10 |
| WN18RR | 复用原验证获选 lambda=.25、step125 | Uniform-9、Full-quota-matched-9 | 125 / 125 | 12,334 / 2,709 | 3,134 | 11 |

均为 seed=2026、K=9，各图内对齐其 Full 的 Stage1、QA文本/顺序、资格门、损失权重、优化器和完整训练参数；只改变候选字符串及其来源标签。两个控制使用相同冻结代码。Full 属于历史模型复用，不能声称三个方法均为本次同版本从头重跑。无新验证选模、HPO、多种子、早停择优或根据测试扩展预算。WN 的 K3 机制实验不在这次预算内。

均匀随机从原 B1 分数所定义的候选池无放回抽9个，分数不参与抽样概率。NELL 全部实体对对照 train 核实，并保留原有 train-pair eligibility，不改变问答文本；WN 有下述历史池限制。配额匹配继承 Full 每道题的 `floor(log2(train_edge_count))` 桶配额，再在同一候选池的桶中无放回随机抽样；保留强制槽、实际候选重叠、桶匹配与零 fallback 审计。

WN 原候选审计：2,703题从10个负关系中选9个，6题从9个中全选。该约束预先记录，不以无差异结果事后解释为方法成功；桶匹配的实际可变性由准备审计给出。

WN 深入核验发现：2,654道关系QA的实体对和gold可以对照train核实；另外55道历史score行的head/tail均为null，现有QA也无法精确匹配train实体对。为对齐已训练Full，保留这55题的原QA与原mask（只排除gold）；不作模糊实体匹配、不编造实体对、不改变Full或悄悄删掉它们。每题在pool/frequency审计和diagnostics中标记 `unverified_pair_original_parent_mask`，单独保存ID。预检要求恰为已冻结来源中的55题，其他不支持的实体对一律失败；它们不能记为train-pair核验通过。WN的结论严格限定为同历史候选池的条件对照，不能声称完整过滤审计通过或独立机制成立。该限制在结果已知前登记。

新WN候选行增加 `train_pair_verified`，逐候选增加 `train_known_absent_independently_verified` 和 `train_known_absent_proof_scope`；55题的独立核验标记明确为false。兼容旧loader的 `train_known_absent=true` 仅表示不在提供的排除mask中，不能替代实体对核验。预检前补充这些说明字段，原版本bank、代码和契约保留在运行目录 `prelaunch_provenance_annotation/`，候选字符串不变。

## 评估与保留产物

每图复用 Full 已有完整直接解码结果，新控制各做一次完整直接解码：NELL generation_max_length=32，WN=1000，与各自历史 Full 一致。三模型各做一次相同原候选列表重排，保留原顺序及全部10/11选项；使用相同 mean tagged-answer log-probability、clip20、原顺序破平局、batch2、bfloat16。报告直接 EM 和排序 H@1/MRR/H@3，明确它们是不同答案选择规则；这里的 MRR 不是全图 filtered link prediction MRR。

保留：冻结设计/代码/输入及来源哈希；原 Full 权重身份与直接输出；所有新候选库、逐候选训练池和桶诊断；完整训练日志、训练曲线、资源心跳；NELL 的8/16/24/32步和 WN 的25/50/75/100/125步完整权重/optimizer/scheduler/RNG；新控制原始 response、三模型所有候选分数/token长度/prompt token SHA/逐题rank；所有方法汇总、配对纠错/退化、逐题绘图CSV、LaTeX结果表、完成审计与归档清单。历史 Full 训练成本缺失时不填0，新进程计时不当作所有尝试的总成本。

## 执行顺序与停止条件

单张 RTX3090 24GB；使用 guardenv 和冻结根目录 PYTHONPATH。WSL 约7.5GiB RAM、2GiB swap；运行前记录真实环境和磁盘余量，启动下一图要求至少80GiB可用。只允许一个 GPU 子任务，无 CPU offload。

**物理存储修正（启动前）**：Windows Lxss注册表确认Ubuntu的VHDX位于C盘，C仅余约5.6GiB，Linux虚拟文件系统显示的约762GiB不能替代宿主空间。六个当前/未来training输出目录均重定向到`/mnt/e/GRIP-CU/artifacts/retained_training_20261107/`；WSL逻辑路径不变，原目录保存在各run的`recovery/storage_redirect_20261107/`。K3复制时清单/大小/mtime前后一致，13个文件逐一SHA相等，随后切换目录符号链接，未重启进程或改变优化器/RNG。既有归档代码能遍历作为根目录的training符号链接。

准入核验时E盘约85.44GiB可用，六组训练剩余产物估计74.49GiB，另预留8GiB。估计包括每个计划检查点的权重/optimizer/tokenizer及最终adapter，属于存储估计而非实测最终成本。新守护在每图启动前核验training解析到指定E盘路径、该图两组预计存储需求加8GiB预留、C盘至少1GiB日志余量，并继续保留Linux至少80GiB的附加门限；不满足则阻断，不删旧模型、不减少检查点或擅自压缩预算。预估留有约3GiB额外裕量，外部写入会降低裕量，后续必须以真实门限为准。具体复制SHA、物理路径、原件位置和预算见`storage_redirects/`、`storage_admission.json`和`RUN_RECORD.json`。

1. 当前 FB K3 → 完整重排 → K15 → 完整重排 → 三点汇总/审计；K9复用。
2. NELL Full完整重排 → Uniform训练/直接解码/重排 → 配额匹配训练/直接解码/重排 → 完整审计。
3. WN 执行相同顺序 → 完整审计 → 停止。

新队列总预算恰为4组训练、4次新直接解码、6次完整排序；Full不重训、不重复直接解码。先完成并验证一个数据集，再启动另一个；不以差值方向决定是否继续WN。预算外实验必须另立计划。

每30秒检查状态，等待前序完整完成、tmux退出和GPU空闲；冻结计划/代码/子任务契约。启动凭据先写 intent 后写 launched，模糊状态、失败、会话丢失、输入哈希变化或检查点不完整均阻止自动重试/后续启动。守护仅在当前 WSL 实例存续时有效；实例关闭后需要核查并恢复，不承诺跨实例存活。

队列和实际准备/启动记录见 `RUN_RECORD.json`；准备审计和验证输出在同目录。准备/调度实现完成不代表新实验结果完成。

## 2026-11-08实际更新：保留错误必须披露

K三点已全部完成。NELL Random也已完成32步，但原保存上限0在真实入口变成1，早期8/16/24状态被轮转删除，不能补回；最终32、optimizer/scheduler/RNG和日志曲线在。14:20用原模型恢复完整测试，无重训。其余三组使用显式5/6/6并真实预检；原计划“全部检查点保留”对这一个历史组没有兑现，须在复现说明披露，不能因保存校验恢复就称早期产物已恢复。详见`RETENTION_RECOVERY_20261108.md`，原契约和错误现场均保留。
