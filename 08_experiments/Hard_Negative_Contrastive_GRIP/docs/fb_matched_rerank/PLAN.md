# FB同候选完整重排：执行计划

问题：现有Full与Full配额匹配随机的直接解码EM接近，固定十候选的排序表现是否也接近？该问题对应论文原有排序目标；新重排不能抹去直接解码近零结果及低频负结果。

## 固定协议

- 只评估已有三个seed2026固定权重：Full → Full配额匹配随机 → Uniform Random，串行执行，无新训练、HPO或模型选择。
- 完整10,000题，保留原问题、目标、十个测试选项及其顺序。测试选项与训练负例K分别记录，不使用训练bank替换测试选项。
- 使用既有统一candidate scorer：只对`<answer>关系</answer>`的token取平均log probability，含答案标签、不含提示/填充/EOS；按既有规则裁剪至[-20,20]。保存裁剪前后全部十个分数及答案token数。
- 同一GRIPEvalDataset、chat template、title和no_graph_context设置，直接复用现有解码评估的模型加载路径。token化提示SHA、长度与答案长度跨三模型核对。
- 分数相等按原选项顺序破平局；报告H@1、MRR、H@3。该MRR是固定十选一候选MRR，不是全关系/全实体词表的filtered link prediction MRR。H@10在该设定必为100%，不作为主要收益展示。
- 固定candidate batch=2、bfloat16、seed2026；提示与最长答案合计须≤4096，超限拒绝继续，不静默截断/删除问题。

## 预算与启动门

单张RTX3090 24GiB，guardenv；准备时GPU无计算任务，Linux盘可用约765GiB。最多三次模型推理，各100,000候选续接分数，合计30,000题、300,000候选分数，无新模型权重生成。先用Full前32题记录吞吐/显存，直接计入其全量输出，不重复评估，不按成绩选择是否继续或改变顺序。耗时以实测吞吐更新，不预先把历史NELL速度当FB保证。

前序完整完成、原始分数重算和模型/提示身份审计通过且GPU空闲后，才继续下一模型。遇错误停止、保留现场，不自动调batch、换评分方式或重启失败任务。单例控制器和各模型文件锁防止重复执行。

## 过程文件与后续图表

运行目录：`/home/kieran/grip-cu-runs/fb_matched_rerank_20261106_seed2026`。

- `contract.json`、`inputs/design.json`、`inputs/cohort.jsonl`：固定协议和10,000题问题/目标/选项；每模型adapter配置与权重SHA。模型文件借用既有归档，源目录必须保留。
- `code_snapshot/`：既有评分、tokenizer提示与加载代码，加本轮控制器及严格审计模块；代码合成摘要固定。
- `base_model_identity`记录本地config/tokenizer SHA和基础权重文件大小/mtime；不冒充新计算过全部基础权重哈希。
- `models/<method>/identity.json`、`predictions.jsonl`：每题全部十个原始及裁剪分数、答案token数、排名、MRR、margin、token化提示SHA和原始对照正确性。
- `heartbeat.json`、`pilot.json`、`summary.json`、`completion_audit.json`：进度、实测吞吐/显存和完成重算。每题落盘并fsync；仅允许备份并修复最后一条未完成JSON，禁止忽略中间损坏、重复或身份冲突。
- `logs/`、`controller.log`、`events.jsonl`、`resources.jsonl`、环境记录：全部保留；动态控制日志与状态不提前纳入固定输出哈希。
- `paper/metrics.csv`、`frequency_slice_metrics.csv`、`paired_*.csv`、`paired_ranks_and_direct.csv`、`rank_distribution.csv`、`result.json`、`table.tex`：用于后续排名分布、MRR、配对变化和论文表。已有215/689训练频率分组原样复用，不按新结果切组。
- 完成后导出`11_paper/artifacts/fb_matched_rerank_seed2026/`，附来源索引和原始数据路径。正、零、负结果均报告。

## 解释与停止

Full排序更好时，只支持当前模型/配置下的排序差异；保持原直接EM及低频结果并列，不能以改答案选择规则宣称生成收益或普适结构机制。若排序仍接近，继续收窄独有收益解释。三组完成后自动停止，不自动启动NELL或K训练。

写作映射：SC_FB_MATCHED_RERANK；section_id=experiments；claim_links=C2；paper_role=claim-carrying；main text同模型直接解码与重排并列表。其后按独立实验安排推进，见`11_paper/EXPERIMENT_ROADMAP.md`。

准备补记：首次CPU预检因“归一化后全部选项须互异”的过强假设停止，未加载GPU模型；其现场保留于`..._prepare_attempt1_alias_audit/`。修正为原始字符串互异且gold归一化匹配唯一，保留12题非gold归一化同名选项。成功预检重新核对全部10,000题与30,000条直接解码判定。

2026-11-07接续更新：本冻结三模型控制器仍在完成三组后退出；已另部署独立守护等待其完整审计与GPU释放后执行K3、K15。其原代码和原评分协议未改动，新增训练预算及失败门见`../fb_rerank_to_k_chain/README.md`。
