# FB 独立训练边频次采样对照

日期：2026-11-10。父结果：FB Full 直接EM89.90%、配额匹配89.83%、均匀随机86.48%；同列表重排H@1分别94.85/94.72/93.68%。现有Profile-only/Violation-only/配额匹配均已完成，不重复训练。

## 问题和干预

检验一个不依赖Full候选库、结构画像或B1分数的简单频率采样器，在既有Full获选配置下的表现。采样只读取train.txt关系边数f(r)，对每道训练QA的合法关系池依次无放回抽九次；每一步剩余关系r的概率为f(r)/sum_remaining f。指数固定1，不混入均匀下限，不做逆频率、平滑指数或验证/测试后调参。

这与旧频率匹配组不同：旧组继承Full每道题的频率桶配额，新组没有Full配额或gold频率相似度。选完新bank后才读取父bank和pool审计作一致性检查。评分文件不参与新bank构造。

## 固定条件

- FB15K237-2、seed2026；共享原Stage1、12,136条QA、同3,109条候选损失资格、233个训练关系。
- 从原训练QA和train-only实体对索引独立重建合法池，排除gold及该实体对全部train-known关系；与原pool逐题SHA比对。
- 相同InfoNCE、生成目标、LoRA、优化器、学习率、微批、累积步数等；仅候选身份/来源变化。
- 实际32优化步，保持120步调度；固定第32步模型，无新全量验证、HPO、额外种子或测试后追加搜索。
- 保留464条QA修复及既往测试暴露的限制；这是Full获选配置的条件对照，非等预算独立调优。

## 评价及停止条件

一组新训练 → 一次完整10,000题直接解码 → 同原十选项完整重排 → 审计与导出 → 停止。主指标为H@1，MRR/H@3和直接EM同时报告，不按新分数挑协议。

复用Full/配额匹配/Uniform三组完整排序，不重跑参考模型。保留原非gold归一化同名选项；不删题或合并选项。复用原tagged-answer token平均对数概率、clip20、batch2、bfloat16、原选项顺序破平局。CPU逐题核验prompt、target、选项顺序、token长度、rank与原始分数；同训练模型权重身份须贯通两种评价。

所有方法和带符号差值都保留。不预设“接近”的等效阈值；单次接近不证明统计等效。新组若接近Full，收窄复杂选择必要性的局部主张；若落后，只表明这一固定频率规则不足，不证明结构因果性、双信号协同或所有频率策略均不足。没有新结果前不重设计方法或宣布论文失败。

## 执行环境与保留

- 正式Python：`/home/kieran/miniconda3/envs/guardenv/bin/python`；snapshot运行环境显式设置PYTHONPATH。
- RTX3090 24GiB，串行一组；启动时无其他GPU计算作业。
- C盘约5.1GB、E盘约19GB可用；Linux表面空闲量不能替代物理卷空间。
- 运行根：`/home/kieran/grip-cu-runs/fb_frequency_only_control_20261110_seed2026`。
- training软链接至`/mnt/e/GRIP-CU/artifacts/fb_frequency_only_control_20261110_seed2026/training`。准备要求E盘至少16GiB，包含四个完整检查点、最终adapter及约3GiB余量；C盘至少1GiB。保留全部8/16/24/32检查点和optimizer/scheduler/RNG，不清理旧实验。
- 按此前FB32步对照预留约3小时训练；两种完整评测另需时间。此为历史预算锚点，实际时间以运行日志为准。
- 保留冻结配置/代码/输入/bank及其SHA、资源日志、原始响应、逐候选分数、逐题配对与绘图CSV。借用父输入/模型不得删除。
- 失败保留现场并停止，不自动重试、改预算或换采样器。

## 论文映射与状态

`exp_id/item_id=SC_CTRL_FREQONLY_FB`；`section_id=experiments`；`paper_role=claim-carrying`；`claim_links=[C2]`；主文来源对照与附录复现。结果导出到`11_paper/artifacts/fb_frequency_only_control_seed2026/`。

目前已完成独立bank和CPU核验准备；准备/启动不代表实验完成。实时状态看运行根`status.json`，最终结果看导出目录`result.json`；没有该文件就不能报告新性能。

2026-11-10 14:18（北京时间）启动：tmux会话`fb_freqonly_20261110`；真实CLI核对12,136/3,109条QA通过，训练进程已运行。初次相关回归55项通过，随后新增报告/模型身份测试后独立频次组16项通过。训练模型性能尚未产生。新bank前24高频关系占槽53.10%，Full为11.16%；仅为训练bank组成诊断。

下一路由：完成后基于所有差异选择write或idea。若方法贡献需要调整，先从证据提出候选假设，再另立实验，不在本队列自动开展“救分”搜索。
