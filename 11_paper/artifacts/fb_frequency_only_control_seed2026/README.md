# FB独立训练频次采样对照

2026-11-10已完成实现、CPU准备并启动，**尚未完成正式训练/评测，不存在新性能结果**。这份说明是启动时点记录；若目录出现`result.json`且状态为completed，以该完整结果为准。

- [固定计划](../../../08_experiments/Hard_Negative_Contrastive_GRIP/docs/fb_frequency_only_control/PLAN.md)
- [启动身份](RUNNING.json)：运行根、tmux会话、契约与控制器SHA；结束后控制器会将其更新为completed。
- [CPU预检记录](prelaunch_validation.json)：两批测试、真实CLI解析及训练元数据一致性、存储与bank身份。
- [构库审计](preparation_audit.json)及[训练关系频次](relation_frequencies.csv)。

新组按train.txt边频次成比例逐次无放回抽9个合法负关系，指数固定1。抽样API不接受Full候选、B1分数或结构画像；不继承Full逐题频率桶。原有五组消融仍保留，其中89.83%的频率配额匹配组依赖Full配额，与本组问题不同。

范围：FB、seed2026、共享Stage1和12,136条QA、3,109条资格；32优化步/120步调度；一次完整直接解码与同原十选项重排。没有新HPO或重复种子，不根据测试结果调整规则。所有正/零/负差保留。

启动时运行根：`/home/kieran/grip-cu-runs/fb_frequency_only_control_20261110_seed2026`；实际训练目录在E盘，见预检记录。`status.json`、`controller.log`、`logs/train_01.log`、后续`evaluation/test`和`rerank/models/frequency_only`保留实时进度和原始产物。tmux会话`fb_freqonly_20261110`仅在当前WSL实例存续时有效；失败保留现场并停止，不自动重试或更改预算。

初次相关回归55项通过；增加完整导出与权重身份拒绝测试后，新组16项全部通过。训练/评测结论须等待完整run，而非这些CPU测试。

论文映射：`SC_CTRL_FREQONLY_FB`，主文候选来源对照，`C2`。若接近Full，收窄复杂选择必要性；若落后，只说明这个独立频率规则不足，不证明结构因果性，也不证明全部频率方法无效。配额组和本组都无法代替多种子。
