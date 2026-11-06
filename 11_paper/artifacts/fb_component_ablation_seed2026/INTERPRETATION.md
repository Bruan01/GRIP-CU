# 论文叙事与证据边界

## 可用于结果段的表述

在FB15K237-2的固定配置单种子对照中，联合结构画像与冻结模型违例反馈的SC-CL获得89.90%的直接解码EM，高于仅使用模型违例反馈的87.09%与仅使用结构画像的84.89%，分别增加2.81和5.01个百分点。均匀随机对照为86.48%；结构画像单独使用低于随机1.59个百分点。因此，本组观测支持结构画像在保留模型违例反馈时的条件性增量，而不支持结构画像单独使用即有收益的主张。

## 复现与解释边界

同10,000题、提示与候选顺序、目标和EM normalization；固定Stage1、12,136训练QA、3,109候选资格、train-known合法池、K=9、InfoNCE、LoRA、优化器与32步/120步调度。唯一预定干预是采样utility的结构/B1权重；Random使用同池均匀采样。组件各一次完整测试，无新全量验证、重复种子或HPO。配置此前依据Full验证表现选定，并非各方法等预算最优比较；共享464条train-pair修复与历史测试暴露。

不要写成跨种子稳定、统计显著、普适协同、两个分量缺一不可或已证明具体语义因果机制。无需为弥补Profile负结果追加搜索；该结果应与主表同时保留。

## 写作映射与下一步

- selected_outline_ref: `11_paper/paper_outline.md`
- research_question: 联合采样相对随机的收益能否由单个信号解释？
- experimental_designs: `SC_CTRL_VIOLATION`、`SC_CTRL_PROFILE`
- section_id: experiments；item_id沿用实验ID；paper_role: claim-carrying；claim_links: C2/C4/C5
- target_display: 主文组件消融表，使用同目录`table.tex`/`metrics.csv`；逐题证据为`paired_four_methods.csv`。
- takeaway: 联合使用取得当前配置最高观测性能；Profile单独使用为负结果。
- comparability: 单seed固定配置的组件替换可比；不等同各方法独立调参后的最优性能比较。
- todo_items: 将四行表和限定文字纳入后续正文修订；频率解释及跨seed稳定性仍开放。
- next_route: write；本轮stop，不再启动训练。

过程证据见`completion_audit.json`及两个单信号组的`evidence_index.json`。完整模型、检查点/优化器/RNG、原始预测、代码、bank及资源日志保留Linux运行目录；其借用的Random输入源目录必须同时保留。
