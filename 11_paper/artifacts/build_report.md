# Build and review report

日期：2026-11-06

最新PDF：`artifacts/SC-CL_results_revision_20261106.pdf`；`artifacts/SC-CL_ICLR_draft.pdf`同步更新，12页。PDF SHA-256：`561ca2c17b601320ecd426e1815338cf9245f8b346fc00534aed8f25f55432e1`。

XeLaTeX、BibTeX及xdvipdfmx编译通过；无overfull、缺失字符、未定义引用或重复label。此次新增GRIP原文基线参照表，转录其表2--3的十选一准确率和平均输入token数，来源PDF SHA-256为`665713911967c3bb13d7568dbe10360c7a0c5541d2fac014c4c81fc64463f434`；该表独立于本地同协议结果呈现。保留现有CJK字形替代警告。摘要、GRIP基线表、FB控制/分层页、附录macro与历史排序页面已目视检查。

检查、构建输出、文本提取、页面预览及来源映射保存在`artifacts/writing_revision_20261106/`。正文新增三数据集当前完整直接解码、FB五方法、逐题变化与train频率分层；历史NELL排序模型独立放入附录。五张实测表从审计JSON生成，另有一张GRIP原文文献表由原论文PDF转录。构建成功不等于论文科学证据已满足投稿要求。
