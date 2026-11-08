# 2026-11-08 论文更新与精简

## 编辑前策略

采用 write / paper-outline skill；远程 quest、memory、paper-contract 工具当前未提供，以本地大纲、矩阵、证据账本和已完成结果为准。保留二区期刊目标；ICLR 仅为排版模板。本轮不新增训练、不改变模型选择、不增加未经核验的文献主张。

| 问题类别 | 问题 | 唯一处理动作 |
|---|---|---|
| manuscript-mapping | 已完成FB三模型重排未进入正文 | add result to main text：同题、同十选项的H@1/MRR/H@3及配对差异 |
| manuscript-mapping | K3/9/15已完成但正文仅有K9 | add result to main text：全部K点，明确训练负例K与测试选项数不同 |
| narrative | 实验与解释混排，来源效用和机制归因易混淆 | repair paper contract：实验负责配置/来源/排序，分析负责频率解释/分层/K边界 |
| unsupported writing | 泛化、统计等效、最优K、跨K优于频率控制缺少证据 | downgrade or remove claim：保留单seed获选配置限定及全部正/近零/负结果 |
| evidence gap | NELL/WN来源控制、多seed、真值风险审计未形成完整证据 | downgrade claim：不把运行中状态当性能，不追加实验 |
| metadata drift | 大纲仍含K计划，章节计划未映射最新结果 | repair paper contract：同步C7、显示角色及矩阵定位 |
| packaging | 教学稿、修订前快照和多个旧PDF并存 | repair paper contract：只保留一套正式正文与当前PDF；历史版本由Git恢复；图表、实验证据、引用库及必要模板保留 |

核心叙事：候选身份与其诱导的统计组成共同变化；相对均匀随机的收益尚不能证明超出Full频率配额的独有选择收益。重排对照检验这一边界是否仅来自解码方式，K分析检验增加训练候选是否单调改善排名。

## 清理边界

删除teaching中的独立教学正文、修订前快照中的重复正文和旧稿PDF；保留其中图表。删除LaTeX模板示例正文/PDF及可再生的旧编译副产物。实验JSON、CSV、审计、来源索引、结果表、图形及生成脚本保留。唯一正式入口为main/main.tex，唯一当前全文PDF为main/main.pdf；旧记录中的历史路径仅作历史来源引用。

## 验证

完成：七张实测表重建；两份10,000题逐题CSV独立重算六组指标和所有配对计数通过；21个LaTeX依赖、36个label、16个引用key通过；57个受版本控制图形文件全部保留。XeLaTeX/BibTeX编译14页通过，无未解析引用、文献、缺字或越界；已渲染检查首页和分析/讨论页。仍有CJK字体样式替代及少量Underfull警告。精确记录见manuscript_validation_20261108.json和build_report.md。删除34个重复/过期受版本控制文件及20个忽略的旧编译产物，清单见cleanup_20261108.json。未启动训练，未修改现有实验结果。
