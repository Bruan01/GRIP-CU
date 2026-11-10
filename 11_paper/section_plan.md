# Section plan：SC-CL

更新日期：2026-11-10。当前稿为已有证据的中期整合，不代表投稿就绪。

| 章节 | 论证职责 | 当前证据与展示 |
|---|---|---|
| Abstract / Introduction | 提出可追溯候选学习问题，区分设计与独有机制收益 | 三数据集当前直接解码；FBFull89.90/Uniform86.48/Full配额匹配89.83；明确机制边界 |
| Method | 如实定义训练图画像、B1违例、混合采样与固定bank | 五维均log1p；度对不同头/尾实体取均值；相似度分母1+两特征绝对值；不改变算法 |
| Experiments: configurations | 展示当前验证选定配置相对本地B1的表现 | 表tab:latest-direct；与候选损失/采样纯效应分开 |
| Experiments: controls | 识别固定Full配置下的候选来源差异 | 表tab:fb-controls；保留Profile负结果及频率控制近零差异 |
| Experiments: matched ranking | 检验频率配额近零差是否仅来自解码方式 | tab:fb-rerank；Full/配额/Uniform H@1=94.85/94.72/93.68；Full−配额净13题 |
| Experiments: cross-dataset sources | 比较NELL与WN候选来源、解释FB观察的外推范围 | tab:cross-dataset-sources；NELL方向反转，WN历史池条件正差；WN保持探索性 |
| Appendix: cross-source pairs | 完整报告两图三来源配对及核验限制 | tab:cross-source-pairs；WN11选项、55条未核实实体对、B1差异；NELL检查点例外 |
| Analysis: transitions | 展示纠错、退化及净变化 | 表tab:fb-paired；Full与配额匹配348/341，689题不同 |
| Analysis: frequency slices | 检查是否存在低频补偿性收益 | 表tab:fb-frequency-slices；train边界215/689；净−17/+1/+23，未显示低频优势 |
| Analysis: sampling distributions (prepared) | 比较原始关系边质量与实际负库，区分边缘与条件变化 | 六图与插入片段已导出；四图JS/TV及QA计数核验完成；未自动增加正文篇幅 |
| Analysis: K sensitivity | 检验更多训练负例是否单调改善排序 | tab:fb-k-sensitivity；K3/9/15同十选项；非等FLOPs、不重新选K |
| Discussion / Conclusion | 把机制解释限定在证据允许范围 | 频率替代解释未排除，配额来源Full且非独立通用频率方法；不宣称统计等效/因果/跨seed稳定 |
| Appendix: reproducibility | 区分历史与当前参数，完整报告关系等权指标 | FB32/120、实际loss温度权重；表tab:fb-macro；所有层级和方法 |
| Appendix: historical ranking | 保留独立模型的NELL重排、旧直接解码与成本 | tab:nell-results/tab:direct-decode/tab:cost，明确旧模型身份；WN与历史FB i0亦单列 |

证据状态：主文的配置、来源、重排、跨图比较、逐题、频率分层与K敏感性显示中，逐题与频率分层复用同一五模型测试结果，不作独立重复计数。NELL/WN来源控制和当前模型完整原选项重排均已完成并写入；CoDEx来源控制与当前模型完整重排、多种子及完整开放世界风险审计仍未完成。结果陈述不等于稿件已达到投稿证据门。

叙事顺序：候选可追溯构造 → 当前配置表现 → 更严格的FB候选替换控制 → FB频率配额近零差与低频负结果 → NELL/WN跨图及协议差异 → 条件性解释和适用范围。


正式稿新增analysis.tex；训练频率分层与K敏感性分设两小节，各保留主文显示。三来源排序与K分析均为单seed，不能将复用K9计作重复实验。
