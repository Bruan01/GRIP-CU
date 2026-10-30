# 论文叙事修订记录（2026-10-26）

## 修订依据

- [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide)：当前模板是工作稿；正文需要把主要论点放在读者必读部分，附录承担细节。
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)：摘要、引言中的贡献和适用范围应与实验及限制一致；复现条件和不确定性应透明。
- [RotatE，ICLR 2019](https://openreview.net/pdf?id=HkgEQnRqYQ)：从明确的关系建模问题进入方法原则，再给证据，不用实现细节开头。
- [SimKGC，ACL 2022](https://aclanthology.org/2022.acl-long.295/)：摘要将任务、具体负例设计和量化结果紧接在一起；读者能够迅速识别方法相对既有工作的改变。
- [Debiased Contrastive Learning，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)：负例可能是假阴性的风险先作为研究问题交代，再引出处理机制。

以上仅用于学习论证顺序和证据表达，不照搬措辞或跨任务性能结论。

## 证据与修订策略

| 问题 | 可核对状态 | 本轮动作 |
|---|---|---|
| 摘要过长，机制、结果和限制挤在一起 | NELL 单种子、full SC-CL、前 200 条 B1 配对；完整同预算对照缺失 | 按“问题—方法—观察—边界”重写，优先给出可比口径 |
| 引言反复强调限制，核心问题被冲淡 | 方法契约清楚，独立收益尚未证明 | 提前写出“难度与开放世界不确定性并存”的问题，末尾集中限定主张 |
| 相关工作偏文献罗列 | GRIP、SimKGC、SANS/KBGAN、ProGCL/DCL 与本文最相关 | 以“生成式关系预测”和“候选难度/假阴性”为两条主线重组 |
| 实验叙述混用绝对结果和训练增益 | full SC-CL 只有绝对值；prefix-200 有同候选配对 | 将两类观察分开表述，避免把 95.27% MRR 写成对 B1 的全量收益 |
| 讨论、结论重复结果表 | 缺同预算随机、五种子、完整假阴性审计 | 讨论聚焦可解释机制与替代解释；结论收束在可审计协议和单次运行 |

**不改变的证据边界：**没有全量 B1 候选重排、Random-9/frequency-matched 公平对照或多种子统计；WN18RR 的近期训练状态不能自动变成论文可报告的完整评估结果。图内简写继续以正文公式为准。

## 完成与核对

已修订 `main.tex` 中的摘要，以及引言、相关工作、实验、讨论和结论。摘要先交代开放世界关系候选的困难与不确定性，再给出 SC-CL 的采样与训练机制，最后分别报告全量绝对值、前缀配对观察和自由生成结果。正文对应区分这些评估口径，并明确缺失的对照。

XeLaTeX/BibTeX 编译通过，最终 PDF 为 9 页。`main.log` 无硬错误、未解析引用或 overfull 提示；详细构建信息见 `artifacts/build_report.md`。
