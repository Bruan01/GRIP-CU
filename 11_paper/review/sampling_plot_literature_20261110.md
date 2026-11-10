# 负采样图表调研与本项目改编（2026-11-10）

已阅读下列论文原文及相应图页。借鉴分析问题与表达方式，重新计算和绘图；不复制论文图片。

| 原始文献与图号 | 原文实际做法 | 本项目采用的部分与边界 |
|---|---|---|
| [ProGCL，ICML 2022](https://proceedings.mlr.press/v162/xia22b.html)，Fig.1、3、4 | 对比负例相似度直方图；Fig.4比较高斯与Beta混合拟合。 | 采用经验分布对比。我们的横轴为关系画像相似度或冻结B1分数，不称余弦相似度；没有真假负例标签，不能照搬混合成分的真假解释。 |
| [Structure Aware Negative Sampling in Knowledge Graphs，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.492/)，Fig.2、附录Table 8 | 比较不同跳数、游走数下的MRR，并展示不同来源的候选例子。Fig.2是性能敏感性图，不是分布图。 | 采用实际采样来源的受控比较。我们的结构来自关系统计画像，不是实体k跳邻域；静态候选库不能生成未做过的跳数敏感性曲线。 |
| [Understanding Negative Sampling in Graph Representation Learning，KDD 2020](https://arxiv.org/abs/2005.09863)，理论分析、Fig.3与6 | 在其条件节点采样设定下讨论正负分布的次线性关联；考察度数幂指数、负例数等对性能的影响。 | 提出“采样与原始频率如何关联”的诊断问题，用关系边际概率散点直接展示；不移植其条件分布定理，不虚构指数扫描，也不把贴合原始分布当作目标。 |
| [Understanding the Behaviour of Contrastive Loss，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.html)，主文Fig.3、补充Fig.1 | 分析相似度与梯度权重、温度的关系；补充图用箱线图呈现正例和不同近邻排序负例的相似度。 | 借鉴按题内排序看困难度、展示分布差异。改用逐题配对差异ECDF保留异质性；没有嵌入或温度轨迹，不补造相应曲线。 |

## 这次实际制作什么

1. 四数据集题内困难度/结构相似度百分位直方图，比较实际Full、Uniform、配额匹配与精确解析Uniform。
2. 四数据集结构×困难度联合采样富集图。它是本项目新增诊断，不声称上述论文绘制过同一张图。
3. 原始训练边关系概率与实际负例概率的双对数散点。
4. FB已完成来源组的平均结构/困难度百分位对照，检查单信号的采样表现。
5. 每数据集一张四面板图：原始结构分数、原始B1困难度、联合富集、逐题配对困难度差异。

CoDEx-M仅有解析Uniform参照。WN18RR保留历史池与55个实体对未核验样本。所有图描述候选生成过程；不代表SC-CL训练后预测分布，不提供真假负例判定，也不单独证明下游性能机制。

方法与预先约定见[figure_contract](../artifacts/sampling_diagnostics_seed2026/figure_contract.md)。可复算数值和逐题数据见[数据包](../artifacts/sampling_diagnostics_seed2026/README.md)，最终图与中文解读见[图包](../main/figures/sampling_diagnostics_seed2026/README.md)。
