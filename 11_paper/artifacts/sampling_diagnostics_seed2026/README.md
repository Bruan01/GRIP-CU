# 文献启发的采样机制诊断：四数据集

本轮对已有冻结候选库及B1评分做CPU分析，没有新增训练、推理或采样。
[调研依据](../../review/sampling_plot_literature_20261110.md)与[预先定义](figure_contract.md)说明图的用途及边界；[八页图册](../../main/figures/sampling_diagnostics_seed2026/sampling_diagnostics_all.pdf)提供最终可读结果。

## 数据与指标

四图共11,507道训练关系QA、103,563个Full负例槽位、1,469,265个逐题合法候选。数据集次序为NELL23K、CoDEx-M、FB15K237-2、WN18RR，对应2,974、2,715、3,109、2,709题。K=9、seed=2026。

结构相似度从冻结关系画像的五个log1p特征重建，严格使用项目原公式；困难度使用生成候选时缓存的`pairwise_confusion`，不重新选择温度，也不解释为校准的真假概率。所有Full所选候选的结构分数、困难度及候选模型分数与原缓存的最大绝对误差均为0。

每题先在其合法池内分别对结构分数、困难度排序，百分位为`(average_rank - 0.5)/pool_size`，并列取中秩。每题等权；实际bank中每个负例权重为`1/(n*9)`，解析Uniform中每个合法候选权重为`1/(n*pool_size)`。因此解析Uniform的每题平均百分位精确为0.5，但其直方图不保证平坦。横轴百分位跨图可比的是相对排序，不是绝对分数或校准难度。

所有分布固定为[0,1]上10个等宽桶。联合图为`log2(P_Full(cell)/P_Uniform_expected(cell))`：0表示同样质量，+1表示2倍，-1表示一半。无合法质量的格子屏蔽并标×；Full为0但参考非0时是负无穷，使用色条下溢端，不加平滑伪计数。色限固定为[-2,2]；绝对log2富集≥0.5的格子加正负号，辅助灰度阅读。

## 实测发现

| 数据集 | Full平均结构百分位 | Full平均困难度百分位 | Full−实际Uniform困难度，百分点 | Full−配额匹配困难度，百分点 |
|---|---:|---:|---:|---:|
| NELL23K | 0.5498 | 0.5575 | +5.87 | +4.95 |
| CoDEx-M | 0.5765 | 0.5520 | 未运行 | 未运行 |
| FB15K237-2 | 0.5531 | 0.5725 | +7.06 | +6.75 |
| WN18RR | 0.5105 | 0.5101 | +1.06 | +0.11 |

CoDEx-M的Full较解析Uniform的平均困难度百分位提高5.20个百分点；这不是已训练Uniform对照。相对解析Uniform，结构与困难度同时位于题内前20%的负例质量分别富集约1.76、1.56、2.01、1.09倍。这些数值描述静态库的选择行为，不是准确率增益。

逐题配对结果也不同：NELL和FB中，Full平均困难度高于实际Uniform的题目分别占67.18%和68.96%；WN为54.15%。WN中Full与配额组的题均困难度百分位在64.78%的题目相同，不能将其明显的下游性能差异简单归因为更困难的负例。

FB已有单信号组提供了直接检查：Profile-only平均结构/困难度百分位为0.5432/0.5001，Violation-only为0.5107/0.5700，Full为0.5531/0.5725。采样侧的两个信号表现符合其设计方向；这不证明联合机制导致性能收益，尤其不能忽略已有Profile-only负性能与Full/配额近零性能差异。

关系边际散点保留了旧审计的结论：实际负例分布比原始训练边分布平坦；不能把这些图写成“成功拟合原始数据分布”。本轮没有拟合幂律、混合分布或最终模型的预测分布。

## 范围限制

- WN沿用历史合法池及55道实体对未核验题；这些题没有在本轮被修复。其小合法池接近被K=9选满，限制了选择空间。
- 三个已有来源对照使用实际、逐题对齐的bank；CoDEx只有Full与解析参考。FB额外单信号诊断也使用已完成bank。尚在独立运行的frequency-only新训练不纳入本固定队列。
- 仅一个冻结bank实现，无独立种子置信区间；不从候选分数推定真假负例。
- B1分数用于Stage 2候选生成，不是SC-CL最终模型分数。联合富集不能单独验证因果交互。

## 文件与复算

`source_data/`保存直方图、二维联合质量、16组方法汇总、11,507行逐题统计，以及从上一轮审计原字节复制的关系边际。`provenance.json`记录原始bank、冻结评分、画像、mask和分析脚本SHA256；不复制大体积原始评分或第三方论文。

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/artifacts/sampling_diagnostics_seed2026/analyze_diagnostics.py
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/artifacts/sampling_diagnostics_seed2026/validate_exports.py
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/main/figures/sampling_diagnostics_seed2026/build_figures.py
```

第一步需要原始归档。后两步只需要本包CSV。`validation.json`核对全部所选候选、合法池与题目；CoDEx的池来自冻结评分标记，没有额外独立pool审计。`independent_validation.json`独立核对质量守恒、联合与一维边际相等、逐题与汇总均值相等、解析均值0.5及WN范围。图形导出与人工渲染检查见图包QA记录。
