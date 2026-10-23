# GRIP 的优势成立，但当前基线设计仍不足以隔离“图记忆”与“任务贴合”收益

更新日期：2026-09-19  
范围：GRIP 论文主结果、论文采用的基线、仓库内 Qwen2.5-7B NELL23K 复现结果  
指标口径：知识图谱任务为 10-way relation classification 的 Accuracy/Exact Match；本仓库同时复算 F1 与 Hit

## 执行摘要

- 论文中的核心基线共有三类：无图上下文 LLM、带 edge-list 图上下文的 LLM、零样本图基础模型 OFA/GOFA；另有只表示 token 可行性的 full-context 行，以及小图任务上的 GRIP+context 变体。
- GRIP-Qwen2.5-7B 在四个知识图谱数据集均超过 Qwen 无上下文和 sampled-context 基线。NELL23K 论文值为 **87.74%**，分别比 Qwen no-context 的 77.38% 高 **10.36 个百分点**、比 Qwen-context 的 81.38% 高 **6.36 个百分点**。
- RecurrentGRIP 仓库的全量复现为 **87.457% EM（4,323/4,943）**，与论文 87.74% 差 0.283 个百分点；但该复现并未严格按论文配置（QA 12,014 条而非 20,000、k-shot 用验证集等），且数据版本不同（198 vs 200 种关系），不能据此说「复现基本对齐」。它与本项目训练器训出的 B1（test 4,944 题：历史 84.91%、修复后 84.16%）是两条不同管线，见 `论文对齐与勘误.md`。旧登记表只记录 direct 子集，已不再代表当前状态。
- 论文的比较并非完全同条件：GRIP 在训练时看过整张训练图并使用与测试高度相似的 relation QA，context LLM 在大图上只得到采样子图；OFA/GOFA 数字取自原论文，未在 GRIP 管线中重跑；经典 KGE、强 KG-RAG/路径推理方法均缺席。
- 论文文字称 GRIP+context 在 reasoning tasks 上“一致优于”context baseline，但 Table 4 的 Qwen/CLEGR 实际为 **31.53% < 33.48%**。该表述需要修正。
- 数据口径有未解决冲突：论文写 NELL23K test=4,952；仓库原始文件只有 4,944 行；预处理器又因 `split("\n")[:-1]` 丢掉末行，最终评测 4,943 条。

## 1. GRIP 怎么做

GRIP 把一张图的知识写入图专属 LoRA，而不是在每次推理时把图转成文本塞入上下文。

1. Stage 1：图记忆
   - context memorization：分别记忆节点和边文本；
   - summarization/rephrasing：从随机节点、边和最多约 10 个节点的子图生成摘要及改写，注入局部与高阶结构。
2. Stage 2：图知识检索与推理
   - context QA：遮盖节点属性或边的头、关系、尾，训练模型补全；
   - reasoning QA：在采样子图上生成 multi-hop、global、binary、k-shot QA。
3. 推理：只输入问题，不再输入图；图信息由 LoRA 参数提供。

NELL23K 的论文配置为 LoRA rank 4、alpha 8、MLP 模块、Stage 1 最多 1 epoch、Stage 2 Qwen 最多 10 epoch、学习率 1e-3。仓库复现元数据只部分一致：LoRA 与模块一致（`down_proj/up_proj/gate_proj`、seed 2026），但 QA 只有 12,014 条（论文 20,000）、80,820 条 S1 样本套了聊天模板（论文为纯文本）、S2 未早停、ε=1e-8（论文 1e-4）。逐项对照见 `论文对齐与勘误.md` §1。

## 2. 论文性能总表

### 2.1 知识图谱任务 Accuracy（%）

| 模型 | FB15K237 | WN18RR | CoDEx-Medium | NELL23K |
|---|---:|---:|---:|---:|
| OFA | 70.84 | 30.96 | — | — |
| GOFA | 80.69 | 32.89 | — | — |
| Qwen2.5-7B no context | 79.65 | 14.74 | 83.09 | 77.38 |
| Qwen2.5-7B sampled context | 79.19 | 41.19 | 73.32 | 81.38 |
| **GRIP Qwen2.5-7B** | **88.75** | **64.30** | **94.42** | **87.74** |
| Llama3.1-8B no context | 56.66 | 28.08 | 71.03 | 20.05 |
| Llama3.1-8B sampled context | 44.68 | 53.86 | 47.46 | 53.33 |
| **GRIP Llama3.1-8B** | **77.79** | **63.26** | **91.58** | **77.40** |

注：WN18RR 报告 5 次运行均值，其余数据集为单次运行；OFA/GOFA 数字由 GRIP 论文从对应原论文引用，并非统一环境复跑。

### 2.2 推理任务 Accuracy（%）

| 模型 | Scene Graph | CLEGR-Reasoning |
|---|---:|---:|
| GOFA | 34.06 | — |
| Qwen2.5-7B no context | 3.54 | 15.81 |
| Qwen2.5-7B full context | **67.46** | **33.48** |
| GRIP Qwen2.5-7B | 53.27 | 27.66 |
| GRIP Qwen2.5-7B + context | **68.71** | 31.53 |
| Llama3.1-8B no context | 6.96 | 0.00 |
| Llama3.1-8B full context | 67.06 | 24.76 |
| GRIP Llama3.1-8B | 53.27 | **27.14** |
| GRIP Llama3.1-8B + context | **73.99** | **32.75** |

小图 reasoning task 的结论应更克制：GRIP 无上下文远强于 no-context LLM，但通常不及 full-context；GRIP+context 多数提升，但 Qwen/CLEGR 是反例。

### 2.3 输入 token 与效率

| 数据集 | context baseline 平均输入 token | GRIP 平均输入 token | 论文报告总推理时间降幅 |
|---|---:|---:|---:|
| FB15K237 | 1,330.56 | 155.74 | 35.0% |
| WN18RR | 1,624.53 | 156.01 | 30.1% |
| Scene Graph | 5,295.33 | 67.94 | 62.0% |
| CLEGR-Reasoning | 6,338.80 | 106.41 | 63.1% |

论文 Table 6 在 FB15K237/Llama3.1-8B 上给出：GRIP 一次训练 15,850 秒，单查询从 83.41 ms 降到 10.62 ms。按这组数字计算，训练成本约需 **21.8 万次查询**才由推理节省摊平；因此“训练成本可忽略”只在高复用查询场景成立。

## 3. 仓库 NELL23K 复现

| 运行 | n | EM | F1 | Hit | 说明 |
|---|---:|---:|---:|---:|---|
| Qwen2.5-7B full | 4,943 | **87.457%** | 87.457% | 87.497% | 与论文 87.74% 相差 -0.283pp（配置与数据版本均不同，仅供参考） |
| Qwen2.5-7B direct | 242 | 73.140% | 73.140% | 73.554% | 同一有向实体对在训练图中已出现，测试关系不同 |
| Qwen2.5-7B novel2hop | 307 | 92.182% | 92.182% | 92.182% | 独立生成候选集的子集文件 |
| Qwen2.5-7B patternnovel | 44 | 93.182% | 93.182% | 93.182% | 样本很小，单个样本约 2.27pp |
| Qwen2.5-0.5B full | 4,943 | 0.607% | 0.758% | 0.971% | 主要是输出实体形式而非关系形式，不能代表 GRIP 机制上限 |

7B full 的 99.84% 输出带 `concept:` 关系形式，复现健康。EM 与 F1 相同是因为正确答案和绝大多数输出均为单个关系标签；Hit 略高来自 2 个包含目标字符串但不完全相等的响应。

## 4. 参考基线是什么、怎么做、有什么问题

### 4.1 无上下文 LLM

做法：仅给 10-way 关系选择问题，不给目标图。Qwen2.5-7B 和 Llama3.1-8B 都作为基座比较。

问题：这是必要的泄漏/先验下界，但不是强图推理对手。它可能利用实体类型、关系标签语义和预训练事实，却无法访问训练图；GRIP 对它的优势同时混合了“获得图信息”和“方法更好”两个因素。

### 4.2 edge-list context LLM

做法：把图表示为自然语言 edge list。大知识图谱无法全文放入上下文，因此从两个查询实体周围采样子图；Scene Graph/CLEGR 小图则输入完整图。

问题：

- 大图结果高度依赖未充分报告的采样策略和检索召回率；缺边会导致无法回答，多余边会成为干扰。
- 它不是强 KG-RAG 或路径检索基线，不能据此得出 GRIP 超过检索增强方法的结论。
- GRIP 在训练时使用整张训练图，而该基线在推理时只见局部子图，信息预算不对称。
- edge list 虽准确率高于 indexed edge，但 token 冗余随节点度增长；这正是论文要解决的问题，也使其成为偏弱且昂贵的对照。

### 4.3 OFA

做法：把跨领域图统一为 text-attributed graph，用语言模型把节点/边描述编码到共同空间；用 nodes-of-interest 统一 node/link/graph 任务，再向图中附加 prompt subgraph，使同一图模型可做监督、少样本和零样本分类。[OFA 原论文](https://arxiv.org/abs/2310.00149)

问题：GRIP 只报告 OFA 在 FB15K237/WN18RR 的数值，且直接从 OFA 论文摘录。模型、数据处理、训练语料、负采样和评测实现没有在同一管线对齐；因此它是参考点，不是严格控制实验。

### 4.4 GOFA

做法：在冻结 LLM compressor 层之间插入 token-level GNN；把节点/边文本压缩成 memory tokens，通过消息传递融合结构，再由 LLM decoder 生成答案。它使用图 next-word、结构任务和 QA-chain 预训练，并用 prompt node 统一不同任务。[GOFA 原论文](https://arxiv.org/abs/2407.09709)

问题：

- 预训练数据、架构、可训练模块与 GRIP 完全不同，比较包含大量系统差异。
- GRIP 论文同样直接引用 GOFA 数字，覆盖数据集不全。
- GOFA 自身承认大规模预训练耗时高、插入 GNN 会增加节点/边 embedding 与推理成本；但论文没有做统一硬件和统一吞吐比较。

### 4.5 full-context 与 GRIP+context

- full-context 在大知识图谱上是 token 需求/OOC 指示器，不是实际得到 accuracy 的模型。
- GRIP+context 只在小图 reasoning 数据上出现，用于测试 LoRA 与显式图上下文能否互补；它不应与纯 GRIP 混为同一方法。

## 5. 当前实验设计最需要警惕的问题

1. **任务贴合带来的优势未被隔离。** GRIP Stage 2 的 edge-level context QA 就是补关系/头/尾，NELL23K 测试又是 10-way 关系选择；论文也承认 link prediction 与训练目标更接近。缺少“同 token/同 QA 数量但与测试关系无关”的控制组。
2. **缺少经典 KGE 和强检索基线。** TransE/RotatE/ComplEx、路径推理、KG-RAG/GraphRAG 等都未进入主表；仅与 LLM 和跨域图基础模型比较，不能证明在标准 KG completion 方法中最优。
3. **信息预算不对称。** GRIP 离线吸收整张训练图；context baseline 在线只取局部子图。至少应报告检索召回、子图大小与 oracle-path 上界。
4. **跨论文数字不可完全比。** OFA/GOFA 未统一重跑，且仅覆盖部分数据集。
5. **方差报告不足。** 除 WN18RR 外均为单次运行；没有置信区间、标准差或显著性检验。
6. **10-way 候选任务可能高估开放世界能力。** 随机负关系可能被实体类型或关系词面排除；应补充 hard-negative、全关系 ranking、MRR/Hits@k。
7. **指标异质。** KG 用 exact match，reasoning 用 Qwen2.5-32B judge；不同数据集分数不能直接横比。
8. **Stage 2 默认替换 Stage 1 语料。** 仓库 `TaskDataset.enable_qa=True` 时默认只保留 QA，不回放 context，存在遗忘风险；需要 rehearsal ablation。
9. **效率结论依赖查询量。** 训练成本只有在大量复用同一图时才摊薄，应报告 break-even query count，而非只报告单次推理加速。
10. **NELL23K 行数不一致。** 论文 4,952、仓库原始 4,944、实际评测 4,943；发布结果前必须固定数据版本并修正解析器。

## 6. 建议的下一版基线矩阵

优先补以下控制，才能回答“GRIP 的收益究竟来自哪里”：

1. Qwen/Llama no-context、sampled-context 保留，并固定同一候选集。
2. 加入 TransE、RotatE、ComplEx 或当前公认 KGE 强基线，以 MRR/Hits@k 和 10-way accuracy 双口径报告。
3. 加入统一 retriever 的 KG-RAG：固定检索 token 与 GRIP 推理 token，并报告 evidence recall。
4. 加入 oracle-path context，得到“检索完美时”的上界。
5. 加入 GRIP task-control：保持 QA 数量、训练 token、LoRA 参数量不变，但去掉与测试同构的 relation QA。
6. 每个主结果至少 3 seeds，报告均值、标准差和 paired test。
7. 报告训练+任务生成成本、单查询成本和 break-even query count。

## 数据与来源

- GRIP 论文 PDF：`13_base_method/grip-exp/grip-paper.pdf`；机器抽取表格来自 Tables 2–6、13。
- 仓库复现预测：`13_base_method/grip-exp/outputs/grip_inf/nell23k/`。
- 复现元数据：`13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter/grip_run_metadata.json`。
- 可复核性能明细：`artifacts/paper_performance.csv`、`artifacts/split_metrics.csv`。
- 外部主来源：[GRIP arXiv](https://arxiv.org/abs/2511.07457)、[OFA arXiv](https://arxiv.org/abs/2310.00149)、[GOFA arXiv](https://arxiv.org/abs/2407.09709)。

