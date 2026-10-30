# Dataset roles

更新日期：2026-10-24

当前论文主线限定为给定实体对 `(h,t)` 后排序关系候选。主实验矩阵使用四个关系预测数据集；CLEGR 资产保留在工作区，但不进入本文。

| 数据集 | 本地目录 | 主要作用 | 关系标签 | 当前资产状态 |
|---|---|---|---|---|
| NELL23K | `04_benchmark/data/nell23k/` | 核心方法与对照 | `concept:*`，直接读取训练关系 | 有 seed=2026 SC-CL；匹配对照未齐 |
| FB15K237-2 | `04_benchmark/data/fb15k237_2/` | 核心外部图验证 | `raw/rel2text.json` 映射 | 有历史 B1/i0；当前 SC-CL 主结果待跑 |
| WN18RR | `04_benchmark/data/wn18rr/` | 标准关系预测迁移 | `raw/wn18rr.json` / 训练关系集合 | 原始与 processed 数据、task 文件已有 |
| CoDEx-M | `04_benchmark/data/codexm/` | 标准关系预测迁移和负例风险审计 | `raw/rel2text.json` 映射为英文标签 | 原始 split、关系文本和全量 valid/test QA 已转换；train-only Stage1/Stage2 任务由本地 vLLM 生成中 |

CoDEx-M 固定在官方仓库 commit `3132e426c2a6b643b70bad679905a3a6270be440`。该版本包含 17,050 个实体、51 个关系、185,584/10,310/10,311 条 train/valid/test 正例。评估 QA 覆盖完整 valid/test 正例 split；每行固定 10 个候选（gold + 9 个负候选），从训练关系标签中确定性采样，并排除同一 `(h,t)` 的 train-known 关系。seed 和文件哈希见 `codexm/manifest.json`。

## 数据目录合同

关系预测数据集的原始官方 split 必须包含：

- `raw/train.txt`、`raw/valid.txt`、`raw/test.txt`；
- `processed/processed_val.json` 和 `processed/processed_test.json`；
- 用于 adapter 训练的完整 Original GRIP task JSON，或明确标记待推理的 `tasks/generation_plan.json`。

CoDEx-M 的 adapter task generation 单独读取 `processed_train_graph.json` 和 `processed_train_reference.json`：图只含 `raw/train.txt` 边，reasoning QA 的 8-shot 示例也只从训练 QA 采样。valid/test QA 不进入任务生成输入。

## CoDEx-M 负例边界

官方 `valid_negatives.txt` 和 `test_negatives.txt` 保留了与 CoDEx-M 正例数量对应的三元组分类硬负例。它们主要通过替换尾实体构造，与固定 `(h,t)` 后替换关系的 SC-CL 候选空间不同。因此：

- 不把这两个文件作为 SC-CL 的关系负候选或真假标签；
- 关系排序负例从训练关系词表产生，并过滤该实体对已知的训练关系；
- train-known 冲突、valid/test 正例重叠和开放世界未判定候选分别审计；
- 未观测事实不直接记为假。

## 防泄漏和任务文件

ErrorMemory、候选分数、关系画像、负候选采样和已知事实过滤只允许读取 `raw/train.txt`；valid/test 仅用于评估。对 CoDEx-M，`entity2text.json` 与 `codexm.json` 保存上游英语实体/关系元数据子集，`rel2text.json` 提供原始关系 ID 到任务标签映射；`entity2text.txt` 是仅由 train 实体生成的 TSV，供 SC-CL 的 train-known 实体对过滤使用。

GRIP 的 Stage1/Stage2 任务由 `GripTaskGeneration` 构造：图事实和未改写的边 QA 由脚本生成，summary/reasoning QA 需要模型推理。CoDEx-M 的关系评估数据不代替这份 adapter 训练任务；task JSON 完成后需检查 generation plan、生成状态和缓存记录再训练。

## 其他资产

`clegr_facts` 与 `clegr_reasoning` 仍以图级问答基准形式保留，但其问题、答案空间和反事实单位都不同于固定实体对上的关系候选排序，不作为当前论文的迁移数据集或 SC-CL 结果。

Canonical FB15K-237、YAGO3-10、CoDEx-S/L 暂不优先扩展。若未来加入，需单独固定版本、转换关系文本和任务格式，并建立对应 adapter 与审计契约。

## FB15K237-2 说明

这里使用 GRIP 代码中的 `fb15k237_2` 版本，而不是另一个空的下载占位目录 `fb15k237/`。FB 的 raw relation ID 通过 `raw/rel2text.json` 转成 task 中使用的文本关系；runner 需要显式传入这个映射文件。
