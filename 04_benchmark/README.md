# Benchmark assets

当前论文的关系候选排序主线使用 `NELL23K`、`FB15K237-2`、`WN18RR` 和 `CoDEx-M`。CLEGR（`clegr_facts` / `clegr_reasoning`）数据资产仍保留在目录中，但不纳入当前论文。

本地数据目录如下：

```text
04_benchmark/data/
├── nell23k/
│   ├── raw/        # train/valid/test/entity2text
│   ├── processed/  # processed_val/processed_test
│   └── tasks/      # full Original GRIP task JSON
├── fb15k237_2/
│   ├── raw/        # train/valid/test/entity2wikidata/rel2text
│   ├── processed/  # standard GRIP graph records
│   └── tasks/      # full relation/entity QA task JSON
├── wn18rr/
│   ├── raw/        # train/valid/test/entity2text/relations
│   ├── processed/  # JSONL graph records readable by the GRIP runner
│   └── tasks/      # Stage1/Stage2 generation plan and cache location
├── codexm/
│   ├── raw/        # pinned official CoDEx-M splits, text maps, verified triple-classification negatives
│   ├── processed/  # valid/test QA and train-only graph/reference inputs for task generation
│   └── tasks/      # Stage1/Stage2 generation plan, resumable local-vLLM cache, task JSON
├── clegr_facts/
│   └── processed/  # complete 300/100/100 graph-level split
└── clegr_reasoning/
    └── processed/  # complete 300/100/100 graph-level split
```

`data/` 已加入根目录 `.gitignore`，因此这些数据和 task JSON 不提交到 Git。CoDEx-M 来源固定为[官方仓库 commit](https://github.com/tsafavi/codex/tree/3132e426c2a6b643b70bad679905a3a6270be440)，每个下载文件先按上游 Git blob SHA-1 校验；其它本地数据仍沿用已有实验资产。权重留在 E 盘，不复制进仓库。

`04_benchmark/data/manifest.json` 是当前数据总清单；每个数据集目录还有自己的 manifest。CLEGR 的大型 PyTorch 源张量保留在原实验资产位置，目标目录保存可直接使用的 JSONL 图记录和来源指针，不重复复制几十 GB 源文件。

仅重建 CoDEx-M（从固定官方 commit 下载并校验，不重跑其他数据准备）：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python \
  04_benchmark/preprocessing/prepare_benchmark_data.py --only-codexm

# Build train-only graph/reference inputs and the SC-CL train-entity text map
/home/kieran/miniconda3/envs/guardenv/bin/python \
  04_benchmark/preprocessing/build_codexm_train_generation_inputs.py
```

CoDEx-M 的 valid/test 关系 QA 覆盖官方完整正例 split，候选表固定为 10 个训练关系文本（gold + 9 个负候选），候选构造按同一实体对过滤 train-known 关系，并固定 seed=2026。官方 `valid_negatives.txt` / `test_negatives.txt` 保留作三元组分类审计；这些通过替换实体构造的硬负例不会被当作固定 `(h,t)` 关系排序中的真实性标签。

## Stage1/Stage2 任务生成

关系数据的原始图和已有任务文件保留不变。WN18RR、CoDEx-M 和 CLEGR 的 `tasks/generation_plan.json` 明确标记 QA、summary、reasoning QA 需要 Qwen-7B 推理生成，不能用静态关系 QA 替代。生成器和断点 cache 入口是：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python \
  04_benchmark/preprocessing/generate_grip_stage_tasks.py \
  --dataset wn18rr
```

CoDEx-M 的 Stage1/Stage2 任务使用本地 Qwen2.5-7B vLLM 生成，batch size 为 8，显存利用率上限设为 0.82 为桌面和系统留余量。正式输入图和 few-shot QA 均来自 train；valid/test 只用于关系预测评估，不作为生成提示。运行参数和可续跑入口记录在 `04_benchmark/preprocessing/run_codexm_task_generation_vllm.sh`，完成并审核任务 JSON 后再进行 adapter 训练。

先用 `--max-records 1 --num-context-qa 1 --num-reason-qa 1 --num-summarization 1` 做 7B 任务生成 smoke，再启动正式任务构造。

任务构造支持断点续传：每个模型阶段按 batch 写入并 `fsync` 到 `task-cache-dir/*.jsonl`，同时写入 `run_state.json`。中断后使用完全相同的命令和 cache 目录重跑即可，已经完成的记录按 `index + prompt_sha256` 复用；不要删除 cache，也不要更换随机种子或任务参数。

## 官方任务构造口径

最终 task JSON 不是只有 LLM 生成文本。官方 `GripTaskGeneration` 会合并四类任务：`context`（脚本直接把节点/边转成事实复述）、`context_qa`（边关系/头尾实体 QA，关闭改写和节点属性选项时也是脚本直接构造）、`summarization`（Qwen Stage1/Stage2）和 `reasoning_qa`（Qwen Stage1/Stage2）。CoDEx-M 也使用这套四类任务；其 summary/reasoning 阶段由本地 vLLM 批量生成，输入图与 few-shot reference 只含 train 数据。可对照 `13_base_method/grip-exp/grip/tasks/train_tasks/` 下的官方实现。

需要区分“官方实现兼容”和“官方完整参数复现”：NELL23K、FB15K237-2、WN18RR 沿用已复现的关系数据任务配置；CLEGR 当前 task JSON 使用同一官方生成器，但为单张 3090 采用每图 `2/2/2` 的低预算，并关闭节点属性 QA、context QA 改写和 context upsampling。官方 CLEGR reasoning shell 是 `20/160/20`，因此当前 CLEGR 文件可以作为低预算实验数据，不能直接宣称为官方完整预算产物。

## 后续比较路径

四个关系预测数据集后续比较以下改进臂。CLEGR 数据资产保留供独立研究使用，但因任务定义和答案空间不同，不作为当前 SC-CL 论文的迁移数据集：

1. Original GRIP/B1；
2. Random-9 + InfoNCE；
3. ErrorMemory-9 + InfoNCE。

ErrorMemory、候选分数和已知事实过滤只允许读取 `raw/train.txt`；valid/test 只用于评估。

五份构造过程、产物元数据、GPU 时间估算和 Stage1/Stage2 样例保存在 `04_benchmark/construction_records/`。进入 adapter 训练前，需检查对应记录和 `tasks/generation_plan.json` 或任务文件中的 generation status。

## E 盘可复用资产

- 完整 Qwen2.5-7B 基座模型：`/mnt/e/GRIP-CU/model_cache/Qwen--Qwen2.5-7B-Instruct`；
- NELL23K 的可用 Original GRIP 7B adapter：`13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter/`；
- FB15K237-2 历史实验目录：`/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/`，保留了数据、任务、日志和 adapter 配置/Tokenizer，但当前未发现 `adapter_model.safetensors`，不能直接作为 FB B1/S1 权重使用。

正式运行时通过环境变量显式传入完整 adapter 路径，并在 `environment.txt` 记录 hash；不能把历史目录或不完整 adapter 直接当作新实验结果。当前 runner 已移除 0.5B 默认路径。
