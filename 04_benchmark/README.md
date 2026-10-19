# Benchmark assets

当前主线只使用两个数据集：`NELL23K` 和 `FB15K237-2`。

本地数据目录如下：

```text
04_benchmark/data/
├── nell23k/
│   ├── raw/        # train/valid/test/entity2text
│   ├── processed/  # processed_val/processed_test
│   └── tasks/      # full Original GRIP task JSON
└── fb15k237_2/
    ├── raw/        # train/valid/test/entity2wikidata/rel2text
    ├── processed/  # standard GRIP graph records
    └── tasks/      # full relation/entity QA task JSON
```

`data/` 已加入根目录 `.gitignore`，因此这些数据和 task JSON 不提交到 Git。源文件来自 `/mnt/e/GRIP-CU` 的既有实验资产；权重仍留在 E 盘，不复制进仓库。

## 三条比较路径

每个数据集都使用相同的三臂协议：

1. Original GRIP/B1；
2. Random-9 + InfoNCE；
3. ErrorMemory-9 + InfoNCE。

ErrorMemory、候选分数和已知事实过滤只允许读取 `raw/train.txt`；valid/test 只用于评估。

## E 盘可复用资产

- 完整 Qwen2.5-7B 基座模型：`/mnt/e/GRIP-CU/model_cache/Qwen--Qwen2.5-7B-Instruct`；
- NELL23K 的可用 Original GRIP 7B adapter：`13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter/`；
- FB15K237-2 历史实验目录：`/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/`，保留了数据、任务、日志和 adapter 配置/Tokenizer，但当前未发现 `adapter_model.safetensors`，不能直接作为 FB B1/S1 权重使用。

正式运行时通过环境变量显式传入完整 adapter 路径，并在 `environment.txt` 记录 hash；不能把历史目录或不完整 adapter 直接当作新实验结果。当前 runner 已移除 0.5B 默认路径。
