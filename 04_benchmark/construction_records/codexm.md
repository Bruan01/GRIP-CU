# CoDEx-M 构造记录

- 记录更新时间：`2026-10-24T06:47:03.644541+00:00`
- 构造状态：`relation_data_ready_task_inference_pending`
- 目标：`给定固定实体对后排序关系候选；保留 CoDEx 官方三元组分类硬负例用于独立审计。`

## 构造过程

从官方 CoDEx 仓库固定 commit 下载 train/valid/test 正例、valid/test 硬负例及英语实体/关系元数据；每个上游文件在转换前按 Git blob SHA-1 校验。生成实体/关系文本映射和标准 GRIP graph record；valid/test 覆盖全量官方正例，候选列表固定为 10 选 1。此次只做数据转换，没有调用模型或使用训练 GPU。

## 官方 split 与防泄漏检查

官方 split 为 train=185,584、valid=10,310、test=10,311，图上下文只使用 train。每个 valid/test 问题从训练关系标签中取 9 个负候选，排除 gold 与同一 (h,t) 的 train-known 关系，使用 seed=2026 固定候选列表。官方 valid/test negatives 是替换实体的三元组分类负例，保留在 raw 目录但不作为关系排序负例或固定实体对候选的真假标签。

## 产物元数据

```json
{
  "manifest": {
    "path": "04_benchmark/data/codexm/manifest.json",
    "size_bytes": 8825,
    "sha256": "f7b4310a65be1dca857b766f2c96ab016bb5b0300be16dba9ef23aecca6c62df"
  },
  "source_manifest": {
    "path": "04_benchmark/data/codexm/raw/source_manifest.json",
    "size_bytes": 3635,
    "sha256": "568782f5a008aa6d95dc3d948bfad07dbc49e9e051733705ae4cb9930d8efc4c"
  },
  "train": {
    "path": "04_benchmark/data/codexm/raw/train.txt",
    "size_bytes": 3660213,
    "sha256": "d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556"
  },
  "valid": {
    "path": "04_benchmark/data/codexm/raw/valid.txt",
    "size_bytes": 203470,
    "sha256": "11c323096367354940846b9ed940c5dabddc1d20a70474222babe7e7fd28b28d"
  },
  "test": {
    "path": "04_benchmark/data/codexm/raw/test.txt",
    "size_bytes": 203693,
    "sha256": "0575ce05e4ce915e395bb4f708e8df9547989dd4c8c71407cbfd78c19daffcf7"
  },
  "valid_verified_negatives": {
    "path": "04_benchmark/data/codexm/raw/valid_negatives.txt",
    "size_bytes": 189665,
    "sha256": "454a4f66ae85495e6e7df0a9f591ed3b8b6efbbcca515414cb641a28625cbcb2"
  },
  "test_verified_negatives": {
    "path": "04_benchmark/data/codexm/raw/test_negatives.txt",
    "size_bytes": 189932,
    "sha256": "7caaf71f7a85fbd915dce8199f5b1e1744ead58e0675549fcb01c60599944648"
  },
  "entity_text": {
    "path": "04_benchmark/data/codexm/raw/entity2text.json",
    "size_bytes": 2843570,
    "sha256": "bc210ff0bdd851a63f13615b3d35ba3d126a79d8fb5eceef0ea77b6e19c701f9"
  },
  "relation_text": {
    "path": "04_benchmark/data/codexm/raw/rel2text.json",
    "size_bytes": 1416,
    "sha256": "7fcf1441b5ee1cb97f03b496a4b81dec136d83c594e5faffa7c503a2dbcdc501"
  },
  "processed_val": {
    "path": "04_benchmark/data/codexm/processed/processed_val.json",
    "size_bytes": 35710401,
    "sha256": "44a6f19c6e65512de897e181f5088bcafcaf1a25af42dd073b855379ba3ea00b"
  },
  "processed_test": {
    "path": "04_benchmark/data/codexm/processed/processed_test.json",
    "size_bytes": 35723751,
    "sha256": "5e7a8469e5365e7f3e6b367dd3ab9333acb12957f0dd2984fb94513d6dc27fbe"
  },
  "generation_plan": {
    "path": "04_benchmark/data/codexm/tasks/generation_plan.json",
    "size_bytes": 779,
    "sha256": "d6bdc889d28b896002c9adb4db2be6e6a9643a49eecfb8c35a0ff80f0b26701c"
  },
  "candidate_and_negative_audit": {
    "val": {
      "rows": 10310,
      "minimum_legal_negative_pool": 48,
      "candidate_count_per_query": 10,
      "negative_candidate_count": 92790,
      "train_known_collision_candidates": 0,
      "heldout_positive_overlap_candidates": 8,
      "queries_with_heldout_positive_overlap": 8,
      "open_world_unresolved_candidates": 92782,
      "holdout_overlap_note": "Measured after candidate lists were frozen; valid/test facts do not filter or change candidate sampling."
    },
    "test": {
      "rows": 10311,
      "minimum_legal_negative_pool": 48,
      "candidate_count_per_query": 10,
      "negative_candidate_count": 92799,
      "train_known_collision_candidates": 0,
      "heldout_positive_overlap_candidates": 6,
      "queries_with_heldout_positive_overlap": 6,
      "open_world_unresolved_candidates": 92793,
      "holdout_overlap_note": "Measured after candidate lists were frozen; valid/test facts do not filter or change candidate sampling."
    }
  }
}
```

## GPU 时间

```json
{
  "this_run_gpu_seconds": 0,
  "full_task_generation": "not run; Stage1/Stage2 remains pending_inference",
  "model": "not invoked; CoDEx-M provides the text metadata needed for data conversion"
}
```

## Stage1 / Stage2 样例

```json
[]
```

## 下游结论

PASS：原始 split、实体/关系映射及全量 10-way valid/test 关系评估记录已生成；adapter 训练前仍需生成并审核完整 Original GRIP Stage1/Stage2 task JSON。
