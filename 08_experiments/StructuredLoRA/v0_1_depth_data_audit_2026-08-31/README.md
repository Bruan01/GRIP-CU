# StructuredLoRA v0.1 — NELL23K Depth Data Audit

## Experiment ID

```text
structured_lora_v0_1_depth_data_audit_2026-08-31
```

## Goal

在编写 StructuredLoRA 模型代码之前，验证两件事：

1. NELL23K 是否存在可复现、不会把训练边本身算作一跳的结构深度标签；
2. 是否能从 GRIP 自带 train graph 生成严格、有 provenance 的 1/2/3/4-hop 参数化图推理任务。

## Why this gate comes first

原始 GRIP reasoning QA 随机抽取多条边，不保证边连通、路径长度、无捷径或答案唯一，因此不能直接监督 hop router。NELL23K 标准关系预测又不等于显式多跳问答。若没有先修复标签定义，后续“深度专家”很可能只是在拟合噪声。

## Implemented

### A. NELL23K support depth

- train：对每条 triple 做 leave-one-edge-out；
- validation/test：只在 train graph 上寻找支撑路径；
- 同时导出 directed 和 undirected depth；
- 多关系平行边被视为合法的一跳替代支撑；
- self-loop 单独标记；
- 没在四步内找到路径的样本标为 `>4_or_unreachable`，不虚构为全局不可达。

### B. Exact-hop path QA

每条任务满足：

- directed simple path；
- 路径长度严格为 1/2/3/4；
- 起点到终点不存在更短 directed path；
- 给定 relation chain 只有一个答案；
- 保存 `path_nodes`、`path_relations` 和 `path_edges`；
- 固定 seed 后完全可复现。

### C. Existing prediction re-analysis

重新读取 RecurrentGRIP v1.1.1 的 768 条 diagnostic predictions，根据问题端点和 target relation 连接新 support-depth 标签。旧文件中的 `true_hop` 不作为依据。

## Run on macOS or WSL CPU

```bash
cd 08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31
bash configs/check_mac_static.sh
```

该实验只使用 Python 标准库，不需要 CUDA、Torch、Transformers 或 PEFT。

## Verified result

```text
unit tests:             8/8 PASS
support-depth labels:   34,216
exact-hop tasks:        1,024
prediction join:        768/768
artifact determinism:   PASS
runtime on local Mac:   about 13–15 seconds
final decision:         GO_ORACLE_PREFIX
```

NELL23K test 的无向 support-depth 分布：

| Bucket | Count | Fraction |
|---|---:|---:|
| self-loop | 2 | 0.04% |
| 1 | 614 | 12.42% |
| 2 | 670 | 13.55% |
| 3 | 2,260 | 45.71% |
| 4 | 427 | 8.64% |
| >4 or unreachable | 971 | 19.64% |

## Interpretation

结果支持进入 **oracle-prefix feasibility pilot**，但不支持直接声称 learned routing 有效：

- 2/3/4 三个非平凡 test bucket 都超过 100 条；
- test relation-depth NMI=`0.159158`，depth 不等于 relation identity；
- exact-hop 任务生成完整；
- 现有诊断预测显示 depth-wise variation，但每个条件只有 32 validation 和 64 test questions，因此只是弱信号。

训练集的 leave-one-out 无向标签中，`63.73%` 落在 `>4_or_unreachable`。正式 NELL23K 训练不能简单把这一大桶当作“真实四跳”；v0.2 首先在 exact-hop 控制集验证机制，NELL23K 只做接口和性能辅助。

## Artifacts

```text
artifacts/
├── depth_labels/nell23k_support_depth.jsonl.gz
├── exact_hop/
│   ├── nell23k_exact_hop_train.jsonl
│   ├── nell23k_exact_hop_validation.jsonl
│   ├── nell23k_exact_hop_test.jsonl
│   └── generation_metadata.json
└── reports/
    ├── REPORT.md
    ├── summary.json
    ├── split_integrity.json
    ├── depth_distribution.csv
    ├── frequency_depth_distribution.csv
    ├── relation_depth_distribution.csv
    ├── baseline_accuracy_by_depth.csv
    └── prediction_join_audit.json
```

## Go/No-Go

**GO：创建 `v0_2_oracle_prefix_smoke`。**

v0.2 必须比较：

1. equal-rank monolithic LoRA；
2. static split、所有组开启；
3. flat routed experts；
4. ordered oracle prefix；
5. oracle prefix + depth-local credit；
6. permuted labels 和 random group order。

如果 oracle prefix 不能超过 equal-rank monolithic LoRA，就终止 StructuredLoRA，不实现 learned router。
