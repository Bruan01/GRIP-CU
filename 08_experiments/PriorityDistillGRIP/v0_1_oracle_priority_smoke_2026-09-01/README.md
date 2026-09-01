# PriorityDistill-GRIP v0.1 — Oracle Priority Smoke

- **日期**：2026-09-01
- **状态**：`READY_FOR_WSL_GPU`
- **主数据集**：NELL23K strict exact-hop 1/2/3/4-hop
- **开发环境**：macOS（数据与静态检查）
- **正式环境**：Windows WSL2 + RTX 3090 24GB

## 1. 研究问题

> 如果训练时知道完美的关键路径，并将这种路径监督蒸馏进普通等 rank LoRA，推理时完全不访问图，是否能比普通 GRIP/answer-only LoRA 和同 token 预算的额外 QA 训练更准确？

这不是先堆一个复杂 PathMind 模块，而是测试该方向最关键的 oracle upper bound。若完美路径都没有稳定作用，learned path scorer 只会加入额外误差和工程复杂度。

## 2. 论文主张与反主张

### 主张 C1

关键路径的**选择质量**可以提高图知识参数内化，而不只是增加训练数据。

### 支撑主张 C2

路径教师的收益在推理时仍保留；测试提示中没有候选路径，模型不访问原始 KG。

### 必须排除的反主张

- 只是训练 token 更多；
- 任意合法路径文本都能产生同样提升；
- 把所有路径塞进去与路径优先选择一样有效；
- 只改善 1-hop 记忆，没有改善 3/4-hop。

## 3. 注册方法

| 方法 | Stage 1 | Stage 2 | 作用 |
|---|---|---|---|
| `answer_only` | 无 | 统一 answer-only | 原始预算基线 |
| `more_qa_equal_token` | 重复 answer-only QA，匹配 Stage-1 input token | 统一 answer-only | 排除“只是更多 token” |
| `random_path_equal_token` | 同深度但与查询无关的合法路径 | 统一 answer-only | 排除任意路径文本效应 |
| `all_paths_equal_token` | gold + 3 distractors，无 gold 标签 | 统一 answer-only | 检验未筛选路径噪声 |
| `oracle_priority_equal_token` | 仅 gold path | 统一 answer-only | 完美路径优先的 oracle 上界 |

Stage 1 以 `all_paths_equal_token` 的实际 tokenizer input tokens 为参考预算。其余 Stage-1 方法确定性循环训练样本直到达到相同预算，并记录 overshoot、optimizer steps、supervised tokens、padding tokens、墙钟时间与峰值显存。Stage 2 对所有方法使用相同 answer-only 数据、4 个 epoch 和相同超参数。

## 4. 数据边界

来源为已审计的严格 exact-hop 数据：

```text
08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/artifacts/exact_hop/
├── nell23k_exact_hop_train.jsonl       # 716 = 179 × 4 depths
├── nell23k_exact_hop_validation.jsonl  # 152 = 38 × 4 depths
└── nell23k_exact_hop_test.jsonl        # 156 = 39 × 4 depths
```

候选池只从 train split 构造：

- gold path + 3 distractors；
- distractor 与 query 深度相同；
- 排除相同 task 和相同 answer；
- 优先 relation-chain overlap 较高者；
- stable hash 打破并列并固定顺序；
- validation/test 从不进入训练候选池。

生成审计：[`artifacts/supervision/audit.json`](artifacts/supervision/audit.json)。

## 5. 推理边界

训练提示与评测提示完全分离。评测只输入原始 graph-path question：

```text
You are answering a graph path query.
Follow the ordered relation chain exactly.
Return only the final entity identifier and no explanation.

Question: {question}
Answer:
```

评测代码有显式泄漏断言，禁止训练期的 `Candidate evidence:` 区块出现在 validation/test prompt 中。`run_summary.json` 固定记录：

```json
{"inference_graph_access": false}
```

## 6. 模型与训练

- Backbone：`Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank：8
- LoRA alpha：16
- Target modules：`down_proj`, `up_proj`, `gate_proj`
- Stage-1 LR：`5e-4`
- Stage-2 LR：`1e-3`
- Token-threshold optimizer step：2048 input tokens
- Seeds：42 / 43 / 44
- 推理显式 graph-free

实现为本版本独立快照，不从 StructuredLoRA 或 Original GRIP 运行时 import。

## 7. Go / Stop Gate

三个 seeds 后，`GO_LEARNED_PRIORITIZER` 要求同时满足：

1. oracle − answer-only ≥ 2 percentage points；
2. oracle − equal-token More-QA ≥ 1 point；
3. oracle > random path；
4. oracle > all paths；
5. oracle 的 3/4-hop 合并准确率高于 answer-only；
6. 四个 Stage-1 方法的 input-token 相对差距 ≤ 1%。
7. 四个 Stage-1 方法的 prompt 截断率均为 0。

单 seed 只能产生 `PRELIMINARY_GO` 或 `PRELIMINARY_STOP`。

失败时记录：

```text
STOP_PRIORITY_DISTILL
```

只有最终通过才创建 v0.2 learned prioritizer。

## 8. macOS 静态验证

```bash
cd 08_experiments/PriorityDistillGRIP/v0_1_oracle_priority_smoke_2026-09-01
bash configs/check_static.sh
```

当前已通过：

- 19 个单元测试；
- 716/152/156 split 与 SHA256 检查；
- 2,864 条 candidate records；
- 五种监督数据构造；
- train-only leakage guard；
- graph-free evaluation prompt guard；
- suite dry-run；
- Python bytecode compilation。

macOS 当前 Python 环境未安装 PyTorch，因此 LoRA runtime self-test 留在 WSL preflight 执行。

## 9. WSL 3090 执行

```bash
git pull
cd 08_experiments/PriorityDistillGRIP/v0_1_oracle_priority_smoke_2026-09-01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-wsl.txt
bash configs/run_wsl_smoke.sh
```

三 seeds 只在 one-seed smoke 值得继续时执行：

```bash
bash configs/run_wsl_full_seeds.sh
```

不要直接跳到 full seeds，也不要在 v0.1 中加入 learned scorer。

## 10. 关键入口

- 注册配置：[`configs/oracle_priority_smoke.json`](configs/oracle_priority_smoke.json)
- 数据构造：[`scripts/build_supervision.py`](scripts/build_supervision.py)
- 单次运行：[`scripts/run_experiment.py`](scripts/run_experiment.py)
- suite：[`scripts/run_suite.py`](scripts/run_suite.py)
- 汇总 gate：[`scripts/summarize_suite.py`](scripts/summarize_suite.py)
- WSL 交接：[`NEXT_STEP_WSL_PROMPT.md`](NEXT_STEP_WSL_PROMPT.md)
- 项目记忆：[`PROJECT_MEMORY.md`](PROJECT_MEMORY.md)
