# Fixed-Depth RecurrentGRIP v1 Implementation Plan

更新日期：2026-08-28
状态：可执行实现计划
目标：在不改变 Original GRIP 默认行为的前提下，增加 CLEGR 固定深度 RecurrentGRIP Pilot 路径。

## 1. 先行结论：需要对 CLEGR Pilot 做一次方法级微调

当前 `data/raw_datasets/clegr/process.py` 只保留 `questions` 与 `answers`，丢弃了原始样本中的：

- `question_type`；
- `question_group`；
- `question_subgroup`。

同时，CLEGR-Reasoning 混合了 Filter、Aggregation、Topology 和 PathReasoning；不能把所有 reasoning 样本直接等同为 1–4 hop 图路径任务。

因此 v1 Pilot 采用更严格的可证伪子任务：

- 主任务只使用 `StationShortestCount`；
- 从问题中识别两个端点；
- 在图的 `edge_index` 上重新计算无向最短距离；
- 训练仅含 1–2 hop，测试仅含 3–4 hop；
- 数据集给出的 `StationShortestCount` 标签必须等于 `max(shortest_distance - 1, 0)`，否则样本进入 rejected 日志；
- 其他 PathReasoning 类型只作为 v1.1 扩展，不进入首轮成功判定。

这保证“hop”表示真实图距离，而不是按问题模板主观分级。

## 2. v1 机制边界

### 2.1 模型改动

将基座模型第 `executor_layer_index` 个 decoder block 替换为：

```text
FixedDepthRecurrentBlock(original_block, K)
```

该包装器在同一次前向中重复执行同一个 block：

```text
h_0 -> block(h_0) -> h_1 -> block(h_1) -> ... -> h_K
```

约束：

- 每轮严格共享同一个 block 对象和参数；
- v1 不加入 halt head、update gate、frontier loss；
- generation 强制 `use_cache=False`，避免同一 token position 在多轮 recurrence 中错误复用 KV cache；
- graph LoRA 只注入 executor layer；
- 基座参数保持冻结；
- 每图单独训练一个 graph adapter；
- Original GRIP runner 和默认 LoRA 路径不改变。

### 2.2 v1 中 storage / execution 的操作化定义

- storage：只位于 executor block 内的图专属 LoRA；
- execution：冻结的共享 decoder block 被重复调用 K 次；
- graph identity：adapter ID；
- compute：测试时 recurrence K；
- 无图推理：prompt 中不含 graph context、三元组、路径和真实 hop。

### 2.3 Pilot 暂不声称

- 共享 executor 已经完成跨图 meta-training；
- 中间状态等价于 BFS frontier；
- recurrence 自动对应一跳；
- 动态停止有效。

v1 只检验：增加对同一参数化图 block 的执行次数，是否在 length OOD 上形成稳定收益。

## 3. 文件级变更

### 3.1 参数与配置

新增：

```text
arguments/recurrent_args.py
```

字段：

- `recurrent_depth_train`：训练 recurrence，默认 2；
- `recurrent_depth_eval`：单次运行的测试 recurrence，默认 2；
- `recurrent_depth_sweep`：默认 `[1,2,3,4,5]`；
- `executor_layer_index`：支持负索引，默认 -1 表示中间层自动选择；
- `recurrent_question_types`：默认 `StationShortestCount`；
- `train_hops`：默认 `[1,2]`；
- `validation_hops`：默认 `[1,2]`；
- `test_hops`：默认 `[3,4]`；
- `validation_fraction`：默认 0.2；
- `save_step_hidden_states`：默认 True；
- `max_graphs`：默认 16；
- `max_questions_per_hop`：默认 32；
- `adapter_control`：`correct|shuffled|none`；
- `split_seed`：默认 2026。

修改：

```text
arguments/__init__.py
constants.py
```

- 导出 `RecurrentArguments`；
- 增加 `qwen-0.5b` 与 `qwen-1.5b` 模型别名；
- 不改变现有模型别名。

### 3.2 Recurrent 核心

新增：

```text
grip/recurrent/__init__.py
grip/recurrent/config.py
grip/recurrent/executor.py
grip/recurrent/model.py
grip/recurrent/outputs.py
```

公共接口：

```python
wrap_decoder_layer(model, layer_index, depth)
set_recurrent_depth(model, depth)
capture_recurrent_trace(model, input_ids, attention_mask)
get_recurrent_block(model)
build_recurrent_peft_model(...)
validate_adapter_scope(model, executor_layer_index)
RecurrentPrediction
```

`FixedDepthRecurrentBlock` 的 trace 只保存每轮最后一个有效 token 的 pooled hidden state，立即 `detach().float().cpu()`，避免训练图保留。

### 3.3 CLEGR 路径样本与 split

新增：

```text
grip/tasks/recurrent_tasks/__init__.py
grip/tasks/recurrent_tasks/path_sampler.py
grip/tasks/recurrent_tasks/frontier_builder.py
grip/tasks/recurrent_tasks/split_builder.py
grip/tasks/recurrent_tasks/task_dataset.py
scripts/prepare_recurrent_clegr.py
```

输入优先级：

1. 带 `question_metadata` 的 recurrent processed JSONL；
2. 原 GRIP processed JSONL，但只在能从问题和节点文本唯一恢复端点时接受；
3. 原始 CLEGR `data_list.pt`，由 `prepare_recurrent_clegr.py` 转换并保留 question type/group/subgroup。

每个合格问题输出：

```json
{
  "question_id": "GRAPH_ID:LOCAL_INDEX",
  "question": "...",
  "answer": "3",
  "question_type": "StationShortestCount",
  "source_node": 4,
  "target_node": 17,
  "true_hop": 3,
  "shortest_path": [4, 9, 11, 17],
  "frontiers": [[4], [...], [...], [17]],
  "split": "test"
}
```

拒绝原因必须计数并输出：

- missing question type；
- endpoint extraction ambiguous；
- disconnected endpoints；
- label/path mismatch；
- hop outside requested range；
- insufficient samples in a bucket。

### 3.4 训练与推理

新增：

```text
grip/training/recurrent_trainer.py
scripts/run_recurrent_grip.py
scripts/run_recurrent_pilot.py
```

训练流程：

1. 加载 tokenizer；
2. 构造原始 graph context memory samples；
3. 构造确定性的 1–2 hop QA samples，不调用任务生成 LLM；
4. 加载 plain base LM；
5. 包装 executor block；
6. 只在该 block 注入 graph LoRA；
7. Stage 1 context memory；
8. Stage 2 1–2 hop QA；
9. 对 `K in recurrent_depth_sweep` 逐一闭卷评估；
10. 保存正确 adapter 结果；
11. 可选运行 shuffled/none adapter control。

为保证公平，Original GRIP / More-QA GRIP 使用同一份 deterministic QA split 和相同训练 token 上限。

### 3.5 指标与分析

新增：

```text
evaluation/recurrent_metrics.py
scripts/analyze_recurrent_results.py
```

输出：

- overall EM/F1；
- `(true_hop, K)` accuracy heatmap 数据；
- 每个 hop 的最佳 K；
- K=3/4 相比 K=1 的 CSG；
- seen-hop 与 unseen-hop gap；
- best-K 与 true-hop Spearman；
- correct/shuffled adapter 差值；
- latency 与 peak memory；
- rejected sample statistics。

## 4. 测试 seam 与验收

测试公共 seam，不测试私有实现。

### Seam A：参数解析

文件：`tests/test_recurrent_args.py`

- depth 必须 >=1；
- hop 集合非空且互斥；
- adapter control 枚举合法；
- 默认 sweep 为 1–5。

### Seam B：精确 hop split

文件：`tests/test_hop_split.py`

手写一个 6 节点图：

```text
0--1--2--3
|     
4--5
```

验证：

- 最短距离不是采样路径长度；
- 1–2 hop 进入 train；
- 3 hop 进入 test；
- 错误标签被拒绝；
- 同一问题不跨 split。

### Seam C：递归执行器

文件：`tests/test_recurrent_executor.py`

用可计数 toy block 验证：

- K=1 调用一次；
- K=4 调用四次；
- 四次调用共享同一对象；
- 输出等于逐次函数组合；
- trace 长度等于 K；
- depth 可在推理时切换。

### Seam D：adapter scope

文件：`tests/test_adapter_scope.py`

验证所有 trainable LoRA 参数名都属于 executor layer，任何跨层 trainable 参数都触发失败。

### Seam E：输出 schema

文件：`tests/test_recurrent_output_schema.py`

验证每条 prediction 必含：

- graph/question ID；
- true hop；
- recurrence K；
- adapter ID/control；
- raw/parsed response；
- target/correct；
- latency；
- recurrent trace。

## 5. 实现顺序

1. 写纯 Python path sampler / split builder 和测试；
2. 写 recurrent block 和 toy 测试；
3. 写 adapter scope 校验；
4. 写输出 schema 与指标；
5. 增加小模型别名和参数；
6. 写 recurrent model factory；
7. 写 deterministic QA dataset；
8. 写单图 runner；
9. 写 pilot orchestrator；
10. 跑语法检查和 CPU toy tests；
11. 在有 CUDA 的环境跑 1 graph smoke test；
12. 再运行 8–16 graph、2 小时 Pilot。

## 6. 命令

### 6.1 准备 CLEGR recurrent split

```bash
uv run python scripts/prepare_recurrent_clegr.py \
  --input_file outputs/data/clegr_reasoning/processed_test.json \
  --output_file outputs/data/clegr_reasoning/recurrent_pilot.jsonl \
  --question_types StationShortestCount \
  --train_hops 1 2 \
  --test_hops 3 4 \
  --max_graphs 16 \
  --seed 2026
```

### 6.2 单图 smoke test

```bash
uv run python scripts/run_recurrent_grip.py \
  --input_file outputs/data/clegr_reasoning/recurrent_pilot.jsonl \
  --output_file outputs/recurrent_grip/clegr_reasoning/smoke.jsonl \
  --model_name qwen-0.5b \
  --model_source local \
  --local_files_only True \
  --num_train_epochs 1 \
  --involve_qa_epochs 1 \
  --recurrent_depth_train 2 \
  --recurrent_depth_sweep 1 2 3 4 \
  --max_graphs 1 \
  --gen_max_length 32 \
  --do_sample False
```

### 6.3 两小时 Pilot

```bash
uv run python scripts/run_recurrent_pilot.py \
  --dataset clegr_reasoning \
  --input_file outputs/data/clegr_reasoning/recurrent_pilot.jsonl \
  --model_name qwen-0.5b \
  --max_graphs 16 \
  --wall_time_limit_minutes 120 \
  --hard_stop_minutes 180 \
  --seed 2026
```

### 6.4 分析

```bash
uv run python scripts/analyze_recurrent_results.py \
  --input_file outputs/recurrent_grip/clegr_reasoning/pilot.jsonl \
  --output_dir outputs/recurrent_grip/clegr_reasoning/analysis
```

## 7. Pilot 验收门

实现验收：

- 原 `scripts/run_grip.py` 无行为变化；
- recurrent block 的调用次数测试通过；
- adapter scope 测试通过；
- split 无 hop 泄漏；
- prompt 不含 graph、path 或 true hop；
- 每个 K 使用同一 adapter、同一测试问题；
- 输出足以重新计算全部主要指标。

研究验收仍沿用既定规则，至少满足两项：

1. 3–4 hop 比 Original GRIP 高至少 5 个绝对百分点；
2. K=3/4 明显优于 K=1；
3. 最优 K 与 true hop 正相关；
4. 1–2 hop 下降不超过 2 个百分点；
5. shuffled adapter 明显破坏性能或执行轨迹。

## 8. 回滚边界

- 所有新机制代码位于 `grip/recurrent/`、`grip/tasks/recurrent_tasks/` 和新脚本；
- 原 runner 不导入 recurrent package；
- constants 仅添加模型别名；
- arguments 仅添加新 dataclass 导出；
- 若 Pilot 失败，可删除新增文件并回退两个 additive 修改，不影响 Original GRIP。
