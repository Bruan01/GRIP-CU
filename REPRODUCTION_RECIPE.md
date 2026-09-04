# GRIP 论文复现配方（已确认，可直接执行）

更新日期：2026-09-03
来源：`13_base_method/grip-exp/grip-exp/shells/qwen_grip_inf/nell23k_single3090.sh`
以及 `scripts/01_generate_nell23k_tasks.sh / 02_train_nell23k_lora.sh / 03_infer_eval_nell23k.sh`

---

## 0. 论文配方 vs 失败 pilot（已逐项确认）

| 配置项 | 论文（NELL23K，EM=87.74%） | v1.1 pilot（4.5%） |
|---|---|---|
| 模型 | Qwen2.5-7B-Instruct | Qwen2.5-0.5B-Instruct |
| LoRA 模块 | **down_proj up_proj gate_proj（MLP）** | q_proj k_proj v_proj（注意力）|
| LoRA 层范围 | 全层 | 仅第 12 层 |
| LoRA r/alpha | 4 / 8 | 4 / 32 |
| 任务量 | 8000 context + 2000 reason + 6000 summary | 512 QA |
| Stage 1 | 1 epoch，早停 loss 0.15 | 1 epoch（loss 停 5~6）|
| Stage 2 | **10 epoch** | 1 epoch |
| 有效 batch | 512 | 8 |
| 学习率 | **1e-3** | 2e-4 |

pilot 在**每一个关键轴**上都偏离了论文配方。失败是配置错误，不是方法失效。

---

## 1. 前置条件现状（已检查）

| 依赖 | 状态 |
|---|---|
| 13_base_method 的 venv | ❌ 无 |
| Qwen2.5-7B 模型缓存 | ❌ 无（需下载 ~15GB）|
| NELL23K processed 数据 | ❌ 无（需 process_raw_data）|
| artifacts（任务/训练/adapter）| ❌ 无 |
| RecurrentGRIP 快照里的 0.5B + venv + 原始数据 | ✅ 有 |

结论：完整复现需要从下载 7B 模型开始，是一次**较重的投入**。

---

## 2. 论文三段式流水线（完整复现）

### 阶段 0：下载模型 + 处理数据

```bash
cd 13_base_method/grip-exp/grip-exp
# 下载 Qwen2.5-7B（ModelScope 或 HF），落到 model_cache/Qwen--Qwen2.5-7B-Instruct
python scripts/download_model.py --model_name qwen-7b --model_cache_dir model_cache

# 处理 NELL23K 原始数据 -> outputs/data/nell23k/processed_{test,val}.json
python scripts/process_raw_data.py --datasets nell23k --output_dir outputs/data --seed 2026
```

### 阶段 1：生成训练任务（16k 条，需 LLM 生成，较慢）

```bash
bash scripts/01_generate_nell23k_tasks.sh
# 内部等价命令：
#   generate_nell23k_tasks.py --num-context-qa 8000 --num-reason-qa 2000 \
#     --num-summarization 6000 --task-generator-model-name qwen-7b ...
```

### 阶段 2：训练 LoRA（7B，MLP，10 epoch）

```bash
bash scripts/02_train_nell23k_lora.sh
# 关键参数：
#   --model-name qwen-7b --lora-r 4 --lora-alpha 8 \
#   --target-modules down_proj up_proj gate_proj \
#   --num-train-epochs 1 --involve-qa-epochs 10 \
#   --s1-stop-loss-threshold 0.15 --s2-stop-loss-threshold 0.15 \
#   --per-device-train-batch-size 1 --gradient-accumulation-steps 512 \
#   --learning-rate 1e-3 --weight-decay 1e-4 --max-grad-norm 1.0
```

### 阶段 3：推理评测（EM / F1 / Hit）

```bash
bash scripts/03_infer_eval_nell23k.sh
```

单条命令一键跑：`bash shells/qwen_grip_inf/nell23k_single3090.sh`（内部走 run_grip.py）。

---

## 3. 建议：先做便宜的 0.5B 验证，再上 7B

完整 7B 复现投入大（下载 15GB + 16k 任务生成 + 10 epoch 训练）。建议先用**已有的 0.5B + venv + 数据**做一次快速验证，回答「图能不能写进参数」这个前提：

### 快速验证要改的三处（相对失败 pilot）

1. **LoRA 挂到 MLP**：`target_modules = [down_proj, up_proj, gate_proj]`（这是失败的头号嫌疑）。
2. **去掉单层限制**：不要 `layers_to_transform=[12]`，LoRA 挂全层。
3. **训练到位**：Stage 1 训到 loss ≤ 0.15（不是固定 1 epoch）；Stage 2 多 epoch；batch 512；lr 1e-3。

### 判定

- 0.5B 上若 `correct adapter EM > base model`，则前提成立，值得投入 7B 完整复现；
- 若 0.5B 上改完这三处仍 `adapter ≤ base`，则先查是不是 0.5B 容量不够，再决定是否上 7B。

---

## 4. 立即执行清单

- [ ] （快速验证）在 RecurrentGRIP 快照内，把训练配置改成「MLP LoRA + 全层 + 足够 epoch/batch/lr」，跑 0.5B，看 correct vs none EM。
- [ ] 若快速验证通过 → 下载 7B 模型，走上面三段式完整复现。
- [ ] 记录 Stage 1 loss 是否收敛、correct vs none EM，写入 `09_results_analysis`。
