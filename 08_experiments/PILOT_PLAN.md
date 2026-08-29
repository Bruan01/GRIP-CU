# NELL23K-First Pilot Plan

更新日期：2026-08-29

## 决策

RecurrentGRIP 首轮 WSL2 RTX 3090 smoke 和两小时 Pilot 改用 NELL23K。CLEGR
不再是运行前置条件，原 CLEGR 方案保存在
[`CLEGR_MECHANISM_PLAN.md`](CLEGR_MECHANISM_PLAN.md)，待 NELL23K 出现递归深度信号后执行。

## 当前代码版本

```text
08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/
```

## 数据协议

```text
train.txt -> train graph + recurrent QA train
valid.txt -> validation
test.txt  -> test
```

默认 smoke 使用 64/32/64 个 train/validation/test 问题；候选关系由训练关系
词表确定性采样。实体对结构距离在 train graph 上通过无向 BFS 重算。不可达样本
保留在整体准确率中，但不进入 hop/K correlation。

## Smoke

```text
Model: Qwen2.5-0.5B
K_train: 2
K_eval: 1,2
Controls: correct adapter / adapter disabled
Hard timeout: 45 minutes
```

命令：

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
bash configs/prepare_nell23k.sh
RUN_ID=wsl3090_nell23k_smoke_20260829_01 bash configs/run_nell23k_smoke_wsl.sh
RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)" bash configs/analyze_nell23k.sh
```

## Smoke Gate

- CUDA、BF16、RTX 3090 验证通过；
- 全部 unittest 通过；
- adapter 只存在于 recurrent executor layer；
- K=1/2 都产生预测；
- correct/none 都产生结果；
- adapter 保存、重载后模型仍在 CUDA；
- 无 OOM、NaN、CUDA error 和 CPU 静默回退；
- 运行目录、环境、输入统计、日志和预测完整保存。

## Two-Hour Pilot

Smoke 通过后：

```bash
MAX_TRAIN_QUESTIONS=512 \
MAX_VALIDATION_QUESTIONS=128 \
MAX_TEST_QUESTIONS=512 \
  bash configs/prepare_nell23k.sh

RUN_ID=wsl3090_nell23k_pilot_20260829_01 \
  bash configs/run_nell23k_pilot_wsl.sh
```

Pilot 比较 K=1/2/3/4，并报告整体 accuracy、correct-vs-none、结构距离分桶、
延迟和峰值显存。

## 解释边界

NELL23K 的 train-graph shortest path 是结构相关性诊断，不是受控的推理 hop 标签。
NELL23K 可以支持“RecurrentGRIP 在原始 GRIP 数据集上有效”和“K 对不同结构距离
产生差异”，但严格的 K↔hop 因果主张仍由后续 CLEGR 机制实验确认。
