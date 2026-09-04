# RecurrentGRIP 涨点实验 — 2×3090 服务器 Runbook

更新：2026-09-01

> **推荐：直接用一键脚本 [`run_recurrentgrip_server.sh`](../run_recurrentgrip_server.sh)**
> `bash run_recurrentgrip_server.sh` 会安全地跑完 env→data→smoke→pilot→analyze。
> 它已内置：单卡固定、缓存隔离到项目内、超时兜底。下面各节是它的详细对照说明。

目标：在 NELL23K 上跑出"测试时递归深度 K → 准确率"的涨点信号。
对照关系：baseline GRIP（单次前向） vs RecurrentGRIP（K=1/2/3/4）。

## 0. 前提（本地已完成）

- NELL23K 数据已在本地确定性生成（seed=2026），可直接复制到服务器，或在服务器上重新生成（结果一致）：
  - smoke 输入：`grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json`（64/32/64）
  - pilot 输入：`grip-exp/outputs/data/nell23k/recurrent_relation_prediction_pilot.json`（512/128/512）
- 数据结构已验证：训练问题 structural distance 全为 1，测试分布 1~12 + unreachable（长度外推的干净轴）。
- 数据单元测试 2/2 通过；baseline 代码在固定 commit `2835b440` 且 clean。

传输建议：把整个 `v1_1_nell23k_first_2026-08-29/` 快照 `scp -r` 到服务器，避免重新生成。

## 1. 上服务器，确认 GPU

```bash
nvidia-smi
# 应看到 2 × RTX 3090，各 24GB
```

`verify_wsl3090.py` 只检查 cuda:0，第二张卡需手动用 `nvidia-smi` 确认。

## 2. 环境搭建 + 测试

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
bash configs/setup_wsl3090.sh     # uv venv + torch 2.7.1 + transformers 4.56.1 + peft 0.17.1
bash configs/test_wsl3090.sh      # 全部 unittest（需 torch/transformers）
.venv/bin/python configs/verify_wsl3090.py   # 验证 cuda:0 是 3090、BF16 可用
```

注意：`setup_wsl3090.sh` 名字带 wsl3090，但逻辑是通用的 torch+PEFT 环境，不依赖 WSL。

## 3. 复现 baseline GRIP（Qwen2.5-7B，参考分数）

论文在 NELL23K 上是 87.74%。用 baseline 子模块复现：

```bash
cd 13_base_method/grip-exp/grip-exp
bash scripts/download_model.py --model_name qwen-7b --model_cache_dir model_cache  # 或用已有 ModelScope 缓存
bash scripts/process_raw_data.py --datasets nell23k --output_dir outputs/data --seed 2026
bash scripts/run_nell23k_full.sh --dry-run   # 先看命令
bash scripts/run_nell23k_full.sh             # 正式跑（单卡，--num_process 1）
```

跑完看日志末尾的 em/f1/hit，记录 baseline 分数。

## 4. RecurrentGRIP smoke（0.5B，K=1/2，验证链路）

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
RUN_ID=smoke_20260901_01 bash configs/run_nell23k_smoke_wsl.sh
```

验收：Qwen2.5-0.5B 在 CUDA 上训练/评估、K=1 和 K=2 都出预测、correct/none adapter 都出结果、无 OOM/NaN/CPU 回退。

## 5. RecurrentGRIP pilot（0.5B，K=1/2/3/4，正式信号）

先准备 pilot 输入（若本地没复制过来，在服务器重新生成）：

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
MAX_TRAIN_QUESTIONS=512 MAX_VALIDATION_QUESTIONS=128 MAX_TEST_QUESTIONS=512 SEED=2026 \
  NELL_OUTPUT="$PWD/grip-exp/outputs/data/nell23k/recurrent_relation_prediction_pilot.json" \
  bash configs/prepare_nell23k.sh
RUN_ID=pilot_20260901_01 bash configs/run_nell23k_pilot_wsl.sh
```

pilot 默认 K sweep = 1 2 3 4，180 分钟硬超时。

## 6. 分析：K 是否带来涨点（核心判断）

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
RUN_DIR="$(cat results/LAST_NELL23K_PILOT_RUN.txt)" bash configs/analyze_nell23k.sh
```

重点看：
- `analysis/k_adapter_accuracy.csv`：K=1/2/3/4 的准确率，是否随 K 上升；
- `analysis/hop_k_accuracy.csv`：按 structural distance 分桶，**高 distance 的样本是否 K 越大涨得越多**。

如果"高 distance + 大 K → 更高准确率"这条曲线成立，涨点故事就成立了。

## 7. 2×3090 怎么用

- 0.5B 单卡就跑得动；两张卡的实用价值：
  - 用 `CUDA_VISIBLE_DEVICES=0` / `=1` 分别并行跑 baseline 和 recurrent pilot；
  - 或并行跑多个 RUN_ID / 种子。
- 7B 主结果：baseline 脚本 `--num_process 1` 单卡可跑；若要 7B 的 RecurrentGRIP（模型并行），后续需要给 recurrent 脚本加 accelerate 多卡支持，这一步先不做。

## 8. 关键风险提醒

- **涨点幅度是未知数**：必须靠第 6 步的曲线说话，不能预设"涨 2~3 个点"。
- **模型规模要对齐**：baseline 是 7B，recurrent smoke/pilot 是 0.5B；最终要补 7B 的 RecurrentGRIP 才能公平对照。
- **compute-matched baseline**：RepeatedGrip 最终要对 GRIP 加"同样测试时计算"（重复采样/beam），证明涨点不是单纯多算。
