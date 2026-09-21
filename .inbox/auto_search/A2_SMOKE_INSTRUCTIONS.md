# A2 0.5B smoke 执行说明

## A2 定义

A2 在当前 listed InfoNCE 的正例和 9 个问题内负例之外，维护一个跨 batch 的 relation token memory。每个新 batch 的候选打分时，会把 memory 中尚未重复的 relation continuation 作为额外负例；当前 batch 结束后，把本 batch 的负关系加入 memory，最多保留 `MEMORY_SIZE` 条。

## 推荐命令

先确认没有其他任务占用 GPU，然后运行：

```bash
ROOT=/home/ubuntu2/linkc/ltw-lkc/GRIP-CU
HNG=$ROOT/08_experiments/Hard_Negative_Contrastive_GRIP

TMUX_SESSION=a2-smoke-05b-20260921 \
RUN_DIR=$ROOT/.inbox/auto_search/runs/a2_smoke_05b \
SCALE=smoke \
MODEL_NAME=qwen-0.5b \
LAMBDA_CANDIDATE=0.25 \
MEMORY_SIZE=16 \
TEMPERATURE=1.0 \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=2 \
bash $HNG/configs/run_listed_vs_b1.sh
```

启动脚本会自动进入 detached tmux，不要在前台直接运行训练 Python。

## 查看运行状态

```bash
tmux attach -t a2-smoke-05b-20260921
```

或者不进入会话直接查看：

```bash
tmux capture-pane -pt a2-smoke-05b-20260921 -S -80
```

运行日志：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b/run.log
```

## 结果文件

训练完成后重点读取：

```text
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b/comparison.json
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b/b1/summary.json
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b/listed/summary.json
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b/b1/adapter/run_metadata.json
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b/listed/adapter/run_metadata.json
```

快速查看：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/a2_smoke_05b')
print(json.dumps(json.loads((p/'comparison.json').read_text()), indent=2, ensure_ascii=False))
print('\nA2 metadata:')
print(json.dumps(json.loads((p/'listed/adapter/run_metadata.json').read_text()), indent=2, ensure_ascii=False))
PY
```

## 判断标准

A2 进入 7B 前至少应满足：

1. `listed.all.em` 不低于 B1 超过 2 个百分点；
2. 相比无 memory 的低权重 listed baseline，candidate MRR 或 Hits@1 有提升；
3. `candidate_forwards` 增加量和每 step 时间可接受；
4. 无 NaN、CUDA OOM 或 checkpoint 损坏；
5. `memory_size=16` 确实被记录到 `run_metadata.json`。

## 重要说明

当前 smoke launcher 的 Stage 1 使用项目既有 smoke 配置，不是严格的 64 条 context 最小数据。结果只能用于方向筛选，不能作为论文结果。

执行完成后，把以下两个文件内容发回即可：

```text
comparison.json
listed/adapter/run_metadata.json
```
