# GRIP-CU 磁盘清理清单（管理员用）

生成日期：2026-10-10
仓库根目录：`/home/ubuntu2/linkc/ltw-lkc/GRIP-CU`

## 磁盘现状（清理前）

```
/dev/nvme0n1p2   937G   832G   58G   94%   /
```

根分区已用 94%，仅剩 58 GB。本清单按「优先级 / 风险」分级，**每个条目都给出了绝对路径和可直接执行的删除命令**。

预计可释放总量：**约 195 GB**。

> 执行顺序建议：先做 P1（零风险），再做 P2–P6。P4/P5 是大头，但删前请对照第 8 节「绝对不要删」。

---

## 汇总表

| 优先级 | 类别 | 路径数 | 可释放 | 风险 |
|---|---|---|---|---|
| P1 | 日志 / 缓存 / 修复前备份 | 11 | ~90 MB | 无 |
| P2 | 已放弃的 DPO 分支 | 15 | ~27 GB | 无 |
| P3 | 草稿区 `.inbox` / `.inbox2` | 2 棵目录树 | ~8.5 GB | 低（见说明） |
| P4 | CLEGR 预处理数据（含重复副本） | 4 | ~87.5 GB | 低 |
| P5 | 已归档旧 run 的原始目录 | 33 | ~68 GB | 低（快照已入库） |
| P6 | RecurrentGRIP 旧版冒烟产物 | 11 | ~4 GB | 低 |
| — | **合计** | | **~195 GB** | |

---

## P1 日志 / 缓存 / 修复前备份（~90 MB，零风险）

全部为一次性产物，无任何脚本引用。

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/all_console.log                       (4.4 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/console.log                           (8 KB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/env_setup.log                         (8 KB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/server_run.log                        (8 KB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/smoke_console.log                     (560 KB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json.before_json_repair   (85 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.pytest_cache/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/.pytest_cache/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/.pytest_cache/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/__pycache__/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox2/exp1/scripts/__pycache__/
```

⚠️ 注意：`grip_nell23k_tasks.json`（88 MB，**没有** `.before_json_repair` 后缀的那个）是训练输入，**不要删**。

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU
rm -f all_console.log console.log env_setup.log server_run.log smoke_console.log
rm -f 08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json.before_json_repair
rm -rf .pytest_cache 08_experiments/.pytest_cache 08_experiments/Hard_Negative_Contrastive_GRIP/.pytest_cache
rm -rf .inbox/__pycache__ .inbox2/exp1/scripts/__pycache__
```

---

## P2 已放弃的 DPO 分支（~27 GB）

DPO 路线已否掉，这些 run 只被 `results/LAST_DPO_RUN.txt` 引用，无其他依赖。

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-full-20260918_235747/    (8.3 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-full-20260918_235920/    (8.3 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-full-20260919_000258/    (8.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152929/   (2.1 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152230/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152307/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152406/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152437/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152539/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152606/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/dpo-smoke-20260918_152654/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/.inbox/dpo_smoke.jsonl
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/.inbox/dpo_full.jsonl
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/LAST_DPO_RUN.txt
```

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
rm -rf results/runs/dpo-full-20260918_235747 \
       results/runs/dpo-full-20260918_235920 \
       results/runs/dpo-full-20260919_000258 \
       results/runs/dpo-smoke-20260918_152*
rm -rf .inbox
rm -f results/LAST_DPO_RUN.txt
```

（DPO 配套脚本 `.inbox/train_dpo_smoke.py`、`.inbox/run_dpo_smoke.sh`、`.inbox/build_dpo_data.py` 随上面 `rm -rf .inbox` 一并删除。）

---

## P3 草稿区 `.inbox` / `.inbox2`（~8.5 GB）

两个自成一体的探索工作区，仓库内**无任何脚本引用**（已 grep 确认）。

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/auto_search/runs/     (7.5 GB：smoke_baseline_0p5b + a2_smoke_05b)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox2/                     (1.1 GB："阶段干扰与适配器融合"实验)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/.venv/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/sample_2000.jsonl
```

⚠️ **删除前必须确认**：以下两个文件有**未提交的本地改动**（`git status` 显示 ` M`）。如改动仍有价值，请先 `git add` + `git commit`，再删目录：

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/mlp_probe.py
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.inbox/train_dpo_smoke.py
```

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU
git status --short .inbox/mlp_probe.py .inbox/train_dpo_smoke.py   # 先确认是否已提交
rm -rf .inbox .inbox2
```

> 若只想省空间、想保留脚本：可只删 `.inbox/auto_search/runs/`（7.5 GB）和 `.inbox2/exp1/runs/`、`.inbox2/exp1/models/`。

---

## P4 CLEGR 预处理数据（~87.5 GB）

路径位于 RecurrentGRIP 旧版本目录。CLEGR 仅用于 `RECOVERY_PLAN.md` 里 Phase 2 的 H1 机制实验，该方向当前处于「未检验」状态，主线（listed vs B1）不涉及。

```
# 54 GB：已解压的 HF 数据集
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/outputs/data/clegr/datasets_for_hf/

# 33 GB：上面 datasets_for_hf 的压缩包，内容重复，二者最多留一份
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/outputs/data/clegr/data_for_hf.tar.gz

# 440 MB + 33 MB：pilot 的 tar 包，内容已在 results/runs/ 下解压存在
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/pilot_20260902_031653_nell23k_qwen05b_pilot.tar.gz
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/pilot_20260902_031653_nell23k_qwen05b_pilot_results.tar.gz
```

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
rm -rf grip-exp/outputs/data/clegr/datasets_for_hf
rm -f  grip-exp/outputs/data/clegr/data_for_hf.tar.gz
rm -f  pilot_20260902_031653_nell23k_qwen05b_pilot.tar.gz pilot_20260902_031653_nell23k_qwen05b_pilot.tar.gz.sha256
rm -f  pilot_20260902_031653_nell23k_qwen05b_pilot_results.tar.gz pilot_20260902_031653_nell23k_qwen05b_pilot_results.tar.gz.sha256
```

> `grip-exp/outputs/data/` 下的 `nell23k/`、`clegr_facts/`、`clegr_facts_large/`、`clegr_reasoning/`（合计约 80 MB）可保留。

---

## P5 已归档旧 run 的原始目录（~68 GB）

这些 run 的结果已由 `results/archive/` 落盘留档（含 `summary.json`、`predictions_*.jsonl`、`comparison.json`、`SNAPSHOT.md`；**不含** adapter 权重）。`archive/INDEX.md` 中有对应记录。原始目录只剩可重建的 checkpoint。

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260926_shared_pool_sampler_smoke/            (13 GB，smoke)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260922_heldout_rollout_smoke_v2/             (8.6 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260922_heldout_rollout_smoke/                (4.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_qwen7b_embed_negatives/               (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260921_score_hard_4hard5uniform/             (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260929_shared_pool_soft_mix_full/            (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260929_shared_pool_calibrated_full/          (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260926_shared_pool_top_k_hard_full/          (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260926_shared_pool_random_k_full/            (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260926_shared_pool_coverage_adaptive_k_full/ (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_soft_mix_full/   (6.4 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260920_qwen7b_whiten_negatives/              (2.2 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260918_qwen7b_train_graph_negatives/         (2.1 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260909_listed_vs_b1_smoke_listed_vs_b1_smoke/ (1.6 GB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260915_140500_qwen7b_full_decode/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_qwen7b_full_listed_embed_negatives/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_score_limit100/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260922_rollout_hard_s1_mining/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_compare20/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_compare20_b8/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_qwen7b_pilot_listed_only/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_030648_qwen7b_pilot_listed_only/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_030854_qwen7b_pilot_listed_only/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_031929_qwen7b_pilot_listed_only/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260919_032013_qwen7b_pilot_listed_only/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260915_101030_qwen7b_pilot_decode/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260922_rollout_hard_mining/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260922_rollout_hard_train/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_benchmark/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260917_qwen7b_train_graph_negatives/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260913_qwen7b_smoke_listed_vs_b1_qwen-7b_smoke/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260921_score_hard_smoke/
```

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs
rm -rf 20260926_shared_pool_sampler_smoke \
       20260922_heldout_rollout_smoke_v2 20260922_heldout_rollout_smoke \
       20260919_qwen7b_embed_negatives 20260919_qwen7b_full_listed_embed_negatives \
       20260921_score_hard_4hard5uniform 20260921_score_hard_smoke \
       20260929_shared_pool_soft_mix_full 20260929_shared_pool_calibrated_full \
       20260926_shared_pool_top_k_hard_full 20260926_shared_pool_random_k_full \
       20260926_shared_pool_coverage_adaptive_k_full \
       20261005_shared_pool_truncated_soft_mix_full \
       20260920_qwen7b_whiten_negatives 20260918_qwen7b_train_graph_negatives \
       20260917_qwen7b_train_graph_negatives \
       20260909_listed_vs_b1_smoke_listed_vs_b1_smoke \
       20260913_qwen7b_smoke_listed_vs_b1_qwen-7b_smoke \
       20260915_101030_qwen7b_pilot_decode 20260915_140500_qwen7b_full_decode \
       20260919_qwen7b_pilot_listed_only 20260919_0306*_qwen7b_pilot_listed_only \
       20260919_0308*_qwen7b_pilot_listed_only 20260919_0319*_qwen7b_pilot_listed_only \
       20260919_0320*_qwen7b_pilot_listed_only \
       20260922_rollout_hard_mining 20260922_rollout_hard_train 20260922_rollout_hard_s1_mining \
       20260923_offline_confusion_benchmark 20260923_offline_confusion_compare20 \
       20260923_offline_confusion_compare20_b8 20260923_offline_score_limit100
```

### P5b 可选：只删 checkpoint，保留 run 与 adapter（再省 ~40 GB）

每个 6.4 GB 的 run 中，`trainer_listed/checkpoint-230` 与 `checkpoint-240` 各占 2.2 GB（共 4.4 GB），真正产物是 `listed/adapter/`（2.1 GB）。若某 run 想保留 adapter，可只删 checkpoint：

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs
# 对仍需保留 adapter 的 run 执行（示例：最新两个）
for r in 20261007_accumfix_random_k_full 20261007_accumfix_calibrated_full; do
  rm -rf "$r"/trainer_listed/checkpoint-*
done
```

---

## P6 RecurrentGRIP 旧版冒烟产物（~4 GB）

v1.1 目录下全是 0.5B 的失败/重试冒烟。唯一有留档价值的是 `quick01_storage_quick/`（`RECOVERY_PLAN.md` 中判定 PASS 的那次）。

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/pilot_20260902_031653_nell23k_qwen05b_pilot/     (682 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/20260904_103219_recurrent_v2/                   (545 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/20260902_120720_nell23k_qwen05b_smoke/          (542 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/smoke_20260902_030616_nell23k_qwen05b_smoke/    (542 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/smoke_20260902_1056_retry_nell23k_qwen05b_smoke/ (542 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/smoke_mlp_20260911_115553_storage_quick/      (542 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/recurrent_v2_01_recurrent_v2/                   (540 MB)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/smoke_20260902_025720_nell23k_qwen05b_smoke/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/smoke_20260901_152613_nell23k_qwen05b_smoke/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/smoke_mlp_20260911_115507_storage_quick/

# 整个目录被 v1_1 取代，可删
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/   (14 MB)
```

```bash
cd /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs
# 保留 quick01_storage_quick
for d in */; do [ "$d" != "quick01_storage_quick/" ] && rm -rf "$d"; done

rm -rf /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28
```

---

## P7 疑似过期文档（**需人工判断，不建议自动删**）

最后更新时间停留在 8 月底 / 9 月初，内容已与当前主线（listed vs B1，最好 90.62%）脱节。建议**归档或改写**，而非直接删除。

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/RESEARCH_STATUS.md        (2026-08-29)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/RESEARCH_CHECKLIST.md     (2026-08-28)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/TODO_WSL3090.md           (2026-08-29)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/RECOVERY_PLAN.md          (2026-09-04，正文停在"下一步下载 7B")
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/REPRODUCTION_RECIPE.md    (2026-09-04)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/SERVER_2x3090_RUNBOOK.md  (2026-09-04)

# 早期脚手架，仅含 .gitkeep 的空目录
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/01_main_results/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/02_ablation/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/03_hyperparameter/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/04_efficiency/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/05_robustness/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/06_error_analysis/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/07_significance/
```

---

## 8. 绝对不要删（当前训练脚本的活跃依赖）

以下路径被 `configs/*.sh` 硬编码为默认值，删除会导致训练/评测脚本直接报错。

### 8.1 代码与运行环境

```
# 所有 HN 脚本的 CODE_DIR 与 .venv/bin/python 来源（20+ 个 config 指向它）
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/
  ├── grip/  scripts/  shells/  evaluation/  arguments/  tests/  utils.py   ← 必须保留
  ├── .venv/                                                              ← 必须保留
  ├── model_cache/Qwen--Qwen2.5-7B-Instruct/   (15 GB，删了要重下)          ← 必须保留
  └── outputs/data/nell23k/  clegr_facts/  clegr_reasoning/  (约 80 MB)    ← 保留
```

### 8.2 活跃 run（含被引用的 adapter / 数据）

```
# s1_adapter（2.1 GB，所有训练脚本的 Stage-1 默认起点）
# b1/（2.1 GB，冻结的 B1 基线 86.46%）
# listed/（对照组）
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/

# 被 config 与 LAST_CONFUSION_DB.txt 引用
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_confusion_db_frozen_b1/

# config 引用十余次
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_full/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_train_filter/

# 最新一轮（90.62%），LAST_SHARED_POOL_SAMPLER_TRAIN_RUN 指向后者，留着续训
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261007_accumfix_random_k_full/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261007_accumfix_calibrated_full/

# config 有引用（体积小）
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260926_shared_pool_samplers/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_mixture_samplers/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260921_031844_score_hard_mining/
```

### 8.3 训练输入与留档

```
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json   (88 MB，训练输入)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/data/nell23k/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/archive/           (唯一的指标留档，已入库)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/13_base_method/grip-exp/     (git submodule)
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.git/
/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/.cursor/rules/
```

---

## 9. 执行后校验

```bash
df -h /home/ubuntu2
du -sh /home/ubuntu2/linkc/ltw-lkc/GRIP-CU
du -sh /home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/*/
```

## 10. 未提交文件的提醒

`git status` 中还有以下**未提交、且未被 ignore** 的文件，它们不是垃圾，删除前请确认：

```
08_experiments/Hard_Negative_Contrastive_GRIP/GRIP_NELL23K_失败模式报告.md
08_experiments/Hard_Negative_Contrastive_GRIP/GRIP_性能与参考基线调研报告.md
08_experiments/Hard_Negative_Contrastive_GRIP/data/nell23k/recurrent_relation_prediction_full.json
08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json
08_experiments/Hard_Negative_Contrastive_GRIP/.inbox/
08_experiments/Hard_Negative_Contrastive_GRIP/results/archive/20260929_shared_pool_mixture_samplers/
```
