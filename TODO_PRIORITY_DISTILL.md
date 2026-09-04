# TODO — PriorityDistill-GRIP

更新日期：2026-09-01

当前执行版本：

```text
08_experiments/PriorityDistillGRIP/v0_1_6_joint_trace_answer_2026-09-02/
```

## 当前状态

- [x] 创建独立版本目录，不修改 Original GRIP。
- [x] 构造 train-only gold + 3 distractor candidate pools。
- [x] 注册五个方法与两阶段训练。
- [x] 实现运行时 tokenizer input-token budget matching。
- [x] 实现 Stage-1 prompt 截断统计与强制 Stop gate。
- [x] 实现 graph-free validation/test 与泄漏断言。
- [x] 实现 one-seed / three-seed Go-Stop gate。
- [x] macOS 19 个测试、数据审计、dry-run 与 py_compile 通过。
- [x] WSL RTX 3090 preflight（v0.1 已完成）。
- [x] seed 42 五方法 smoke（v0.1 已完成，PRELIMINARY_STOP）。
- [x] 结果与 token budget 审计（v0.1 已完成）。
- [x] 编写 v0.1.1 Oracle-only Stage-2 延长、分阶段 checkpoint、双条件评估诊断代码。
- [x] 在 guardenv + RTX 3090 运行 v0.1.1 diagnostic。
- [x] 根据 graph-free 曲线决定进入 anti-copy / candidate-selection 协议。
- [x] 创建 v0.1.3 candidate selection + terminal-masked anti-copy + Stage-2 replay 代码。
- [ ] 在 guardenv + RTX 3090 运行 v0.1.3 五个受控协议。
- [ ] 分析 terminal-masked 与 full-path 的差异，再决定是否实现 learned prioritizer。
- [ ] 仅在 `PRELIMINARY_GO` 后运行 42/43/44。
- [ ] 仅在 `GO_LEARNED_PRIORITIZER` 后创建 v0.2。

## WSL 命令（v0.1.1 诊断）

```bash
cd 08_experiments/PriorityDistillGRIP/v0_1_1_oracle_stage2_diagnostic_2026-09-01
source /home/kieran/miniconda3/etc/profile.d/conda.sh
conda activate guardenv
CONDA_ENV=guardenv \
RUN_ID=wsl3090_oracle_stage2_diagnostic_20260901_01 \
bash configs/run_wsl_diagnostic.sh
```


## v0.1.7 公平多 seed follow-up（2026-09-03）

- [x] 完成 v0.1.6 的公平 deployment candidate ranking、constrained greedy 与严格 composition split。
- [x] 添加 `--training-seed`，保证 direct/joint 可以使用完全相同协议做 seed sweep。
- [x] 修正公平评估配置的生成预算为 `max_new_tokens=80`，避免 trace 截断污染。
- [x] 完成 seed 43/44 的 direct vs joint 公平训练；核心 validation/test 结果已落盘。
- [ ] 补齐 seed 44 的 candidate-set constrained/deployment diagnostics（seed 43 已完成）。
- [x] 汇总两 seed 均值/标准差：joint test 33.97% ± 0.91 pp，direct 31.73% ± 2.27 pp；尚不足以通过 three-seed gate。
- [ ] 只有 joint 在公平多 seed 比较中稳定优于 direct，才进入显式组合器/learned router 方案；当前不继续堆 trace LoRA。
