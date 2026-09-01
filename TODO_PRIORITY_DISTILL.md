# TODO — PriorityDistill-GRIP

更新日期：2026-09-01

当前执行版本：

```text
08_experiments/PriorityDistillGRIP/v0_1_oracle_priority_smoke_2026-09-01/
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
- [ ] WSL RTX 3090 preflight。
- [ ] seed 42 五方法 smoke。
- [ ] 结果与 token budget 审计。
- [ ] 仅在 `PRELIMINARY_GO` 后运行 42/43/44。
- [ ] 仅在 `GO_LEARNED_PRIORITIZER` 后创建 v0.2。

## WSL 命令

```bash
git pull
cd 08_experiments/PriorityDistillGRIP/v0_1_oracle_priority_smoke_2026-09-01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-wsl.txt
RUN_ID=wsl3090_priority_distill_smoke_01 bash configs/run_wsl_smoke.sh
```
