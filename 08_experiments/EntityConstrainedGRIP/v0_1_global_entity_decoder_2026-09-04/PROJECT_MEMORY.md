# Project Memory — EntityConstrained-GRIP v0.1

## Frozen decisions (2026-09-04)

1. 最终论文证据必须回到官方 GRIP Qwen2.5-7B 配置；0.5B 只做机制 smoke。
2. `13_base_method/grip-exp/` 只读。
3. 实体词表仅来自 NELL23K train graph，共 20,789 entities；所有 query 共用同一词表。
4. 旧 64.74% 4-way rank-1 使用 gold intermediate trace，不作为严格 graph-free 证据。
5. Priority fair sweep 的 validation-selected direct seed43 epoch5、seed44 epoch4 为 E03 primary；More-QA seed42 为 reference；joint seed43/44 为 secondary。
6. 首轮只运行 validation D0/D1；D0 复用冻结 prediction artifact，禁止重复生成造成基线漂移。
7. D1 gate 关注 semantic EM 与 error transitions；valid-rate 增益不能代替 EM。
8. D2 仅在 `PRELIMINARY_GO_D2` 后运行 validation；D3 始终 diagnostic-only。
9. test 在 Phase-A 机制 gate 与官方 Phase-B 协议冻结前保持关闭。
10. D4 learned/discriminative head 不在本版本实现。

## Current status

```text
READY_TO_PUSH_VALIDATION_ONLY_E03
```

Priority 完整审计后，E03 已按 validation-first 证伪协议完成代码修订：

- primary 改为 direct seed43/44；
- More-QA seed42 改为 reference evidence；
- joint seed43/44 降为 secondary；
- 首轮 decoder 固定为 D0 artifact reuse + D1 global trie；
- Phase-A runner 不打开 test；
- 增加 D0→D1 semantic error-transition decomposition；
- gate 改为 direct 平均 canonical EM `≥ +2 pp`、每 checkpoint raw/canonical 非负、novel-composition 平均下降不超过 `1 pp`；
- D2 shell 入口被 `PRELIMINARY_GO_D2` artifact 硬门禁；
- D3 改为显式 diagnostic mode。

## Static re-audit result

2026-09-04 已完成完整 macOS 静态复审：

1. 40 个单元测试全部通过；
2. 三个首轮 D0 artifact 的 SHA256、152 task_id 完整对齐与 answer 一致；
3. config、split、prompt leakage、checkpoint registry 全部通过；
4. shell/Python 语法检查通过，旧协议残留搜索为空；
5. Phase-A runner 只加载 train/validation，不打开 test；
6. `git diff -- 13_base_method/grip-exp` 为空，无关 untracked 文件未修改或删除。

## Immediate next action

等待用户明确要求后，仅提交并推送 E03、根 README 与根 TODO 的相关变更；推送后在原 WSL2 + RTX 3090 运行 `RUN_MODE=phase_a`。当前未 commit、未 push、未运行 WSL GPU。
