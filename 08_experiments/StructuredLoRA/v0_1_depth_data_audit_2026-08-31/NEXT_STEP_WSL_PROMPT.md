# Prompt for WSL Codex — StructuredLoRA v0.2

```text
仓库是 GRIP-CU。先执行 git pull，并阅读：

1. 08_experiments/StructuredLoRA/README.md
2. 08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/README.md
3. 08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/PROJECT_MEMORY.md
4. artifacts/reports/REPORT.md
5. artifacts/reports/summary.json

然后在下面的新目录独立实现，不修改 13_base_method/grip-exp，也不覆盖 v0.1：

08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/

目标：先验证完美 depth 路由下，有序累计 LoRA 前缀是否优于同总 rank 的普通 LoRA。不要实现 learned router。

固定数据：
- 使用 v0.1 的 exact-hop train/validation/test；
- 所有基线使用完全相同的数据、token 数、batch、optimizer 和训练步数；
- NELL23K 64/32/64 只能作接口 smoke，机制主判断使用 exact-hop。

第一轮模型：Qwen2.5-0.5B-Instruct，WSL2 + RTX 3090，conda 环境 guardenv。

至少实现：
1. monolithic LoRA，总 rank=8；
2. static split 4x2，所有组激活；
3. flat oracle expert 4x2；
4. ordered oracle prefix 4x2；
5. ordered oracle prefix + depth-local credit；
6. permuted depth labels；
7. random group order。

LoRA target modules 与 Original GRIP 保持一致：down_proj, up_proj, gate_proj。

必须记录：
- overall 和 1/2/3/4-hop accuracy；
- macro-hop 与 worst-hop accuracy；
- 每个方法的 trainable params、GPU memory、训练时间；
- group knockout；
- 不同 hop batch 的 LoRA gradient cosine；
- 三个 seeds 的配置入口，smoke 可以先跑一个 seed。

Go gate：
- ordered oracle prefix > equal-rank monolithic；
- ordered prefix > flat expert；
- 3/4-hop 至少 +3 pp；
- 1-hop 下降不超过 1 pp；
- permuted labels 或 random ordering 明显下降。

如果 gate 失败，记录结果并停止，不实现 learned router。完成后运行测试，保存独立 results 目录、README、PATCH_MANIFEST、环境和结果报告，提交并推送远程。
```
