# Source Baseline

## Immediate Parent

- Parent snapshot: `../v1_1_nell23k_first_2026-08-29`
- Parent experiment ID: `RG-E2-v1.1-nell23k-first-2026-08-29`
- Fork date: `2026-08-29`
- Copy script: `../create_version.sh`
- Excluded during copy: old `results/`, old `logs/`, `.venv/`, `model_cache/`, `outputs/`, bytecode.

## Git Provenance at Fork

- Branch: `wsl/nell23k-smoke-20260829`
- Commit: `ca12e0f9952a0b08da7eab028b003532958b848b`
- Commit short: `ca12e0f`
- Commit subject: `fix: complete WSL NELL23K RecurrentGRIP smoke`

## Original GRIP Provenance

- Source: `13_base_method/grip-exp`
- Git commit: `2835b440bfd2c4de36f0380ae19bc1c22e6cb459`
- Original GRIP mutation: none.

## Version Delta

v1.1.1 保留 v1.1 的 NELL23K 数据适配、WSL 修复和 fixed-depth executor，只增加
storage × execution diagnostic：relation-balanced edge context、context manifest、
`K_train × K_eval` 交叉运行和输出质量审计。旧版本不被覆盖。
