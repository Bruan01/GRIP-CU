# Source Baseline

## Immediate Parent

- Parent snapshot: `../v1_fixed_depth_2026-08-28`
- Parent experiment ID: `RG-E1-v1-fixed-depth-2026-08-28`
- Fork date: `2026-08-29`
- Copy script: `../create_version.sh`
- Excluded during copy: old `results/`, old `logs/`, `.venv/`, `model_cache/`, `outputs/`, bytecode.

## Original GRIP Provenance

- Source: `13_base_method/grip-exp`
- Git commit: `2835b440bfd2c4de36f0380ae19bc1c22e6cb459`
- Commit short: `2835b44`
- Original GRIP mutation: none.

## Version Delta

v1.1 保留 v1 的 fixed-depth shared executor，只新增 NELL23K-first 数据适配、
可空结构距离输出、NELL23K WSL 运行入口和对应分析。CLEGR-first v1 不被覆盖。
