# RecurrentGRIP Research Status

更新日期：2026-08-29

## 当前主线

**RecurrentGRIP: Compiling Graphs into Executable Parametric Programs for Closed-Book Graph Reasoning**

核心问题不是继续优化 GRIP 的训练配方，而是区分：

- 图是否成功写入参数（storage）；
- 参数化图是否能被模型逐步调用（retrieval）；
- 被调用的图知识是否能形成可重复执行的多步计算（execution）。

## 当前主张

RecurrentGRIP 将模型拆分为共享问题编码器、共享递归图执行器、图专属 LoRA
参数记忆和答案解码器。图专属 LoRA 定义“执行哪张图”，共享递归模块学习“如何
重复执行一步”；推理时不提供原图。

## 当前实验决策

首轮 WSL2 RTX 3090 验证改为 **NELL23K-first**：

1. NELL23K 已随 GRIP 代码提供，不需要先下载 CLEGR；
2. 先验证 recurrent executor、adapter scope、CUDA 重载和 K sweep；
3. 先回答 RecurrentGRIP 是否在 Original GRIP 数据集上产生收益；
4. NELL23K 出现可信 recurrence 信号后，再运行 CLEGR 的严格 K↔hop 机制实验。

## 当前阶段

- [x] 研究问题与 RecurrentGRIP 机制确定
- [x] 初步 novelty check
- [x] Fixed-Depth v1 实现
- [x] CLEGR schema/hop split 实现并冻结在 v1
- [x] NELL23K-first v1.1 独立快照
- [x] NELL23K train/valid/test 数据适配与结构距离分析
- [x] NELL23K WSL smoke/pilot 脚本
- [ ] WSL2 RTX 3090 ML 集成测试
- [ ] NELL23K Qwen2.5-0.5B smoke
- [ ] Original GRIP 公平对照
- [ ] NELL23K 两小时 Pilot
- [ ] CLEGR 严格机制确认
- [ ] 多数据集实验
- [ ] 机制分析与论文写作

## 当前代码版本

- Original GRIP：`13_base_method/grip-exp/`，保持 clean；
- CLEGR-first v1：`08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/`，冻结保留；
- 当前 NELL23K-first v1.1：`08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/`；
- macOS：代码、版本管理、静态检查和纯 Python 数据测试；
- WSL2 + RTX 3090：Torch/Transformers/PEFT 集成、训练、推理和正式结果。

## 立即执行顺序

1. WSL 拉取最新仓库与 Original GRIP submodule；
2. 进入 NELL23K-first v1.1；
3. 安装环境并运行全部 unittest；
4. 从仓库自带 NELL23K 文件生成 64/32/64 smoke 输入；
5. 运行 Qwen2.5-0.5B、K=1/2、correct/none smoke；
6. 分析输出并记录 CUDA、显存、延迟和准确率；
7. Smoke 通过后才准备 512/128/512 两小时 Pilot；
8. NELL23K 出现机制信号后再执行 CLEGR。

## 核心入口

- `TODO_WSL3090.md`
- `08_experiments/PILOT_PLAN.md`
- `08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/README.md`
- `08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/design/NELL23K_FIRST_PLAN.md`
- `08_experiments/CLEGR_MECHANISM_PLAN.md`
