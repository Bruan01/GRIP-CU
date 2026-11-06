# 后台监控与自动接续

## 2026-11-06 完成：FB固定配置组件消融

两组新实验均完成32步训练及各一次10,000题完整直接解码测试；四行表于北京时间11:56生成，守护正常退出，未继续其他训练。

| 候选来源 | 正确数 / 10,000 | 直接解码EM |
|---|---:|---:|
| Random-9 | 8,648 | 86.48% |
| Profile-only-9 | 8,489 | 84.89% |
| Violation-only-9 | 8,709 | 87.09% |
| Full SC-CL-9 | 8,990 | 89.90% |

Full−Violation为+2.81个百分点（净281题，纠错582、退化301）；Full−Profile为+5.01个百分点（净501题，纠错755、退化254）。Violation−Random为+0.61个百分点，Profile−Random为−1.59个百分点；结构画像单独使用的负结果必须保留。

结论限于seed2026及此前针对Full验证选定的固定配置：联合使用取得本表最高观测性能，结构项在保留B1违例反馈时有条件性增量；不能声称结构单独有效、普适协同、统计显著或跨种子稳定。全部方法共享464条train-pair QA修复及历史测试暴露条件。两组组件不做新全量验证、HPO或重复种子，实际32步并保持120步调度。

原始预测、四检查点/optimizer/scheduler/RNG、冻结代码、bank及日志均保留在Linux运行目录；借用输入所在的Random运行目录也必须保留。完成复核重算40,000条模型-题目正确标记、核对10,000行四方法配对表及424个归档文件SHA。论文CSV、LaTeX、结果及完成审计见 `11_paper/artifacts/fb_component_ablation_seed2026/`；两个单信号组另有独立产物目录。下一步转入结果写作，本轮停止训练。

以下保留原预注册计划与启动说明，历史“待运行”等措辞不代表当前状态。

当前队列只包含一个获授权接续：**Violation-only结束并通过审计 → Profile-only一次训练/测试 → 四行消融表 → 停止**。不自动扩展seed、HPO或其他实验。

Profile-only移除B1违例项，结构权重1、违例权重0；仍为FB、seed2026、32步/120步调度、同一Stage1和训练QA。已有Full和Random不重新运行。新模型只做一次完整测试，不做全验证和新选模。

## 当前已验证的准备

- 当前Violation-only运行使用它原先的独立冻结代码，仓库的组件入口扩展不会修改该运行。
- Profile-only已在CPU精确重放全部3,109题Full bank，再冻结本组候选。3,098题集合改变，平均共享2.2650/9个Full候选；QA与非候选元数据不变。
- 共享组件入口沿用`run_fb_violation_only_control.py`文件名，通过显式配置`fb_profile_only_matched.json`选择Profile-only；候选文件、方法标签、配对列名及论文输出路径均按variant独立生成。
- 47项测试覆盖Profile不依赖B1分数、两组论文汇总、前序未完不启动、失败/消失阻断、GPU占用等待、CPU预检结束状态识别、启动幂等、残留intent拒绝重试和四行CSV/LaTeX汇总。

## 守护的真实行为

独立tmux守护每30秒更新队列`status.json`及`events.jsonl`。具体目录、会话和冻结命令见`RUN_RECORD.json`。

前序完成后先只读重算三组10,000题EM标记及配对结果，核对模型选择/缓存身份、32步训练预算、120步调度、完整原始预测和归档SHA-256。当前会话退出且GPU空闲后，才放行Profile-only。

启动前先持久化`launch_receipt.json`的intent，成功后标记launched。已有回执不会再次启动；若启动过程中中断而残留intent，守护停下要求检查，不会猜测重试。实验失败或文件错配会写入blocked并保留现场，禁止自动改变配置或重新训练。

守护本身不发送外部通知。可随时读取队列状态；后续询问进度时，可据此检查真实阶段和结果。

## 完成后保留什么

两组实验各自保留全部检查点、optimizer/RNG、冻结代码、借用输入依赖、bank、逐题原始response、配置、日志与资源记录。队列保留两次独立完成审计及启动回执。

四行表及数据会写入`11_paper/artifacts/fb_component_ablation_seed2026/`：

- `metrics.csv`：Random / Profile-only / Violation-only / Full的正确数和EM。
- `paired_four_methods.csv`：同10,000题上的四个正确标记。
- `result.json`：Full减各单信号、各单信号减Random的逐题纠错/退化/净差。
- `table.tex`、`RESULTS.md`：论文可用表格与限定解释。

所有数值都取真实完整测试，单信号负结果也保留。整张表限于已有SC-CL获选配置及一个seed，不直接证明统计交互、普适协同或稳定性。队列完成后自动退出，不继续其他实验。
