# Violation-only：结构分量的下一项关键对照

本轮只新增 **Violation-only-9**：移除采样utility中的结构项，保留冻结B1违例度。已有Full SC-CL=89.90%与Random-9=86.48%均直接复用。完整叙事和正/零/负结果解释见[PLAN.md](PLAN.md)。新方法EM尚未产生，不能预先填写或声称结构项已被证明有效。

叙事链条是：已有主实验描述方法表现；Random对照量化组合候选构造的价值；本轮进一步判断结构项在B1困难候选基础上是否有增量。Profile-only尚未运行，因此即便本轮为正，也不能宣称双信号的统计交互或“两者缺一不可”。

## 固定协议

- FB15K237-2、唯一seed2026、同一Stage1和12,136条QA，3,109条候选损失资格、每题9个候选。
- 仅将`structure_weight`从1改为0。违例项权重1、均匀下限1/3、采样温度0.25、InfoNCE权重/温度、LoRA和优化器均不变。
- 实际32步，仍按5-epoch/120步线性调度；保留8/16/24/32全部检查点、optimizer、scheduler和RNG状态。
- 第32步预先固定，只做一次10,000题完整测试；不做全量验证、不另行选步、不根据测试结果续搜。
- 预计新增约6–7小时（训练3–3.5小时、测试和归档约2.5–3小时）。这是按上一轮同机实测推算，不是保证时间。

## 可运行入口

在仓库根目录执行。默认只做CPU准备；`check`不加载模型：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python \
  08_experiments/Hard_Negative_Contrastive_GRIP/scripts/run_fb_violation_only_control.py prepare

/home/kieran/miniconda3/envs/guardenv/bin/python \
  08_experiments/Hard_Negative_Contrastive_GRIP/scripts/run_fb_violation_only_control.py check
```

显式`run`才会训练和测试；正式后台启动使用已准备的冻结代码路径以及`--config <运行目录>/inputs/design.json`。确切命令和会话见`RUN_RECORD.json`。同一命令支持检查点恢复，禁止修改bank/代码/输入后混用原检查点。

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python \
  08_experiments/Hard_Negative_Contrastive_GRIP/scripts/run_fb_violation_only_control.py run
```

## 准备和归档内容

CPU准备先重放Full bank：所有3,109题的九候选顺序必须与原方法完全一致，随后才生成移除结构项的bank。这避免把B1评分版本、合法池或关系画像计算差异误当作组件效应。若所有候选集合均未改变，准备阶段即拒绝继续消耗GPU。

| 路径 | 内容 |
|---|---|
| `contract.json` | 设计、命令、全部输入/代码/候选hash、评测政策 |
| `inputs/`、`reference/` | 只读借用上一轮的冻结数据、Stage1、Full预测及Random预测 |
| `code_snapshot/` | 从上一轮复制相同训练/评测代码，并加入本轮冻结控制器 |
| `bank/violation_only.jsonl` | 明确标记`ablation_variant=violation_only`的九候选库 |
| `bank/sampler_audit.jsonl` | 逐题结构相似度、违例度及池统计 |
| `bank/diagnostics.csv` | Full/V候选重合、集合/顺序变化、结构/违例度均值；不是新模型性能 |
| `bank/preparation_audit.json` | 全量重放、候选覆盖和QA一致性核验 |
| `training/` | 四个检查点、完整优化器状态及最终adapter |
| `selection.json` | 测试前固定模型hash，不使用新验证选模 |
| `evaluation/test/` | 原始response、正确标记、summary及模型/提示缓存身份 |
| `logs/`、`events.jsonl`、`resources.jsonl` | 每次命令、stdout/stderr、错误与资源轨迹 |
| `paper/` | 三方法指标、Full−V和V−Random配对CSV、训练历史、配置对齐及LaTeX表 |

`inputs/`和部分`reference/`使用明确记录的符号链接，不额外复制Stage1大权重。因此必须同时保留已完成Random对照的Linux运行目录；控制器逐项验证这些借用文件的hash。上一轮没有自动清理逻辑，本轮也不会删除任何父文件。

现有训练器将本轮视为SC-CL家族的参数消融，因此内部bank loader名仍为`structure_counterfactual`；方法身份由`structure_weight=0`、`ablation_variant=violation_only`和本轮experiment_id明确区分。论文输出只写Violation-only，不将其误称Full SC-CL。

完整测试通过核验后，脚本自动把小型论文产物导出到`11_paper/artifacts/fb_violation_only_control_seed2026/`。原始大产物及optimizer留在Linux目录；零或负结果同样导出。

## 解释边界

这是一组固定配置、单seed的组件替换。配置此前针对Full SC-CL验证集选择；历史测试暴露及464条训练QA修复条件继续保留。移除结构项改变整个采样分布，也可能改变困难度和频率分布，因此正差异不能直接证明语义结构、频率无关性或普适协同；这些问题需更细的对照。
