# FB 单种子 Random-9 对照：运行与论文产物

实验标识：`SC_CTRL_RANDOM_FB_SEED2026_FIXED32`。设计见 [PLAN.md](PLAN.md)。仅 seed=2026；实际32优化步，保持5-epoch/120步线性调度。没有新超参搜索、重复种子或按测试结果追加预算。

## 已完成的观测结果

2026-11-05 21:27（北京时间）单次运行已完整结束。

| 方法 | 完整测试正确数 | 直接解码EM |
|---|---:|---:|
| 原SC-CL-9 | 8,990/10,000 | 89.90% |
| Random-9 + 同一InfoNCE | 8,648/10,000 | 86.48% |

SC-CL减Random为+3.42个百分点、净增342题；纠错553、退化211、共同正确8,437、共同错误799。Random完整验证为15,196/17,535（86.660964%）。全部过程及四个检查点保留；结果文件位于仓库 `11_paper/artifacts/fb_random9_control_seed2026/`。本结果只支持已有SC-CL获选配置下的一次条件性采样比较。

它回答“在已有获选训练配置下，组合采样相对均匀采样的增量是多少”。配置此前按SC-CL验证表现选定，因此不能把它写成双方等预算调参后的最优方法比较；均匀随机比较也不独立识别结构分量或排除关系频率解释。

## 运行方式

在仓库根目录执行；必须使用项目 guardenv：

```bash
# 仅CPU：冻结输入/代码，构造bank并核验原方法合法池
/home/kieran/miniconda3/envs/guardenv/bin/python \
  08_experiments/Hard_Negative_Contrastive_GRIP/scripts/run_fb_random9_control.py prepare

# 仅CPU：以实际训练CLI解析配置和数据，不加载模型
/home/kieran/miniconda3/envs/guardenv/bin/python \
  08_experiments/Hard_Negative_Contrastive_GRIP/scripts/run_fb_random9_control.py check

# 完整单次实验：训练 -> 全验证 -> 固定第32步模型全测试 -> 论文产物
/home/kieran/miniconda3/envs/guardenv/bin/python \
  08_experiments/Hard_Negative_Contrastive_GRIP/scripts/run_fb_random9_control.py run
```

也可显式使用 `train`、`evaluate`、`report` 分阶段执行。`evaluate` 先冻结权重身份，再完整验证及测试；验证分数不用于另选检查点。主指标是直接解码EM，本轮不自动增加候选MRR/Hits推理预算。

默认运行目录由配置给出，位于Linux运行盘。`--run-dir` 允许隔离开发验证，不能用来伪装不同实验配置或重复种子。正式运行使用已经准备好的冻结代码入口（其路径见 `RUN_RECORD.json`），使后续仓库代码改动不影响后台训练和评测。

## 恢复与过程保留

- 控制器文件锁阻止同一目录重复启动。输入、bank及完整代码快照都有SHA-256；恢复前逐项检查，禁止变更后混用旧检查点。
- 训练保留第8/16/24/32步的检查点，包含adapter、optimizer、scheduler、RNG及Trainer状态。中断后同一 `run` 命令继续；最新检查点缺文件则报错保留现场，不自动丢弃或降级恢复。
- 已完成的32步adapter通过预算检查后复用；测试预测由原评测器的权重/提示/配置身份保护并断点续跑。
- 没有自动重试、重新抽bank、清理检查点或测试后扩大训练。
- `refresh-code` 只供训练前修复控制器使用：保存旧代码版本、记录新旧hash并重新执行CLI检查。任何训练检查点或模型已存在时禁止使用。它不重新抽候选，也不改变已锁定训练设置。
- GPU训练和评测在Linux目录运行；基础7B模型使用原缓存，Stage1 adapter独立复制以保留初始化。父SC-CL模型与结果保持只读。
- 优化步数、有效batch及候选数量匹配；关系文本token长度可能不同，因此真实耗时与显存另行记录，不宣称严格等FLOPs或等墙钟。

## 产物目录

| 位置 | 内容 | 用途 |
|---|---|---|
| `contract.json` | 冻结设计、全部输入和代码hash、实际训练argv | 复现与比较身份 |
| `inputs/` | 完整训练QA、验证/测试、train图、关系/实体映射、资格表及Stage1模型 | 独立重放 |
| `reference/` | 原SC-CL审计、选模记录、配置、训练历史、原候选库及完整测试预测 | 父结果身份与配对参考 |
| `bank/random9.jsonl` | 每题冻结的九个候选、初始/条件概率、排除关系、seed | 采样复现 |
| `bank/pool_audit.jsonl` | 每题完整合法关系池、pool hash及抽样顺序 | 验证与原方法同池；不需再次B1推理 |
| `bank/wiring.json` | QA覆盖、候选池对齐、原score dump hash | train-only与数据接入审计 |
| `code_snapshot/` | 完整Python执行代码及其包结构 | 防止运行中代码漂移 |
| `training/` | 四个检查点、最终模型、预算与运行配置 | 训练恢复及模型保留 |
| `selection.json` | 测试前冻结的第32步模型hash | 固定选模证据 |
| `evaluation/{validation,test}/` | 每题response、正确标记、完整summary、cache identity | 复核指标与错误分析 |
| `logs/`、`events.jsonl` | 每次尝试命令、stdout/stderr、阶段与错误 | 完整执行轨迹 |
| `environment_*.json`、`resources.jsonl` | 软件、设备、存储、每20秒资源与墙钟 | 方法细节与成本分析 |
| `paper/` | 结果JSON、指标CSV、匿名配对CSV、配置对齐表、损失历史、LaTeX表与结果说明 | 写论文与绘图 |
| `artifact_manifest.json` | 输入、代码、模型、optimizer及结果的完整文件hash与大小 | 长期归档证据 |

原721MB B1 score dump只读复用并记录hash，不复制到运行包；每题完整合法池及train输入均已独立保存，后续复现随机bank不依赖再次B1评分。

## 论文使用规则

实验完成并通过全量逐题核验后，脚本自动将 `paper/` 小型产物复制到仓库 `11_paper/artifacts/fb_random9_control_seed2026/`，附模型/原预测定位及归档清单hash。大模型、optimizer和完整原始response留在Linux运行目录，不进入Git。

核心表格为SC-CL参考与Random-9的完整测试EM；配对差异方向固定为 **SC-CL − Random-9**：rescue表示SC-CL正确而Random错误，regression表示相反变化。无论正差、负差还是零差都保留。

单个训练seed不报告跨seed标准差、显著性或稳定性结论。不同历史B1、闭集重排、当前直接解码不可混合。共享训练QA包含既有464条train-pair修复，应在方法/数据说明中保留。

合成fixture仅用于验证采样、错配拒绝、断点保护和论文导出，不属于研究结果。正式Random-9尚无完整测试时，任何说明不得填写估计性能。
