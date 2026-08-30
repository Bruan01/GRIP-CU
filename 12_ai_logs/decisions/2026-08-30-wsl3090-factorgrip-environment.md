# Environment Memory: FactorGRIP v0.1 WSL2 Probe

日期：2026-08-30

## 已人工核验的可用环境

本记录是 FactorGRIP v0.1 candidate-energy probe 的项目记忆，供后续
Codex/WSL 会话直接复用。

- WSL 用户：`kieran`
- 工作区：`/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU`
- conda 环境：`guardenv`（必须使用；不要创建或切换到项目 `.venv`）
- GPU：`NVIDIA GeForce RTX 3090`
- GPU 显存：`24576 MiB`（`nvidia-smi` 当时占用约 `1736 MiB`）
- NVIDIA Driver：`581.80`
- NVIDIA-SMI 报告 CUDA：`13.0`
- `guardenv` PyTorch：`2.12.0+cu130`
- `torch.version.cuda`：`13.0`
- 交互式 WSL 中 `torch.cuda.is_available()`：`True`（以用户终端实际检查为准）

验证命令：

```bash
conda activate guardenv
nvidia-smi
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

说明：Codex 的受限命令执行通道可能报告 `GPU access blocked by the operating
system`，这不代表用户 WSL 终端不可用。正式 ML probe 必须从能看到 RTX 3090 的
用户 WSL 终端执行。

## 已发现的 Qwen2.5-0.5B 本地缓存

已发现完整的 Hugging Face 标准缓存：

```text
/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/
└── snapshots/7ae557604adf67be50417f59c2c2f167def9a775/
    ├── config.json
    ├── model.safetensors       (~988 MB)
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    └── merges.txt
```

这不是项目自定义的：

```text
grip-exp/model_cache/Qwen--Qwen2.5-0.5B-Instruct/
```

因此原先 `--model_source local --local_files_only` 的 runner 找不到模型，
不是模型不存在，而是缓存布局不同。

FactorGRIP wrapper 已修复为自动按以下顺序寻找模型：

1. `grip-exp/model_cache/Qwen--Qwen2.5-0.5B-Instruct/`；
2. `$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*`；
3. 通过 `MODEL_PATH` 显式指定完整 Transformers 模型目录。

如需显式指定当前缓存，可使用：

```bash
export MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
```

不需要重复下载 Qwen2.5-0.5B，也不要把模型文件提交到 Git。

## FactorGRIP v0.1 执行设置

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30
conda activate guardenv
CONDA_ENV=guardenv \
RUN_ID=wsl3090_nell23k_candidate_energy_20260830_01 \
RESUME=1 \
bash configs/run_nell23k_candidate_probe_wsl.sh
```

该 run ID 曾在模型路径检查阶段失败并已创建目录，因此恢复时必须带
`RESUME=1`。若开始全新实验，必须使用新的不可覆盖 `RUN_ID`。

固定实验约束：NELL23K、Qwen2.5-0.5B、64/32/64 输入、每题 10 个 candidates、
评估 recurrence `K=1`、`correct/none/wrong_depth` controls；不启动 7B、其他
数据集、重训或 FactorGRIP experts。

## 运行时修复记录

2026-08-30 20:04 的第一次恢复运行已通过 CUDA 和缓存发现阶段，但因
`grip/recurrent/model.py::_model_id()` 只接受逻辑 alias，不接受 wrapper 传入的
绝对本地模型目录而报 `KeyError`。现已修复：`_model_id()` 会先识别完整本地
Transformers 目录，再执行 alias 映射；无需下载或复制模型。对应 wrapper 仍使用
`--model_source local --local_files_only`，避免联网和隐式模型替换。

## 2026-08-30 20:31 constrained decoder 修复

正式运行已经通过模型缓存解析和 adapter 加载，但在 Transformers 4.57.3 的
`LogitsProcessorList` 调用阶段失败。原 `CandidateTrieLogitsProcessor.__call__` 使用
`**_: Any`，该版本通过 `inspect.signature()` 将 `_` 误判为必需参数，产生：

```text
ValueError: Make sure that all the required parameters: ['input_ids', 'scores', '_'] ...
```

现已将签名收紧为仅接受 Hugging Face 标准的两个参数：

```python
def __call__(self, input_ids, scores)
```

并增加 `LogitsProcessorList` 回归测试；静态检查和全部 32 个 unittest 已通过。

Codex 受限执行通道当前无法访问宿主 GPU（`GPU access blocked by the operating system`，
`torch.cuda.is_available() == False`），所以不能在该通道代跑正式 CUDA probe。用户自己的
WSL 终端已核验 RTX 3090 可用；请在用户终端用原 RUN_ID + `RESUME=1` 继续。

## 2026-08-30 20:45 probe 完成

修复后使用 `guardenv`、用户 WSL RTX 3090、原 RUN_ID 和 `RESUME=1` 成功完成正式
NELL23K probe。输出目录：

```text
08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30/results/runs/wsl3090_nell23k_candidate_energy_20260830_01/
```

验收记录：

- `predictions.jsonl`：864 行（64 test + 32 validation，3 controls × 3 decoders）
- `candidate_scores.jsonl`、`analysis/summary.json`、`decoder_accuracy.csv`、`paired_decoder_effects.csv` 均已生成
- `run.log`：`status=complete`，elapsed 402.85 s，GPU wall time 429.95 s
- `environment.txt`：`guardenv` PyTorch `2.12.0+cu130`，`cuda_available=true`
- 最大记录峰值显存约 4.55 GB（score decoder）；无 OOM/CUDA error

关键分析结果（仅记录实验输出，不将其解释为新的训练结论）：

- validation：correct/free 12.5%，correct/constrained 43.75%，correct/score 37.5%
- test：correct/free 17.1875%，correct/constrained 34.375%，correct/score 34.375%
- `score_vs_free_gain`：validation +25 pp，test +17.1875 pp
- `correct_none_delta`：validation -15.625 pp，test -3.125 pp
- Go gate：`false`，原因包含单图没有 shuffled adapter，且 `wrong_or_shuffled_below_correct=false`；按设计记录为 STOP，不是运行失败。

## Mac/WSL 跨平台编码防错规范（必须遵守）

本次连续报错的共同原因不是 RTX 3090 显存不足，而是接口、路径和执行环境没有
在代码层面显式校验。后续在 Mac 上写代码时按以下规则实现和验收：

1. **CLI 参数名必须统一并做别名兼容。** Python `argparse` 的 `dest` 使用下划线，
   shell/PowerShell 常用连字符；同一个参数同时注册两个 spelling，例如
   `--local_files_only` 和 `--local-files-only`，并显式写 `dest="local_files_only"`。
   wrapper 不得凭习惯混用。所有入口至少执行 `python ... --help` 和一个最小参数解析
   smoke test；shell 入口执行 `bash -n`。
2. **框架回调不要使用会被反射误判的可变参数。** Transformers 的
   `LogitsProcessorList` 会通过 `inspect.signature()` 判断回调参数；因此
   `__call__` 必须严格匹配框架协议（本例仅 `input_ids, scores`），不要用
   `**kwargs`/`**_` 伪装兼容参数。应增加真实 `LogitsProcessorList` 回归测试，而不是
   只直接调用 processor。
3. **路径不得硬编码机器用户名、盘符或运行目录。** Python 使用 `pathlib.Path`，
   shell 使用 `BASH_SOURCE[0]` 定位项目；模型路径提供 `MODEL_PATH` 显式覆盖，并按
   项目缓存、`HF_HOME`/Transformers 标准缓存顺序解析。结果文件可以记录绝对路径用于
   provenance，但源代码和默认配置不得依赖 `/home/kieran`、`/mnt/c/...` 或 Windows
   用户名。
4. **shell 工具要考虑 macOS BSD 与 GNU 差异。** 不在可移植 wrapper 中依赖
   GNU-only 的 `find -printf`、`-mindepth` 等选项；优先使用 shell glob、Python
   `pathlib` 或已验证的 POSIX 选项。当前模型 snapshot 搜索已改为 shell glob。
5. **环境必须显式记录和验证。** 统一使用 conda `guardenv`，运行前检查 Python、
   PyTorch、Transformers、PEFT 版本；CUDA 任务先检查
   `torch.cuda.is_available()` 和设备名，CPU fallback 必须显式指定，不能静默回退。
   Codex/受限容器看不到 GPU 时，不把它误判为用户 WSL 的 GPU 故障。
6. **模型加载阶段要区分“基础加载”和“最终放置”。** 基础模型可能先在 CPU 上
   加载，随后才执行 `model.to(device)`；日志必须标明阶段，不能仅凭基础加载日志
   判断发生了 CPU fallback。正式验收同时检查最终参数 device、`cuda_available`、
   峰值显存和运行结果。
7. **结果目录和权重目录必须分离。** Git 提交实验分析、CSV/JSON/JSONL、环境记录、
   run metadata；忽略 `model_cache/`、`*.safetensors`、`*.ckpt`、`*.pt`、adapter
   权重等大文件。需要提交被全局 ignore 的实验结果时，只 force-add 明确的结果 run，
   提交前用 `git diff --cached --name-only` 和扩展名扫描确认没有权重。

本规范对应的回归检查：`configs/check_mac_static.sh`、32 个 unittest、候选约束
processor 的 `LogitsProcessorList` 测试，以及正式运行前的入口 `--help` 检查。
