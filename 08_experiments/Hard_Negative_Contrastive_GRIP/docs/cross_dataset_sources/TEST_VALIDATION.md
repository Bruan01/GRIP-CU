# 实际验证

2026-11-07 guardenv：test_cross_dataset_controls.py、test_matched_rerank.py、test_frequency_matched_control.py、test_k_sensitivity_chain.py 最终合计76 passed / 14 warnings，44.28s。最终警告为torch弃用；此前一次75项回归还出现过NVML初始化提示，实际nvidia-smi与运行中的K3正常。无新模型测量、无模拟性能。

四个真实CLI已解析配置、解析完整QA及候选库，并逐字段对齐Full训练配置；NELL每组12014/2974、WN每组12334/2709，model_loaded=false。共享训练入口、QA loader、loss、candidate scorer的冻结SHA与当前源文件逐一一致。WN预检前补充的独立核验范围字段没有改变任何候选字符串；每组恰55题明确未核实。

FB恢复后的第8步检查点：global_step=8 / max_steps=120；完整safetensors数据长度与文件长度一致，optimizer/scheduler/RNG文件具有有效zip结构，训练随后继续到第9步。守护运行观察另见RUN_RECORD。

磁盘核验：Windows Lxss注册表确认Ubuntu VHDX位于C盘，C仅余约5.6GiB。六个当前/未来训练输出目录重定向到E盘；复制前后文件清单/大小/mtime不变，所有复制文件SHA一致，原目录保留recovery。K3进程PID10010未重启，继续第10步；现有归档rglob能遍历training根目录符号链接。实际E盘85.44GiB可用，后续产物估计74.49GiB，另保留8GiB；按每数据集启动前重新核验输出路径和真实物理余量。
