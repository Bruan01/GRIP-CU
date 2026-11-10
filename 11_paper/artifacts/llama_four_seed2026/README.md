# Llama 四数据集主表

使用GRIP采用的`meta-llama/Llama-3.1-8B-Instruct`，seed2026；WN18RR、FB15K237-2、CoDEx-M、NELL23K依次串行。每图重新训练Stage1、B1和SC-CL，两臂共享Stage1及Stage2优化配置/步数，B1候选损失权重为0。

本文件记录准备时点：权重下载、原生聊天转换及CPU预检进行中，尚无Llama性能。实时状态见`/home/kieran/grip-cu-runs/llama_four_20261111/status.json`。完成后各图结果在子目录`result.json`，汇总为`main_table.csv/tex`；只有四图全部完成才会生成`completion.json`。

固定范围与存储策略见[计划](../../../08_experiments/Hard_Negative_Contrastive_GRIP/docs/llama_four_dataset_20261111/PLAN.md)。模型只下载一份HF分片，排除original重复权重。新运行每阶段保留一个活动恢复点，最终adapter及训练历史身份核对后删除本轮临时checkpoint tensor/optimizer状态。最终Stage1/B1/SC-CL模型、原始预测与候选分数、bank、配置、代码快照、SHA和资源日志保留。

每图流程：聊天转换并核对所有消息 → 真实CPU入口预检 → Stage1 → B1 → B1 train全关系评分 → SC-CL bank审计 → SC-CL → 两臂完整直接EM与原列表重排 → 配对及汇总。WN为11选项，其余为10。不会把Qwen adapter或teacher分数当作Llama资产。

本轮为原Qwen获选配置的固定迁移，无新Llama HPO、重复种子或测试后预算扩展；微批1，累积步数保留原有效批大小，两臂启用梯度检查点。保留单种子、原任务开发历史和WN历史实体对核验限制；不宣称严格复现原文数值或跨种子稳定。

控制器每30秒记录进程、日志、GPU及实际C/E空间；看门程序检测控制器退出和心跳中断，异常写入`alerts.jsonl`。失败停止后续接续并保留现场，不自动改模型、重采bank或无限重试。WSL关闭会中断作业。

旧实验仅删除用户2026-11-11明确授权的80份文件，回执在计划目录；旧归档的“全部中间权重保留”自该回执起不再适用于这些文件，最终模型及最后恢复点均保留。

2026-11-11 03:06启动更新：权重完整SHA核验完成，首图WN Stage1已启动。主会话`llama_four_20261111`，看门会话`llama_watch_20261111`；启动并不代表四图训练/测试完成。发现的重复BOS问题已在任何GPU训练前修正并核对真实token序列，源码与预检版本历史保留。

启动恢复记录：首轮在训练前embedding扩容时显存不足并触发C盘保护，保留失败现场后改用Llama已有专用PAD（ID128004），不扩词表。已按相同预算重新启动，详见计划目录padding_recovery.json；未完成任何优化步的失败启动不计作训练结果。
