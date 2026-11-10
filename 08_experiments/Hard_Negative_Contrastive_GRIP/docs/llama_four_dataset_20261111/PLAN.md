# Llama 四数据集主表与监控计划

2026-11-11；用户已明确授权存储清理，80份文件按清单删除，释放49.385GiB；108份保留文件大小/mtime一致，见storage_reclaim_receipt.json。Llama四个分片及配套文件全部通过SHA，四图聊天转换及8次CPU预检已完成，43项相关回归通过；WN Stage1已启动，尚无Llama模型性能。后续状态看运行根status.json。

## 固定目标

使用GRIP原文Llama3.1-8B及仓库映射`llama3-8b = meta-llama/Llama-3.1-8B-Instruct`，在WN18RR、FB15K237-2、CoDEx-M、NELL23K上生成B1与SC-CL配对主表。原Qwen结果独立保留。使用guardenv正式Python，seed2026；不自动扩展种子或HPO。

每图重新训练Llama Stage1及generation-only B1，并从同Stage1分叉SC-CL；不能使用Qwen adapter。冻结Llama B1后重新评分train关系，构造SC-CL候选bank，不能将Qwen teacher评分视为Llama teacher。验证选模及训练预算必须在正式测试前钉住。具体步数及两臂优化设置需在输入盘点后写入可执行契约，未冻结前不启动昂贵计算。

## 模板与数据

复用已有任务的语义内容；现有QA包含Qwen的im_start/im_end标记，需解析消息并用Llama原生chat template重新编码。对全部QA核对消息角色、内容、答案、question ID和顺序，更新派生任务SHA及绑定的eligibility身份。训练context原始文本保持同一语义内容。

评测保持现有原题、原选项、原顺序及EM规则，WN为11选项，其他为10。保留完整直接EM及同原候选列表MRR/H@1/H@3。四图验证集与测试集身份需逐项固定。WN既有55条未核实训练实体对及B1复现问题必须显式审计，不能静默声称清洗完成。

## 存储准入

实际C盘仅余约3.9GB，E盘仅余约6.3GB；WSL虚拟根目录报告761GB不代表宿主物理空间。Llama不在本地。

已从ModelScope仓库API获取文件清单：所需4份safetensors共16,060,556,376 bytes，约14.96GiB；另有tokenizer小文件。仅下载HF格式分片；排除`original/consolidated.00.pth`，避免再下载一份16GB原始权重。

`storage_reclaim_proposal.json`仅为提案：20个早期检查点的80份tensor/optimizer/scheduler/RNG文件，共49.385GiB；保留6组实验全部最终adapter及最后恢复点、全部JSON/配置/日志、曲线、bank、原始预测和候选分数。删除后早期模型/恢复状态不可找回。此前保留全部检查点的约定仍有效，本提案已由用户明确允许并按原清单执行，回执独立保留；不得扩大清理范围。

新运行共享一份底座和输入文本；每个活动阶段保留一个完整恢复点，阶段完成且最终adapter及日志身份核验后，仅新运行标记为临时的中间tensor可按冻结保留策略清理。最终Stage1/B1/SC-CL adapter各一份，不重复复制到论文目录。保留原始预测/候选分数及其压缩归档、bank、配置、代码快照、SHA、训练曲线和资源日志；论文目录保存小体积结果/配对CSV及证据索引。最终实际空间预算必须按Llama adapter大小核定后写入契约。

## 运行和监控门

单卡RTX3090 24GiB，串行。每个任务启动前核查真实C/E余量、无其他GPU计算作业、上游输出完成及输入SHA。先做Llama tokenizer/数据入口与显存预检，再部署持久队列。

监控每30秒采集进程、GPU显存/利用率、C/E真实余量、日志mtime/最近进度和退出码；区分模型加载、tokenization、训练、检查点保存及评测，不能仅凭GPU利用率低判死。worker失败、身份变化、产物缺失、磁盘不足或非有限损失阻断下一阶段，记录现场；不自动改预算、重采bank、换模型、清理旧实验或无限重试。WSL重启后须重新核对状态，不能保证守护跨实例存活。

目前尚无运行worker，因此没有Llama性能或训练进度。下一步为确认存储方案，再完成小文件tokenizer检查、可执行契约与队列；所有训练/评测结论只来自完整实测产物。

## 执行配置补记

B1与SC-CL使用各图原Qwen获选Stage2配置固定迁移：WN125步/125步调度，FB/CoDEx32/120，NELL32/240。每图Llama Stage1训练1 epoch。无新HPO或测试后改预算；微批1，累积步数保留原有效批大小。两臂均启用梯度检查点，训练后统一做两种完整测试评测。具体源配置、数据哈希、实际argv和代码清单见campaign_contract.json。

## 2026-11-11 03:06实际启动

底座四个分片和小文件全部通过SHA256。48,784条QA原生聊天内容核对、四图8次CPU入口检查、28,388题选项/gold核对完成；WN保持55个未核实实体对限制。发现并修复Llama聊天文本重复BOS，训练与评测均验证单BOS，相关回归43项通过。修复发生在GPU训练前，旧代码及CLI预检记录另存prelaunch历史，不冒充从旧训练恢复。

主tmux会话llama_four_20261111、独立看门llama_watch_20261111；运行根/home/kieran/grip-cu-runs/llama_four_20261111。03:06 WN Stage1已开始加载正确Llama底座到GPU；并非四图实验完成。当前状态、日志和告警以status.json、watchdog.json、alerts.jsonl为准；论文小产物入口11_paper/artifacts/llama_four_seed2026/。

## 首次启动失败及native PAD修复

首次WN Stage1在任何优化步之前，因默认新增[PAD]导致embedding扩容临时显存不足，宿主C盘同时触及保护线；控制器停止且未接续。失败状态、stderr、控制器日志均保存于运行根recovery/native_padding_attempt1/，不视为成功训练或断点续训。

修正为复用Llama原生finetune_right_pad_id（128004），词表保持128256，不增加embedding行。CPU tokenizer检查通过，修正代码重新冻结并记录padding_recovery.json后，按相同预算从头重新启动首图。此变更仅处理padding，未量化模型、缩减训练预算或更换底座。既有带[PAD]的adapter以及Qwen的padding行为保持兼容。

独立看门程序正在运行，当前异常以watchdog.json为准；alerts.jsonl中的首轮失败仍保留，不将恢复后的正常状态解释为之前没有失败。

03:15恢复后真实GPU验证：LlamaForCausalLM在cuda:0、42,467,328个LoRA可训练参数、Stage1已进入0/320训练循环；未触发embedding扩容。此时未报告任何完整训练/测试性能。
