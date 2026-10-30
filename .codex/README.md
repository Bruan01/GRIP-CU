# 本仓库任务收尾

每个任务完成后执行：总结完成内容、验证与可复用经验 → 提交到当前本地分支 → 提醒是否推送远程。`AGENTS.md` 保存持续约定，`hooks.json` 的 Stop 事件检查是否遗漏，`hooks/task_closeout.py finish` 执行总结和提交。

首次使用在 Codex CLI 的 `/hooks` 中审阅并信任本项目的 Stop hook；已有配置变更后也可能需要重新信任。项目 hook 的加载和信任规则见 [官方文档](https://learn.chatgpt.com/docs/hooks)。本仓库没有自动修改用户全局 hook 信任配置。

按 AGENTS.md 的示例调用 `finish`，`--paths` 明确列出本任务文件。没有文件改动的只读任务也会产生简短总结。脚本通过 `git commit --only` 保留其他已暂存文件，不执行 push、分支切换、强推或训练。Stop 自动续跑时传入其给出的 task-id；同 task-id 不重复提交。

`--staged` 只用于用户明确要求提交整份当前工作区、且已审查暂存区的情形。不要将其用于普通任务。运行凭据写入 Git 私有目录 `task-closeout/`；总结正文保存在可版本化的 `12_ai_logs/task_summaries/`。
