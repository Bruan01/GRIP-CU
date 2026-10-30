#!/usr/bin/env python3
"""Record task learning, commit explicit paths, and guard Codex turn completion."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args], cwd=root,
        capture_output=True, text=True, check=check,
    )


def repository(cwd: str | Path) -> Path:
    return Path(git(Path(cwd), "rev-parse", "--show-toplevel").stdout.strip())


def receipt_dir(root: Path) -> Path:
    path = Path(git(root, "rev-parse", "--git-path", "task-closeout").stdout.strip())
    return path if path.is_absolute() else root / path


def token(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:20]


def write_receipt(root: Path, task_id: str, data: dict) -> None:
    folder = receipt_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    data = {**data, "task_id": task_id}
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    for path in (folder / f"{token(task_id)}.json", folder / "latest.json"):
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(payload)
        temporary.replace(path)


def finish(args: argparse.Namespace) -> dict:
    root = repository(args.cwd)
    branch = git(root, "branch", "--show-current").stdout.strip()
    if not branch:
        raise ValueError("当前为 detached HEAD，请先明确任务应提交的分支。")
    if git(root, "ls-files", "--unmerged").stdout:
        raise ValueError("存在未解决的合并冲突，任务收尾未提交。")
    if args.paths and args.staged:
        raise ValueError("--paths 与 --staged 只能选择一种。")
    if args.staged:
        # For explicit whole-worktree snapshots only; ordinary tasks use --paths.
        paths = git(root, "diff", "--cached", "--name-only", "-z").stdout.split("\0")
        paths = [p for p in paths if p]
    else:
        paths = args.paths or []
    normalized = []
    for value in paths:
        path = Path(value)
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        normalized.append(resolved.relative_to(root.resolve()).as_posix())
    task_id = args.task_id or f"manual-{uuid.uuid4().hex}"
    prior = receipt_dir(root) / f"{token(task_id)}.json"
    if prior.exists():
        receipt = json.loads(prior.read_text())
        if receipt["commit"] == git(root, "rev-parse", "HEAD").stdout.strip():
            return {**receipt, "reused": True}
        raise ValueError("该 task-id 已提交，请为新任务使用新的标识。")
    now = datetime.now().astimezone()
    note = Path("12_ai_logs/task_summaries") / f"{now:%Y%m%d-%H%M%S}-{token(task_id)}.md"
    (root / note).parent.mkdir(parents=True, exist_ok=True)
    (root / note).write_text(
        f"# {args.title}\n\n时间：{now.isoformat(timespec='seconds')}  \n分支：`{branch}`\n\n"
        f"## 完成内容\n\n{args.summary}\n\n## 验证\n\n{args.validation}\n\n"
        f"## 沉淀\n\n{args.lesson}\n\n"
        "## 远程策略\n\n本收尾脚本只做本地提交。推送需用户在对应任务中明确授权。\n"
    )
    selected = sorted(set([*normalized, note.as_posix()]))
    git(root, "add", "--", *selected)
    # --only avoids committing unrelated files already present in the index.
    git(root, "commit", "--only", "-m", f"task: {args.title}", "--", *selected)
    result = {
        "branch": branch, "commit": git(root, "rev-parse", "HEAD").stdout.strip(),
        "note": note.as_posix(), "paths": selected, "push_performed": False,
    }
    write_receipt(root, task_id, result)
    return result


def stop(payload: dict) -> dict:
    try:
        root = repository(payload.get("cwd", "."))
        # Do not apply this repository's policy to another checkout.
        if root.resolve() != Path(__file__).resolve().parents[2]:
            return {}
        head = git(root, "rev-parse", "HEAD").stdout.strip()
        task_id = f"{payload.get('session_id', '')}:{payload.get('turn_id', '')}"
        exact = receipt_dir(root) / f"{token(task_id)}.json"
        latest = receipt_dir(root) / "latest.json"
        for path in (exact, latest):
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            message = payload.get("last_assistant_message") or ""
            if data.get("commit") == head and (
                path == exact or head[:7] in message or head[:12] in message
            ):
                if head[:7] in message and any(word in message for word in ("推送", "远程", "push")):
                    return {}
                if payload.get("stop_hook_active"):
                    return {"systemMessage": f"本地提交 {head[:7]} 已完成；最终回复请报告提交号与远程推送状态/询问，无需再次提交。"}
                return {
                    "decision": "block",
                    "reason": f"本地提交 {head[:7]} 已完成，无需再次提交。请在最终回复中报告该提交号，并询问是否推送；本任务已授权推送则完成推送并核验后报告状态。",
                }
        if payload.get("stop_hook_active"):
            return {"systemMessage": "任务收尾尚无匹配提交记录；请如实报告未完成状态，不声称已自动提交。"}
        return {
            "decision": "block",
            "reason": (
                "仓库收尾规则：若本任务已完成，请先总结完成内容、实际验证和可复用经验，"
                "调用 .codex/hooks/task_closeout.py finish，传入明确的 --paths（只读任务可省略）"
                f"及 --task-id {json.dumps(task_id)}。脚本只在当前分支做本地提交。"
                "最终回复包含短提交号，并询问是否推送；本任务已授权推送则直接完成推送并核验。"
                "如果任务未完成或正在等待用户输入，不提交完成记录，准确说明状态后结束。"
            ),
        }
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        return {"systemMessage": f"任务收尾检查未完成：{type(exc).__name__}。请按 AGENTS.md 手动核验。"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    close = commands.add_parser("finish")
    for name in ("title", "summary", "validation", "lesson"):
        close.add_argument(f"--{name}", required=True)
    close.add_argument("--cwd", default=".")
    close.add_argument("--task-id")
    close.add_argument("--paths", nargs="*")
    close.add_argument("--staged", action="store_true")
    commands.add_parser("stop")
    args = parser.parse_args()
    if args.command == "stop":
        try:
            result = stop(json.load(sys.stdin))
        except (ValueError, TypeError):
            result = {"systemMessage": "Stop hook 输入不是有效 JSON，未执行任何提交。"}
    else:
        result = finish(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
