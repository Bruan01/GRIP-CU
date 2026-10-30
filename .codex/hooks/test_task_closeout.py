"""Exercise closeout in disposable repos, never in the research worktree."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("task_closeout.py")


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.git("init", "-b", "fixture")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        folder = self.root / ".codex/hooks"
        folder.mkdir(parents=True)
        self.script = folder / SCRIPT.name
        shutil.copy2(SCRIPT, self.script)
        for name in ("own.txt", "other.txt"):
            (self.root / name).write_text("initial\n")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, text=True,
                              capture_output=True, check=True).stdout.strip()

    def finish(self, *args, check=True):
        return subprocess.run([
            sys.executable, str(self.script), "finish", "--title", "fixture task",
            "--summary", "完成指定文件", "--validation", "临时仓库验证",
            "--lesson", "只提交明确指定的文件", *args,
        ], cwd=self.root, text=True, capture_output=True, check=check)

    def stop(self, **overrides):
        payload = {"cwd": str(self.root), "session_id": "fixture", "turn_id": "turn",
                   "last_assistant_message": "完成", **overrides}
        result = subprocess.run([sys.executable, str(self.script), "stop"],
            cwd=self.root, input=json.dumps(payload), text=True,
            capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_only_task_paths_committed_and_unrelated_index_preserved(self):
        (self.root / "own.txt").write_text("own change\n")
        (self.root / "other.txt").write_text("unrelated staged change\n")
        self.git("add", "other.txt")
        result = json.loads(self.finish("--paths", "own.txt").stdout)
        committed = self.git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
        self.assertIn("own.txt", committed)
        self.assertNotIn("other.txt", committed)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "other.txt")
        self.assertFalse(result["push_performed"])
        self.assertEqual(result["branch"], "fixture")

    def test_same_task_id_is_idempotent(self):
        first = json.loads(self.finish("--task-id", "fixture:turn").stdout)
        second = json.loads(self.finish("--task-id", "fixture:turn").stdout)
        self.assertEqual(first["commit"], second["commit"])
        self.assertTrue(second["reused"])
        self.assertEqual(len(list((self.root / "12_ai_logs/task_summaries").glob("*.md"))), 1)

    def test_read_only_task_creates_summary_and_commit(self):
        old = self.git("rev-parse", "HEAD")
        result = json.loads(self.finish().stdout)
        self.assertNotEqual(old, result["commit"])
        note = (self.root / result["note"]).read_text()
        self.assertIn("## 沉淀", note)
        self.assertIn("## 验证", note)

    def test_stop_requests_closeout_when_missing(self):
        self.assertEqual(self.stop()["decision"], "block")

    def test_stop_accepts_matching_turn_receipt(self):
        result = json.loads(self.finish("--task-id", "fixture:turn").stdout)
        self.assertEqual(self.stop(last_assistant_message=f"已提交 {result['commit'][:7]}。是否推送到远程？"), {})

    def test_stop_accepts_manual_commit_reported_in_final(self):
        result = json.loads(self.finish().stdout)
        self.assertEqual(self.stop(last_assistant_message=f"已推送 {result['commit'][:7]}"), {})

    def test_stop_reminds_without_requesting_a_second_commit(self):
        self.finish("--task-id", "fixture:turn")
        result = self.stop()
        self.assertEqual(result["decision"], "block")
        self.assertIn("无需再次提交", result["reason"])

    def test_stop_continuation_is_bounded(self):
        result = self.stop(stop_hook_active=True)
        self.assertNotIn("decision", result)
        self.assertIn("未完成", result["systemMessage"])

    def test_detached_head_rejected_before_writing_summary(self):
        self.git("checkout", "--detach")
        result = self.finish(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "12_ai_logs/task_summaries").exists())


if __name__ == "__main__":
    unittest.main()
