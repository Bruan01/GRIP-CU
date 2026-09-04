import hashlib
import tempfile
import unittest
from pathlib import Path

from entity_decoder.checkpoints import resolve_checkpoint_registry


class CheckpointRegistryTest(unittest.TestCase):
    def test_validates_hash_and_preserves_no_adapter_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "adapter.pt"
            ckpt.write_bytes(b"fixture")
            sha = hashlib.sha256(b"fixture").hexdigest()
            entries = [
                {"name": "base", "kind": "no_adapter", "prompt_protocol": "answer_only"},
                {"name": "trained", "kind": "adapter", "path": "adapter.pt", "sha256": sha, "prompt_protocol": "answer_only"},
            ]
            resolved = resolve_checkpoint_registry(entries, root, require_files=True)
            self.assertIsNone(resolved[0]["resolved_path"])
            self.assertEqual(resolved[1]["observed_sha256"], sha)

    def test_can_audit_missing_remote_checkpoint_without_passing_runtime_gate(self):
        entries = [{"name": "trained", "kind": "adapter", "path": "missing.pt", "sha256": "0" * 64, "prompt_protocol": "answer_only"}]
        resolved = resolve_checkpoint_registry(entries, Path("/tmp"), require_files=False)
        self.assertEqual(resolved[0]["status"], "missing")
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_registry(entries, Path("/tmp"), require_files=True)


if __name__ == "__main__":
    unittest.main()
