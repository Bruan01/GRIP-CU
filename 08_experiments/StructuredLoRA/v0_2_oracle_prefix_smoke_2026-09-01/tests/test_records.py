import unittest
from pathlib import Path

from structured_lora.records import load_jsonl, validate_splits


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
DATA_ROOT = REPO_ROOT / "08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/artifacts/exact_hop"


class RecordTest(unittest.TestCase):
    def test_registered_splits_are_strict_and_disjoint(self):
        splits = {
            "train": load_jsonl(DATA_ROOT / "nell23k_exact_hop_train.jsonl"),
            "validation": load_jsonl(DATA_ROOT / "nell23k_exact_hop_validation.jsonl"),
            "test": load_jsonl(DATA_ROOT / "nell23k_exact_hop_test.jsonl"),
        }
        audit = validate_splits(splits)
        self.assertEqual(audit["split_sizes"], {"train": 716, "validation": 152, "test": 156})
        for split in splits:
            self.assertEqual(audit["depth_counts"][split], {"1": len(splits[split]) // 4, "2": len(splits[split]) // 4, "3": len(splits[split]) // 4, "4": len(splits[split]) // 4})


if __name__ == "__main__":
    unittest.main()
