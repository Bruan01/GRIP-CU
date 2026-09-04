import json
import tempfile
import unittest
from pathlib import Path

from priority_distill.config import load_config


class SeedSweepTests(unittest.TestCase):
    def test_fair_configs_use_corrected_generation_budget(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("direct_answer_only_fair_20260903.json", "graph_free_trace_joint_fair_20260903.json"):
            config = load_config(root / "configs" / name)
            self.assertEqual(config["data"]["max_new_tokens"], 80)

    def test_aggregate_inputs_are_json_only(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "scripts" / "aggregate_seed_results.py").is_file())
        self.assertNotIn("adapter_model", (root / "scripts" / "aggregate_seed_results.py").read_text())
