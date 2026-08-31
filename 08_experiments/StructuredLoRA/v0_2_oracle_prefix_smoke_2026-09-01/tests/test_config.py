import copy
import json
import unittest
from pathlib import Path

from structured_lora.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/oracle_prefix_smoke.json"


class ConfigTest(unittest.TestCase):
    def test_registered_config_is_valid(self):
        config = load_config(CONFIG)
        self.assertEqual(config["lora"]["total_rank"], 8)
        self.assertEqual(config["lora"]["groups"] * config["lora"]["group_rank"], 8)

    def test_rank_mismatch_is_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["lora"]["group_rank"] = 3
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
