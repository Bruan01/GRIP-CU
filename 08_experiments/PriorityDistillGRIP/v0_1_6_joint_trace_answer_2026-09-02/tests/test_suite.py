"""Protocol-level tests for the v0.1.3 controlled experiment.

The v0.1/v0.1.2 suite gate is intentionally not used here: v0.1.3 compares
controlled protocols with different supervision mechanisms, including the
anti-copy and evidence-replay conditions.
"""
import unittest
from pathlib import Path

from priority_distill.config import load_config, validate_config
from priority_distill.controlled import PROTOCOLS


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_all_registered_configs_load(self):
        expected = {
            "direct_answer_only.json": "direct_answer_only",
            "oracle_two_stage.json": "oracle_two_stage",
            "candidate_selection_two_stage.json": "candidate_selection_two_stage",
            "candidate_selection_anti_copy.json": "candidate_selection_anti_copy",
            "candidate_selection_anti_copy_replay.json": "candidate_selection_anti_copy_replay",
            "graph_free_trace.json": "graph_free_trace",
            "graph_free_trace_joint.json": "graph_free_trace_joint",
            "candidate_index_anti_copy_bridge.json": "candidate_index_anti_copy_bridge",
        }
        for filename, protocol in expected.items():
            config = load_config(ROOT / "configs" / filename)
            self.assertEqual(config["protocol"], protocol)
            self.assertEqual(config["protocol"], PROTOCOLS[protocol] and protocol)

    def test_replay_is_only_enabled_for_anti_copy_replay(self):
        for filename in (
            "direct_answer_only.json",
            "oracle_two_stage.json",
            "candidate_selection_two_stage.json",
            "candidate_selection_anti_copy.json",
        ):
            config = load_config(ROOT / "configs" / filename)
            self.assertEqual(config["training"].get("stage2_replay_ratio", 0.0), 0.0)

        replay = load_config(ROOT / "configs" / "candidate_selection_anti_copy_replay.json")
        self.assertEqual(replay["training"]["stage2_replay_ratio"], 0.2)

    def test_non_replay_protocol_rejects_nonzero_replay(self):
        config = load_config(ROOT / "configs" / "direct_answer_only.json")
        config["training"]["stage2_replay_ratio"] = 0.2
        with self.assertRaisesRegex(ValueError, "registered replay/bridge protocols"):
            validate_config(config)

    def test_replay_protocol_requires_ratio_point_two(self):
        config = load_config(ROOT / "configs" / "candidate_selection_anti_copy_replay.json")
        config["training"]["stage2_replay_ratio"] = 0.1
        with self.assertRaisesRegex(ValueError, "fixes stage2_replay_ratio at 0.2"):
            validate_config(config)

    def test_joint_protocol_uses_joint_supervision(self):
        spec = PROTOCOLS["graph_free_trace_joint"]
        self.assertEqual(spec["stage1_method"], "graph_free_trace_answer_joint")
        self.assertEqual(spec["stage2_primary_method"], "graph_free_trace_answer_joint")

    def test_bridge_uses_trace_as_stage2_primary_and_explicit_replay(self):
        spec = PROTOCOLS["candidate_index_anti_copy_bridge"]
        self.assertEqual(spec["stage1_method"], "explicit_path_selection_terminal_masked")
        self.assertEqual(spec["stage2_primary_method"], "graph_free_trace_terminal_masked")
        self.assertEqual(spec["replay_method"], "explicit_path_selection_terminal_masked")



if __name__ == "__main__":
    unittest.main()
