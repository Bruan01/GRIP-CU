import copy
import unittest

from entity_decoder.config import validate_config


class ConfigTest(unittest.TestCase):
    def _checkpoint(self, name, role):
        return {
            "name": name,
            "kind": "adapter",
            "path": f"{name}.pt",
            "sha256": "a" * 64,
            "prompt_protocol": "answer_only",
            "role": role,
            "d0_predictions": {
                "validation": {"path": f"{name}.jsonl", "sha256": "b" * 64}
            },
        }

    def _config(self):
        return {
            "format_version": 1,
            "experiment_id": "x",
            "phase": "A_validation_only_d0_d1",
            "data": {
                "train_graph": "train.txt",
                "vocabulary_source_split": "train",
                "train": "a",
                "validation": "b",
                "test": "c",
            },
            "model": {"name_or_path": "m", "device": "cuda", "dtype": "bfloat16"},
            "lora": {"rank": 8, "alpha": 16, "target_modules": ["down_proj", "up_proj", "gate_proj"]},
            "checkpoints": [
                self._checkpoint("direct43", "primary_direct"),
                self._checkpoint("direct44", "primary_direct"),
                self._checkpoint("more42", "reference_more_qa"),
                self._checkpoint("joint43", "secondary_joint"),
            ],
            "decoders": ["D0", "D1"],
            "evaluation": {
                "splits": ["validation"],
                "initial_checkpoints": ["direct43", "direct44", "more42"],
                "d2_score_normalization_candidates": ["sum", "mean"],
            },
            "follow_up_policy": {
                "D2": "run_validation_only_after_preliminary_go_d2",
                "D3": "diagnostic_only_never_used_for_gate",
                "test": "locked_until_mechanism_gate_and_official_phase_b",
            },
            "gate": {
                "split": "validation",
                "decoder": "D1",
                "primary_checkpoints": ["direct43", "direct44"],
                "reference_checkpoints": ["more42"],
                "minimum_primary_checkpoints": 2,
                "mean_canonical_gain_pp": 2.0,
                "max_mean_novel_composition_drop_pp": 1.0,
                "require_nonnegative_per_checkpoint_raw_and_canonical": True,
            },
        }

    def test_accepts_validation_only_d0_d1_mechanism_protocol(self):
        validate_config(self._config())

    def test_rejects_non_train_graph_vocabulary_source(self):
        config = self._config()
        config["data"]["vocabulary_source_split"] = "test"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_gold_trace_prompt_protocol(self):
        config = self._config()
        config["checkpoints"][0]["prompt_protocol"] = "gold_trace"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_test_in_initial_evaluation_splits(self):
        config = self._config()
        config["evaluation"]["splits"] = ["validation", "test"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_d2_or_d3_in_initial_decoder_list(self):
        config = self._config()
        config["decoders"] = ["D0", "D1", "D2"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_requires_two_primary_direct_checkpoints(self):
        config = self._config()
        config["gate"]["primary_checkpoints"] = ["direct43"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_non_direct_checkpoint_in_primary_gate(self):
        config = self._config()
        config["gate"]["primary_checkpoints"] = ["direct43", "more42"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_initial_checkpoints_require_validation_d0_artifacts(self):
        config = self._config()
        del config["checkpoints"][0]["d0_predictions"]
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
