"""CPU checks for Stage-2-only routing and explicit train-pair loss gating."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HNG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HNG / "scripts"))
sys.path.insert(0, str(HNG / "src"))
import supervise_stage2_three_datasets as campaign
from hard_negative_grip.task_file import build_qa_assets_from_task_texts, load_relation_eligibility


def qa():
    return "<|im_start|>user\nWhat is the relation between a and b?<|im_end|>\n<|im_start|>assistant\n<answer>concept:gold</answer><|im_end|>"


def bank():
    return {"task_qa:0": {"question_id": "task_qa:0", "positive_relation": "concept:gold",
                         "hard_negative_relations": ["concept:negative"], "uniform_negative_relations": [],
                         "negative_relations": ["concept:negative"]}}


class EligibilityTests(unittest.TestCase):
    def test_excluded_relation_stays_in_generation_training_with_text_unchanged(self):
        texts = [qa(), qa()]
        output, meta = build_qa_assets_from_task_texts(texts, seed=2026, listed_negative_k=1,
            listed_negative_source="structure_counterfactual", relation_order=["concept:gold", "concept:negative"],
            counterfactual_manifest=bank(), eligible_question_ids={"task_qa:0"})
        self.assertEqual(output, texts)
        self.assertEqual(meta[0]["listed_relations"], ["concept:negative"])
        self.assertEqual(meta[1]["listed_relations"], [])
        self.assertFalse(meta[1]["candidate_eligible"])

    def test_missing_bank_rows_still_fail_without_explicit_eligibility(self):
        with self.assertRaisesRegex(ValueError, "no row"):
            build_qa_assets_from_task_texts([qa(), qa()], seed=2026, listed_negative_k=1,
                listed_negative_source="structure_counterfactual", relation_order=["concept:gold", "concept:negative"],
                counterfactual_manifest=bank())

    def test_eligibility_cannot_silently_admit_unknown_question_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown|exactly"):
            build_qa_assets_from_task_texts([qa()], seed=2026, listed_negative_k=1,
                listed_negative_source="structure_counterfactual", relation_order=["concept:gold", "concept:negative"],
                counterfactual_manifest=bank(), eligible_question_ids={"task_qa:17"})

    def test_eligibility_is_bound_to_train_and_task_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, train, gate = root / "task.json", root / "train.txt", root / "gate.json"
            task.write_text('{}'); train.write_text('a concept:gold b\n')
            gate.write_text(json.dumps({"train_only": True, "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
                "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(), "eligible_question_ids": ["task_qa:0"]}))
            self.assertEqual(load_relation_eligibility(gate, task, root), {"task_qa:0"})
            train.write_text('x concept:gold y\n')
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_relation_eligibility(gate, task, root)


class CampaignPolicyTests(unittest.TestCase):
    def test_codex_baseline_command_explicitly_skips_stage1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
            (root / "adapter_model.safetensors").write_bytes(b"fixture")
            command = campaign.baseline_command({"dataset": "codexm", "baseline_root": str(root),
                "s1_adapter": str(root), "input": str(root / "tasks.json"), "epochs": 5})
            self.assertEqual(command[command.index("--stage1_epochs") + 1], "0")
            self.assertEqual(command[command.index("--seed") + 1], "2026")

    def test_controller_rejects_accidental_new_stage1_training(self):
        controller = object.__new__(campaign.Controller)
        with self.assertRaisesRegex(ValueError, "Stage-1 training is disabled"):
            controller.run("invalid", [str(campaign.PYTHON), "train_task_file_adapter.py", "--stage1_epochs", "1"])

    def test_registry_has_three_datasets_single_seed_and_only_stage2(self):
        records = campaign.registry(Path('/tmp/stage2-policy-test'))
        self.assertEqual(set(records), {"nell23k", "fb15k237_2", "codexm"})
        for record in records.values():
            self.assertEqual(record["seed"], 2026)
            self.assertEqual(record["search_stage"], 2)
            self.assertFalse(record["stage1_hyperparameters_searched"])

    def test_failed_dataset_is_visible_and_cannot_count_as_all_targets_achieved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "codexm"
            folder.mkdir()
            (folder / "failure.json").write_text(json.dumps({"error": "fixture failure", "phase": "search"}))
            controller = object.__new__(campaign.Controller)
            controller.root, controller.dataset, controller.adopted_pid, controller.action = root, None, None, "fixture"
            with patch.object(campaign.legacy.Supervisor, "snapshot", return_value={"updated_at": "fixture", "latest_log": None}):
                state = controller.snapshot()
            self.assertEqual(state["dataset_results"]["codexm"]["status"], "failed")
            self.assertFalse(state["all_three_targets_achieved"])

    def test_global_success_requires_each_dataset_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in campaign.DATASETS:
                folder = root / dataset / "search"
                folder.mkdir(parents=True)
                (folder / "final_result.json").write_text(json.dumps({"selected_test_em": 0.99,
                    "target_achieved": dataset != "codexm"}))
            controller = object.__new__(campaign.Controller)
            controller.root, controller.dataset, controller.adopted_pid, controller.action = root, None, None, "fixture"
            with patch.object(campaign.legacy.Supervisor, "snapshot", return_value={"updated_at": "fixture", "latest_log": None}):
                self.assertFalse(controller.snapshot()["all_three_targets_achieved"])


if __name__ == "__main__":
    unittest.main()
