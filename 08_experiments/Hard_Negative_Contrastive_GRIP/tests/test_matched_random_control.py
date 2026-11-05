from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HNG = Path(__file__).parents[1]
sys.path.insert(0, str(HNG/"src"))
from hard_negative_grip.frozen_random import score_groups, uniform_bank_row, validate_frozen_random_manifest
from hard_negative_grip.matched_control import (
    MATCHED_FIELDS, audit_predictions, paired_result, parameter_diff, training_command,
)
from hard_negative_grip.task_file import build_qa_assets_from_task_texts
from hard_negative_grip.training_budget import assert_listed_training_budget


def group(qid="task_qa:0"):
    vocabulary = ["gold", "known", *[f"rel{i}" for i in range(14)]]
    return vocabulary, [
        {"qa_id": qid, "split": "train", "candidate_relation": name,
         "matched_train_relation": "gold", "gold_relation": "gold",
         "is_gold": name == "gold", "is_valid_negative": name not in ("gold", "known"),
         "candidate_score": i, "pairwise_confusion": .99 if i % 2 else .01}
        for i, name in enumerate(vocabulary)]


def bank():
    vocabulary, rows = group()
    row, audit = uniform_bank_row(rows, relation_order=vocabulary, known_relations={"gold", "known"})
    return vocabulary, row, audit


def test_uniform_selection_does_not_use_scores_features_or_dump_order():
    vocab, rows = group()
    first, audit = uniform_bank_row(rows, relation_order=vocab, known_relations={"gold", "known"})
    altered = deepcopy(rows)
    for row in altered:
        row.update(candidate_score=-999, pairwise_confusion=0, structure_similarity=100)
    second, again = uniform_bank_row(list(reversed(altered)), relation_order=list(reversed(vocab)),
                                    known_relations={"gold", "known"})
    assert first == second
    assert audit == again
    assert len(first["negative_relations"]) == len(set(first["negative_relations"])) == 9
    assert not {"gold", "known"} & set(first["negative_relations"])
    assert set(audit["legal_relations"]) == set(vocab)-{"gold", "known"}


@pytest.mark.parametrize("change", ["validation", "pool_flag", "duplicate", "shortfall"])
def test_uniform_bank_refuses_illegal_parent_pools(change):
    vocab, rows = group()
    if change == "validation": rows[0]["split"] = "valid"
    if change == "pool_flag": rows[1]["is_valid_negative"] = True
    if change == "duplicate": rows[-1] = deepcopy(rows[-2])
    known = {"gold", "known"}
    if change == "shortfall": known |= set(vocab[2:10])
    with pytest.raises(ValueError):
        uniform_bank_row(rows, relation_order=vocab, known_relations=known)


@pytest.mark.parametrize("change", ["seed_provenance", "weight", "split", "duplicate", "probability"])
def test_manifest_rejects_tampering(change):
    _, row, _ = bank()
    if change == "seed_provenance": row["negative_source"] = "structure_counterfactual"
    if change == "weight": row["lambda_candidate"] = 99
    if change == "split": row["filter_splits"] = ["train", "test"]
    if change == "duplicate": row["negative_relations"][1] = row["negative_relations"][0]
    if change == "probability": row["selected_negatives"][0]["sampling_probability"] = .9
    with pytest.raises(ValueError):
        validate_frozen_random_manifest({row["question_id"]: row})


def test_frozen_bank_attaches_without_changing_qa_or_generation_prefix():
    vocab, row, _ = bank()
    qa = "<|im_start|>user\nwhat is the relation between a and b?<|im_end|>\n<|im_start|>assistant\n<answer>gold</answer><|im_end|>"
    texts, metas = build_qa_assets_from_task_texts([qa], seed=2026, relation_order=vocab,
        listed_negative_source="frozen_random", random_manifest={"task_qa:0": row},
        eligible_question_ids={"task_qa:0"})
    assert texts == [qa]
    assert metas[0]["prefix_text"].endswith("<answer>")
    assert metas[0]["positive_relation"] == "gold"
    assert metas[0]["lambda_candidate"] is None
    assert metas[0]["listed_relations"] == row["negative_relations"]
    with pytest.raises(ValueError, match="exactly"):
        build_qa_assets_from_task_texts([qa], seed=2026, relation_order=vocab,
            listed_negative_source="frozen_random", random_manifest={"task_qa:0": row},
            eligible_question_ids=set())


def test_large_score_dump_can_be_streamed_and_repeated_groups_fail(tmp_path):
    _, a = group("task_qa:0")
    _, b = group("task_qa:1")
    p = tmp_path/"scores.jsonl"
    p.write_text("\n".join(json.dumps(row) for row in a+b))
    assert [len(g) for g in score_groups(p)] == [len(a), len(b)]
    p.write_text("\n".join(json.dumps(row) for row in a+b+a))
    with pytest.raises(ValueError, match="duplicate"):
        list(score_groups(p))


def test_training_contract_preserves_120_step_schedule_and_32_step_stop():
    ref = {key: 0 for key in MATCHED_FIELDS}
    ref.update(dataset="fb15k237_2", seed=2026, target_modules=["up_proj", "down_proj"],
               stop_after_step=32, involve_qa_epochs=5.0, listed_negative_k=9,
               s1_gradient_checkpointing=True, training_arguments={"lr_scheduler_type": "linear"})
    cmd = training_command("guardenv/python", Path("train.py"), ref,
                           inputs=Path("inputs"), bank=Path("bank.jsonl"), output=Path("out"), model_cache=Path("cache"))
    assert cmd[cmd.index("--stop_after_step")+1] == "32"
    assert cmd[cmd.index("--involve_qa_epochs")+1] == "5.0"
    assert cmd[cmd.index("--seed")+1] == "2026"
    assert "--max_steps" not in cmd
    assert cmd[cmd.index("--listed_negative_source")+1] == "frozen_random"
    assert "--no_resume" not in cmd
    assert cmd[cmd.index("--save_total_limit")+1] == "4"
    assert all(row["equal"] for row in parameter_diff(ref, dict(ref)))
    bad = dict(ref, lora_r=24)
    with pytest.raises(ValueError, match="lora_r"):
        parameter_diff(ref, bad)
    budget = assert_listed_training_budget(12136, listed_negative_source="frozen_random", epochs=5)
    assert budget["total_steps"] == 120


def sample_and_prediction(qid, correct):
    return ({"question_id": qid, "split": "test", "question": ["q"+qid, "unused"], "answer": "gold"},
            {"question_id": qid, "split": "test", "question": "q"+qid, "target": ["gold"],
             "response": "gold" if correct else "wrong", "correct": correct})


def test_prediction_audit_and_direction_are_sccl_minus_random():
    samples, sc, rnd = [], [], []
    for i, (s, r) in enumerate([(True, True), (True, False), (True, False), (False, True), (False, False)]):
        sample, prediction = sample_and_prediction(str(i), s)
        samples.append(sample); sc.append(prediction)
        rnd.append(sample_and_prediction(str(i), r)[1])
    left, right = [audit_predictions(rows, samples, str.lower) for rows in (sc, rnd)]
    result, pairs = paired_result(left, right)
    assert result["net_correct"] == 1
    assert result["delta_sccl_minus_random9_pp"] == 20
    assert result["paired_counts"] == {"both_correct": 1, "rescue": 2, "regression": 1, "both_wrong": 1}
    assert len(pairs) == 5
    stale = deepcopy(rnd); stale[0]["correct"] = False
    with pytest.raises(ValueError, match="flag mismatch"):
        audit_predictions(stale, samples, str.lower)
    with pytest.raises(ValueError, match="incomplete"):
        audit_predictions(rnd[:-1], samples, str.lower)
    mismatch = deepcopy(right); mismatch["0"]["question"] = "new prompt"
    with pytest.raises(ValueError, match="question mismatch"):
        paired_result(left, mismatch)


def load_controller():
    spec = importlib.util.spec_from_file_location("fb_random_controller", HNG/"scripts/run_fb_random9_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_identity_gate_rejects_changed_input_code_or_bank(tmp_path):
    controller = load_controller()
    p = tmp_path/"bank.jsonl"; p.write_text("frozen")
    controller.write_json(tmp_path/"contract.json", {"immutable_artifacts": {"bank.jsonl": {"sha256": controller.sha256(p)}}})
    controller.verify(tmp_path)
    p.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        controller.verify(tmp_path)


def test_existing_run_lock_prevents_duplicate_launch(tmp_path):
    controller = load_controller()
    with controller.locked(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with controller.locked(tmp_path):
                pass


def test_prelaunch_code_refresh_is_forbidden_once_training_started(tmp_path):
    controller = load_controller()
    (tmp_path/"training/trainer_listed/checkpoint-8").mkdir(parents=True)
    with pytest.raises(ValueError, match="after training"):
        controller.refresh_code(tmp_path, {"immutable_artifacts": {}})


def test_paper_report_end_to_end_and_rejects_other_model_predictions(tmp_path):
    """Small synthetic fixtures test export logic, never research performance."""
    c = load_controller()
    run = tmp_path/"fixture_run"
    def save(relative, value):
        c.write_json(run/relative, value)
    ref = {key: 0 for key in MATCHED_FIELDS}
    ref.update(target_modules=["up_proj"], training_arguments={"lr_scheduler_type": "linear"})
    save("reference/run_config.json", ref)
    save("training/run_config.json", dict(ref, listed_negative_source="frozen_random", negative_sampling_mode="random"))
    save("training/trainer_listed/checkpoint-32/trainer_state.json",
         {"global_step": 32, "max_steps": 120, "log_history": [{"step": 32, "loss": .5, "generation_loss": .4, "candidate_loss": .1}]})
    save("training/listed/adapter/run_metadata.json", {"trainer_global_step": 32, "trainer_max_steps": 120})
    save("training/listed/adapter/adapter_config.json", {"r": 24})
    (run/"training/listed/adapter/adapter_model.safetensors").write_bytes(b"synthetic test fixture")
    fingerprints = {name: c.sha256(run/"training/listed/adapter"/name)
                    for name in ("adapter_config.json", "adapter_model.safetensors")}
    save("selection.json", {"adapter_sha256": fingerprints})
    for split in ("validation", "test"):
        record = {"id": "fixture", "split": split, "title": "fixture", "graph": {},
                  "questions": ["q0", "q1"], "answers": ["gold", "gold"]}
        save("inputs/"+split+".json", record)
        preds = [{"question_id": f"fixture:{split}:{i}", "split": split, "question": f"q{i}",
                  "target": ["gold"], "response": "gold" if i == 0 else "wrong", "correct": i == 0} for i in range(2)]
        out=run/"evaluation"/split; out.mkdir(parents=True)
        (out/"predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds)+"\n")
        save("evaluation/"+split+"/reuse_identity.json", {"adapter": fingerprints})
        save("evaluation/"+split+"/summary.json", {"all": {"count": 2, "em": .5}})
        if split == "test":
            preds[1].update(response="gold", correct=True)
            (run/"reference/predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds)+"\n")
    save("reference/evidence.json", {"selected_correct": 2})
    norm = run/"code_snapshot/13_base_method/grip-exp/evaluation/utils.py"
    norm.parent.mkdir(parents=True); norm.write_text("def normalize_answer(s): return s.lower()\n")
    contract = {"expected_optimizer_steps": 32, "expected_scheduler_max_steps": 120,
                "experiment_id": "SYNTHETIC_TEST_ONLY", "repository_root": str(tmp_path/"fake_repo")}
    save("contract.json", contract)
    result = c.report(run, contract)
    assert result["delta_sccl_minus_random9_pp"] == 50
    assert (run/"paper/training_history.csv").exists()
    assert (run/"paper/table.tex").read_text().count("\\\\") == 3
    assert (tmp_path/"fake_repo/11_paper/artifacts/fb_random9_control_seed2026/evidence_index.json").exists()
    save("evaluation/test/reuse_identity.json", {"adapter": {"adapter_model.safetensors": "different weights"}})
    with pytest.raises(ValueError, match="different weights"):
        c.report(run, contract)
