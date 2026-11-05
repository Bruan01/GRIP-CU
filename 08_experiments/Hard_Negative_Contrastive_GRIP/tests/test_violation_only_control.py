"""Synthetic implementation fixtures; none are experimental measurements."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HNG = Path(__file__).parents[1]
sys.path[:0] = [str(HNG/"scripts"), str(HNG/"src")]
import run_fb_violation_only_control as control
from hard_negative_grip.counterfactuals import RelationProfile, sample_structure_counterfactual_group
from hard_negative_grip.frozen_random import pool_digest
from hard_negative_grip.matched_control import MATCHED_FIELDS


def fixture():
    vocab = ["gold", "known", *[f"r{i}" for i in range(12)]]
    profiles = {name: RelationProfile(name, (i+1)*5, i+1, i+2, i+1.5, i+2.5) for i,name in enumerate(vocab)}
    group = [{"qa_id": "task_qa:0", "split": "train", "gold_relation": "gold", "matched_train_relation": "gold",
              "candidate_relation": name, "is_gold": name == "gold", "is_valid_negative": name not in ("gold", "known"),
              "candidate_score": -i, "gold_score": 0, "score_gap": i, "pairwise_confusion": (i+1)/20}
             for i,name in enumerate(vocab)]
    params = dict(control.EXPECTED_SAMPLER, structure_weight=1.)
    parent,_ = sample_structure_counterfactual_group(group, profiles, total_k=9, seed=2026, known_relations={"gold","known"}, **params)
    legal = sorted(set(vocab)-{"gold","known"})
    pool = {"question_id": "task_qa:0", "legal_relations": legal, "excluded_train_relations": ["gold","known"],
            "pool_sha256": pool_digest(legal)}
    return group, profiles, vocab, parent, pool, params


def test_zero_structure_coefficient_removes_profile_effect_but_retains_b1_scores():
    group,profiles,vocab,parent,pool,params = fixture()
    row,_,diag = control.ablate_group(group,profiles,vocab,parent,pool,params)
    assert row["structure_weight"] == 0 and row["violation_weight"] == 1
    assert row["uniform_mix"] == 1/3 and row["sampling_temperature"] == .25
    assert diag["original_full_replayed"] and diag["train_only_pool_verified"]
    changed = {name:RelationProfile(name,10**(i+1),100,100,2,3) for i,name in enumerate(vocab)}
    alternate,_ = sample_structure_counterfactual_group(group,changed,total_k=9,seed=2026,
                        known_relations={"gold","known"}, **control.EXPECTED_SAMPLER)
    assert alternate["negative_relations"] == row["negative_relations"]
    assert [r["sampling_probability"] for r in alternate["selected_negatives"]] == [r["sampling_probability"] for r in row["selected_negatives"]]


@pytest.mark.parametrize("bad", ["parent_bank", "pool", "split", "score_flag"])
def test_ablation_refuses_broken_parent_identity_or_filtering(bad):
    group,profiles,vocab,parent,pool,params = fixture()
    if bad == "parent_bank": parent["negative_relations"] = list(reversed(parent["negative_relations"]))
    if bad == "pool": pool["pool_sha256"] = "incorrect"
    if bad == "split": group[0]["split"] = "test"
    if bad == "score_flag": group[1]["is_valid_negative"] = True
    with pytest.raises(ValueError): control.ablate_group(group,profiles,vocab,parent,pool,params)


def test_no_change_gate_avoids_identical_gpu_comparison():
    with pytest.raises(ValueError,match="no candidate sets"):
        control.require_intervention([{"same_set":True}]*3)
    control.require_intervention([{"same_set":True},{"same_set":False}])


def reference():
    ref = {key:0 for key in MATCHED_FIELDS}
    ref.update(dataset="fb15k237_2", seed=2026, target_modules=["up_proj"], stop_after_step=32,
               involve_qa_epochs=5., listed_negative_k=9, training_arguments={"lr_scheduler_type":"linear"})
    return ref


def test_training_command_changes_only_sampler_not_loss_or_schedule():
    cmd = control.make_training_argv(reference(),Path("run"),"cache")
    for flag,value in (("--stop_after_step","32"),("--involve_qa_epochs","5.0"),("--seed","2026"),
                       ("--counterfactual_structure_weight","0.0"),("--counterfactual_violation_weight","1.0")):
        assert cmd[cmd.index(flag)+1] == value
    assert "--max_steps" not in cmd and "--random_manifest" not in cmd
    assert cmd[cmd.index("--listed_negative_source")+1] == "structure_counterfactual"


def test_three_way_summary_has_correct_names_and_directions():
    def predictions(flags):
        return {str(i):{"question_id":str(i),"question":f"q{i}","target":["gold"],"split":"test","correct":b}
                for i,b in enumerate(flags)}
    result,first,second = control.three_way_summary(predictions([True,True,False,True]),
                    predictions([True,False,False,False]),predictions([True,False,True,True]))
    assert result["full_correct"] == 3 and result["violation_only_correct"] == 3
    assert result["random9_correct"] == 1 and result["delta_full_minus_violation_pp"] == 0
    assert result["delta_violation_minus_random_pp"] == 50
    assert "violation_only_correct" in first[0] and "random9_correct" not in first[0]
    assert "random9_correct" in second[0]


def test_evaluate_calls_one_test_and_does_not_call_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(control,"validate_training",lambda *args:None)
    adapter=tmp_path/"training/listed/adapter";adapter.mkdir(parents=True)
    for name in ("adapter_config.json","adapter_model.safetensors"):(adapter/name).write_text("synthetic fixture")
    calls=[]
    monkeypatch.setattr(control.base,"execute",lambda run,stage,argv:calls.append((stage,argv)))
    control.evaluate(tmp_path,{"selection":"fixed_step32_before_one_full_test","model_cache":"cache"})
    assert len(calls)==1 and calls[0][0]=="eval_test"
    assert "--eval_file" in calls[0][1] and "validation.json" not in " ".join(calls[0][1])


@pytest.mark.parametrize("variant", ["violation_only", "profile_only"])
def test_completed_three_row_report_and_cache_identity_guard(tmp_path, variant):
    run=tmp_path/"synthetic_run";save=lambda rel,value:control.write_json(run/rel,value)
    ref=reference()
    save("reference/run_config.json",ref)
    actual=dict(ref,listed_negative_source="structure_counterfactual",negative_sampling_mode="structure_counterfactual",
                counterfactual_sampling=control.settings(variant)["sampler"],counterfactual_manifest=str(run/"bank"/(variant+".jsonl")))
    save("training/run_config.json",actual)
    save("training/trainer_listed/checkpoint-32/trainer_state.json",{"global_step":32,"max_steps":120,"log_history":[{"step":32,"loss":.5}]})
    save("training/listed/adapter/run_metadata.json",{"trainer_global_step":32,"trainer_max_steps":120})
    save("training/listed/adapter/adapter_config.json",{"r":24})
    (run/"training/listed/adapter/adapter_model.safetensors").write_bytes(b"synthetic fixture")
    weights={n:control.sha256(run/"training/listed/adapter"/n) for n in ("adapter_config.json","adapter_model.safetensors")}
    save("selection.json",{"adapter_sha256":weights})
    save("evaluation/test/reuse_identity.json",{"adapter":weights})
    save("evaluation/test/summary.json",{"all":{"count":10000,"em":.887}})
    save("inputs/test.json",{"id":"fixture","split":"test","graph":{},"questions":[f"q{i}" for i in range(10000)],"answers":["gold"]*10000})
    for path,correct in (("reference/predictions.jsonl",8990),("reference/random9_predictions.jsonl",8648),("evaluation/test/predictions.jsonl",8870)):
        p=run/path;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w") as stream:
            for i in range(10000):
                stream.write(json.dumps({"question_id":f"fixture:test:{i}","split":"test","question":f"q{i}","target":["gold"],
                                         "response":"gold" if i<correct else "wrong","correct":i<correct})+"\n")
    norm=run/"code_snapshot/13_base_method/grip-exp/evaluation/utils.py";norm.parent.mkdir(parents=True)
    norm.write_text("def normalize_answer(s): return s.lower()\n")
    c={"experiment_id":"SYNTHETIC_TEST_ONLY","expected_optimizer_steps":32,"expected_scheduler_max_steps":120,
       "design":{"scope":"synthetic fixture", "variant":variant},"validation_policy":"not_run_by_registered_budget_plan",
       "repository_root":str(tmp_path/"fake_repo"),"borrowed_source_run":"synthetic reference"}
    save("contract.json",c)
    result=control.report(run,c)
    short=control.settings(variant)["short"]
    assert result['delta_full_minus_'+short+'_pp']==1.2
    assert result['delta_'+short+'_minus_random_pp']==2.22
    assert result['variant']==variant
    assert (run/('paper/paired_full_vs_'+short+'.csv')).exists()
    assert (run/"paper/table.tex").read_text().count(chr(92)*2)==4
    assert not (run/"evaluation/validation").exists()
    save("evaluation/test/reuse_identity.json",{"adapter":{"wrong":"weights"}})
    with pytest.raises(ValueError,match="identity mismatch"):control.report(run,c)
