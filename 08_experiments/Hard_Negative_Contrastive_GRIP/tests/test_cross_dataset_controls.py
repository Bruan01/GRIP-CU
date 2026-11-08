"""Scheduling, comparability and original 11-option WN contracts; no GPU."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

HNG = Path(__file__).parents[1]
sys.path[:0] = [str(HNG/'src'), str(HNG/'scripts')]
import run_cross_dataset_controls as driver
import watch_cross_dataset_controls as watcher
from hard_negative_grip.matched_control import MATCHED_FIELDS, write_json, sha256
from hard_negative_grip.rerank_audit import validate_choices, rank_fields, audit_row, read_rows, compare, SCORE_RULE


def design(dataset='wn18rr'):
    vals = (125,125,12334,2709,3134,11,1000) if dataset=='wn18rr' else (32,240,12014,2974,4943,10,32)
    d = dict(zip(('optimizer_steps','scheduler_max_steps','qa_count','listed_qa_count',
                  'test_count','candidate_count','generation_max_length'), vals))
    return dict(d, dataset=dataset, seed=2026, k=9, arms=list(driver.ARMS),
                max_new_training_runs=2, model_cache='/offline', save_steps=25 if dataset=='wn18rr' else 8)


@pytest.mark.parametrize('dataset',['wn18rr','nell23k'])
def test_original_budgets_and_null_outcome_reporting_do_not_expand_runs(dataset):
    d = design(dataset)
    assert driver.checked_design(d) == d
    for field, value in [('k',3),('seed',2027),('max_new_training_runs',3),('scheduler_max_steps',120),('generation_max_length',17)]:
        bad = dict(d, **{field:value})
        with pytest.raises(ValueError): driver.checked_design(bad)


@pytest.mark.parametrize('dataset',['wn18rr','nell23k'])
def test_generated_commands_preserve_training_budget_and_frozen_source(tmp_path,dataset):
    d = design(dataset)
    arm = tmp_path/'frequency_matched'
    (arm/'inputs').mkdir(parents=True)
    ref = {k:0 for k in MATCHED_FIELDS}
    ref.update(target_modules=['up_proj'], s1_gradient_checkpointing=True, seed=2026,
               stop_after_step=d['optimizer_steps'], involve_qa_epochs=5. if dataset=='wn18rr' else 10.)
    cmd = driver.make_training_argv(ref, arm, d)
    assert cmd[cmd.index('--stop_after_step')+1] == str(d['optimizer_steps'])
    assert cmd[cmd.index('--involve_qa_epochs')+1] == str(ref['involve_qa_epochs'])
    assert cmd[cmd.index('--listed_negative_source')+1] == 'frequency_matched'
    assert cmd[cmd.index('--random_manifest')+1] == str(tmp_path/'bank/frequency_matched.jsonl')
    assert '--relation_eligibility' not in cmd and '--relation_text_file' not in cmd
    from hard_negative_grip.run_protection import checkpoint_training_kwargs
    from source_checkpoint_retention import retention_limit
    limit=int(cmd[cmd.index('--save_total_limit')+1])
    assert limit==retention_limit(d)
    effective=checkpoint_training_kwargs(d['save_steps'],limit)
    assert effective['save_total_limit']>=d['optimizer_steps']//d['save_steps']
    assert '--max_steps' not in cmd


def test_mode_guard_rejects_hyperparameter_or_bank_changes(tmp_path):
    ref = {k:0 for k in MATCHED_FIELDS}
    ref.update(target_modules=['up_proj'], memory_size=0, training_arguments={'bf16':True})
    arm = tmp_path/'random9'
    actual = dict(ref, listed_negative_source='frozen_random', negative_sampling_mode='random',
                  random_manifest=str(tmp_path/'bank/random9.jsonl'), listed_negative_seed=2026)
    assert driver.verify_cli(ref, actual, 'random9', arm)
    for key,value in [('learning_rate',.1),('random_manifest',str(tmp_path/'other.jsonl')),('listed_negative_source','frequency_matched')]:
        with pytest.raises(ValueError): driver.verify_cli(ref, dict(actual,**{key:value}), 'random9', arm)


def test_only_candidates_and_source_may_change_in_qa_metadata():
    left = dict(question_id='q0',positive_relation='gold',listed_relations=['a'],
                listed_negative_source='structure_counterfactual',lambda_candidate=None,
                prefix_text='same prompt',candidate_eligible=True)
    right = dict(left,listed_relations=['b'],listed_negative_source='frozen_random')
    driver.match_qa_metadata([left],[right])
    for key,value in [('lambda_candidate',.25),('candidate_eligible',False),('prefix_text','changed')]:
        with pytest.raises(ValueError): driver.match_qa_metadata([left],[dict(right,**{key:value})])


def test_wn_frequency_profile_uses_the_same_human_labels_as_train_vocabulary(tmp_path):
    from hard_negative_grip.counterfactuals import load_train_relation_profiles
    from hard_negative_grip.official_lists import load_train_relation_order
    (tmp_path/'train.txt').write_text('h\t_hypernym\tt\nh\t_hypernym\tu\nv\t_also_see\tw\n')
    vocab=load_train_relation_order(tmp_path)
    profiles=load_train_relation_profiles(tmp_path,strip_first_underscore=True)
    assert set(profiles)==set(vocab)=={'hypernym','also see'}
    assert profiles['hypernym'].edge_count==2


def test_unverified_pairs_are_not_silently_treated_as_train_verified():
    group=[dict(head_entity=None,tail_entity=None,is_gold=True,matched_train_relation='gold',candidate_relation='gold',is_valid_negative=False),
           dict(head_entity=None,tail_entity=None,is_gold=False,matched_train_relation='gold',candidate_relation='neg',is_valid_negative=True)]
    known,status=driver.group_known_relations(group,{},'wn18rr')
    assert known=={'gold'} and status=='unverified_pair_original_parent_mask'
    with pytest.raises(ValueError): driver.group_known_relations(group,{},'nell23k')
    bad=deepcopy(group);bad[1]['head_entity']='invented'
    with pytest.raises(ValueError): driver.group_known_relations(bad,{},'wn18rr')
    bad=deepcopy(group);bad[1]['is_valid_negative']=False
    with pytest.raises(ValueError): driver.group_known_relations(bad,{},'wn18rr')


def test_parent_mask_boolean_is_not_presented_as_independent_train_fact_proof():
    row=dict(selected_negatives=[dict(train_known_absent=True)])
    driver.mark_pair_scope(row,'unverified_pair_original_parent_mask')
    assert row['train_pair_verified'] is False
    assert row['selected_negatives'][0]['train_known_absent_independently_verified'] is False
    assert 'unverified' in row['selected_negatives'][0]['train_known_absent_proof_scope']


def eleven_row():
    names = [f'r{i}' for i in range(11)]
    item = dict(dataset_index=0,question_id='q0',question='original WN prompt',target=['r10'],
                candidates=names,gold_index=10,frequency_group='low',train_edge_count=10,direct_correct={'full':False})
    row = dict(item,split='test',raw_candidate_scores=[-float(i) for i in range(11)],
               candidate_token_lengths=[3]*11,prompt_tokens=20,prompt_token_sha256='a'*64,score_rule=SCORE_RULE,
               **rank_fields(names,[-float(i) for i in range(11)],10,[3]*11,expected_candidates=11))
    return item,row


def test_wn_keeps_all_eleven_choices_and_does_not_award_hits10_to_rank11(tmp_path):
    item,row = eleven_row()
    gold,aliases = validate_choices(item['candidates'],['r10'],str.lower,expected_candidates=11)
    assert gold==10 and not aliases and row['gold_rank']==11 and not row['hits10']
    assert audit_row(row,item,expected_candidates=11)==row
    with pytest.raises(ValueError): audit_row(row,item)  # legacy FB default stays ten
    path=tmp_path/'predictions.jsonl';path.write_text(json.dumps(row)+'\n')
    assert read_rows(path,[item],expected_candidates=11)==[row]
    result,pairs=compare([row],[row],[item],expected_candidates=11)
    assert result['net_correct']==0 and result['delta_mrr_pp']==0
    changed=deepcopy(row);changed['raw_candidate_scores']=changed['raw_candidate_scores'][:10]
    with pytest.raises(ValueError): audit_row(changed,item,expected_candidates=11)


def test_original_ten_option_default_and_stable_ties():
    r=rank_fields([str(i) for i in range(10)],[-1.]*10,9,[1]*10)
    assert r['gold_rank']==10 and r['hits10'] and r['response']=='0'


def test_queue_waits_for_fb_and_never_touches_gpu_while_waiting(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'status.json',{'status':'running'})
    plan=dict(parent_queue=str(parent),parent_session='fb')
    monkeypatch.setattr(watcher.previous,'alive',lambda name:True)
    monkeypatch.setattr(watcher.previous,'idle',lambda:pytest.fail('GPU polling/launch before prerequisite'))
    assert watcher.cycle(tmp_path,plan) is False
    assert driver.read_json(tmp_path/'status.json')['stage']=='waiting_fb_k'
    write_json(parent/'status.json',{'status':'blocked'})
    with pytest.raises(ValueError,match='failed/blocked'): watcher.cycle(tmp_path,plan)


def test_disappeared_fb_session_blocks_instead_of_skipping(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'status.json',{'status':'running'})
    monkeypatch.setattr(watcher.previous,'alive',lambda name:False)
    with pytest.raises(ValueError,match='vanished'):
        watcher.cycle(tmp_path,dict(parent_queue=str(parent),parent_session='fb'))


def test_launch_receipts_are_idempotent_but_ambiguous_intents_block(tmp_path):
    stage=dict(dataset='nell23k',run_dir=str(tmp_path/'unused'),contract_sha256='frozen')
    write_json(tmp_path/'launch_nell23k.json',dict(status='launched',contract_sha256='frozen'))
    assert watcher.launch_once(tmp_path,stage)['status']=='launched'
    write_json(tmp_path/'launch_nell23k.json',dict(status='intent',contract_sha256='frozen'))
    with pytest.raises(ValueError,match='unresolved'): watcher.launch_once(tmp_path,stage)


def test_child_failure_stops_before_second_dataset(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'status.json',{'status':'completed'})
    write_json(tmp_path/'parent_completion_audit.json',dict(parent_plan_sha256='abc'))
    child=tmp_path/'nell';child.mkdir();write_json(child/'status.json',{'status':'failed'})
    plan=dict(parent_queue=str(parent),parent_session='fb',parent_plan_sha256='abc',
              stages=[dict(dataset='nell23k',run_dir=str(child),session='nell')])
    monkeypatch.setattr(watcher.previous,'alive',lambda name:False)
    monkeypatch.setattr(watcher,'launch_once',lambda *args:pytest.fail('launch after failed run'))
    with pytest.raises(ValueError,match='dataset failed'): watcher.cycle(tmp_path,plan)


def test_direct_driver_cannot_bypass_prerequisite(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'plan.json',{'fixture':True})
    write_json(parent/'status.json',{'status':'running'})
    c=dict(design=dict(prerequisite_queue=str(parent),prerequisite_plan_sha256=sha256(parent/'plan.json')))
    with pytest.raises(ValueError,match='prerequisite'): driver.prerequisite(c)


def test_busy_gpu_does_not_launch_a_ready_dataset(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'status.json',{'status':'completed'})
    write_json(tmp_path/'parent_completion_audit.json',dict(parent_plan_sha256='abc'))
    child=tmp_path/'nell';child.mkdir();write_json(child/'status.json',{'status':'ready'})
    plan=dict(parent_queue=str(parent),parent_session='fb',parent_plan_sha256='abc',
              stages=[dict(dataset='nell23k',run_dir=str(child),session='nell')])
    monkeypatch.setattr(watcher.previous,'alive',lambda name:False)
    monkeypatch.setattr(watcher.previous,'idle',lambda:False)
    monkeypatch.setattr(watcher,'launch_once',lambda *args:pytest.fail('launch on busy GPU'))
    assert watcher.cycle(tmp_path,plan) is False
    assert driver.read_json(tmp_path/'status.json')['stage']=='waiting_gpu_nell23k'


def test_completed_two_dataset_queue_stops_without_any_extra_launch(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'status.json',{'status':'completed'})
    write_json(tmp_path/'parent_completion_audit.json',dict(parent_plan_sha256='abc'))
    stages=[]
    for ds in ('nell23k','wn18rr'):
        child=tmp_path/ds;child.mkdir();write_json(child/'status.json',{'status':'completed'})
        write_json(tmp_path/(ds+'_completion_audit.json'),dict(contract_sha256=ds))
        stages.append(dict(dataset=ds,run_dir=str(child),session=ds,contract_sha256=ds))
    plan=dict(parent_queue=str(parent),parent_session='fb',parent_plan_sha256='abc',stages=stages)
    monkeypatch.setattr(watcher.previous,'alive',lambda name:False)
    monkeypatch.setattr(watcher,'launch_once',lambda *args:pytest.fail('extra launch after completion'))
    assert watcher.cycle(tmp_path,plan) is True
    assert driver.read_json(tmp_path/'result.json')['datasets']==['nell23k','wn18rr']
    assert driver.read_json(tmp_path/'status.json')['status']=='completed'


def test_storage_guard_checks_real_output_volume_instead_of_virtual_free_space(tmp_path,monkeypatch):
    run=tmp_path/'run';storage=tmp_path/'physical';storage.mkdir();arm=run/'random9';arm.mkdir(parents=True)
    target=storage/'random9';target.mkdir();(arm/'training').symlink_to(target)
    stage=dict(run_dir=str(run),training_storage_root=str(storage),training_targets={'random9':str(target)},minimum_training_storage_free_bytes=100)
    plan=dict(host_backing_volume=str(tmp_path/'host'),minimum_host_backing_free_bytes=10)
    monkeypatch.setattr(watcher,'shutil_free',lambda p:200)
    watcher.check_training_storage(stage,plan)
    monkeypatch.setattr(watcher,'shutil_free',lambda p:50 if p==storage else 10000)
    with pytest.raises(ValueError,match='physical-volume'):watcher.check_training_storage(stage,plan)
    monkeypatch.setattr(watcher,'shutil_free',lambda p:200)
    stage['training_targets']['random9']=str(storage/'wrong')
    with pytest.raises(ValueError,match='redirect'):watcher.check_training_storage(stage,plan)
