"""Synthetic fixtures only: these are not research measurements."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

HNG=Path(__file__).parents[1]
sys.path[:0]=[str(HNG/'src'),str(HNG/'scripts')]
from hard_negative_grip.frequency_matched import matched_bank_row, validate_frequency_manifest, frequency_bin
from hard_negative_grip.task_file import build_qa_assets_from_task_texts
from hard_negative_grip.matched_control import MATCHED_FIELDS
import run_fb_frequency_matched_control as controller


def fixture():
    vocabulary=['gold','known']+[f'r{i}' for i in range(18)]
    rows=[dict(qa_id='task_qa:0',split='train',candidate_relation=n,matched_train_relation='gold',gold_relation='gold',is_gold=n=='gold',is_valid_negative=n not in ('gold','known'),candidate_score=i,pairwise_confusion=.1) for i,n in enumerate(vocabulary)]
    frequencies={n:(8+i%4 if i<11 else 64+i%20) for i,n in enumerate(vocabulary)}
    parent=dict(question_id='task_qa:0',split='train',positive_relation='gold',matched_train_relation='gold',negative_relations=[f'r{i}' for i in (0,1,2,3,4,9,10,11,12)])
    options=dict(relation_order=vocabulary,known_relations={'gold','known'},frequencies=frequencies,parent_row=parent)
    return rows,options


def test_exact_quotas_legal_unique_and_independent_of_model_scores_or_order():
    rows,options=fixture();first,audit=matched_bank_row(rows,**options)
    altered=deepcopy(rows)
    for r in altered:r.update(candidate_score=-1e9,pairwise_confusion=.999,structure_similarity=-999)
    options['relation_order']=list(reversed(options['relation_order']))
    second,again=matched_bank_row(list(reversed(altered)),**options)
    assert first==second and audit==again
    assert first['negative_relations']!=first['parent_full_negative_relations']
    assert len(set(first['negative_relations']))==9
    assert not {'gold','known'}&set(first['negative_relations'])
    assert [frequency_bin(options['frequencies'][n]) for n in first['negative_relations']]==first['parent_slot_bins']
    assert audit['fallback_count']==0 and audit['exact_bin_match']
    validate_frequency_manifest({first['question_id']:first})


@pytest.mark.parametrize('bad',['count','quota','probability','source','split','duplicate','weight','parent_count','bin_size'])
def test_frequency_manifest_rejects_tampering(bad):
    rows,options=fixture();row,_=matched_bank_row(rows,**options)
    if bad=='count':row['selected_negatives'][0]['train_edge_count']=100000
    if bad=='quota':row['selected_negatives'][0]['bin_quota']+=1
    if bad=='probability':row['selected_negatives'][0]['conditional_probability']=.99
    if bad=='source':row['negative_source']='train_only_frozen_uniform_without_replacement'
    if bad=='split':row['filter_splits']=['train','valid']
    if bad=='duplicate':row['negative_relations'][1]=row['negative_relations'][0]
    if bad=='weight':row['lambda_candidate']=9
    if bad=='parent_count':row['parent_train_counts'][0]=1
    if bad=='bin_size':row['selected_negatives'][0]['bin_pool_size']=10000
    with pytest.raises(ValueError):validate_frequency_manifest({row['question_id']:row})


@pytest.mark.parametrize('bad',['parent_illegal','parent_id','parent_duplicate','heldout','wrong_pool','missing_frequency'])
def test_broken_parent_and_pool_never_fallback(bad):
    rows,options=fixture()
    if bad=='parent_illegal':options['parent_row']['negative_relations'][0]='known'
    if bad=='parent_id':options['parent_row']['question_id']='task_qa:9'
    if bad=='parent_duplicate':options['parent_row']['negative_relations'][1]=options['parent_row']['negative_relations'][0]
    if bad=='heldout':rows[0]['split']='test'
    if bad=='wrong_pool':rows[1]['is_valid_negative']=True
    if bad=='missing_frequency':options['frequencies'].pop('r0')
    with pytest.raises(ValueError):matched_bank_row(rows,**options)


def test_frequency_bins_have_fixed_train_count_boundaries():
    assert [frequency_bin(x) for x in (1,2,3,4,7,8,15,16)]==[0,1,1,2,2,3,3,4]
    for x in (0,-1,1.5,True):
        with pytest.raises(ValueError):frequency_bin(x)


def test_real_task_builder_preserves_text_gold_and_loss_eligibility():
    rows,options=fixture();row,_=matched_bank_row(rows,**options)
    qa='<|im_start|>user\nwhat is the relation between a and b?<|im_end|>\n<|im_start|>assistant\n<answer>gold</answer><|im_end|>'
    texts,meta=build_qa_assets_from_task_texts([qa],seed=2026,relation_order=options['relation_order'],listed_negative_source='frequency_matched',random_manifest={'task_qa:0':row},eligible_question_ids={'task_qa:0'})
    assert texts==[qa] and meta[0]['listed_relations']==row['negative_relations']
    assert meta[0]['lambda_candidate'] is None and meta[0]['positive_relation']=='gold'
    with pytest.raises(ValueError,match='exactly'):
        build_qa_assets_from_task_texts([qa],seed=2026,relation_order=options['relation_order'],listed_negative_source='frequency_matched',random_manifest={'task_qa:0':row},eligible_question_ids=set())


def reference():
    ref={k:0 for k in MATCHED_FIELDS}
    ref.update(dataset='fb15k237_2',seed=2026,target_modules=['up_proj'],stop_after_step=32,involve_qa_epochs=5.,listed_negative_k=9,training_arguments={'lr_scheduler_type':'linear'})
    return ref


def test_command_uses_distinct_loader_and_preserves_budget():
    cmd=controller.make_training_argv(reference(),Path('/fixture/run'),'cache')
    for flag,value in (('--listed_negative_source','frequency_matched'),('--negative_sampling_mode','random'),('--stop_after_step','32'),('--involve_qa_epochs','5.0'),('--seed','2026')):
        assert cmd[cmd.index(flag)+1]==value
    assert '--max_steps' not in cmd and '--defer_eval' in cmd
    assert cmd[cmd.index('--save_total_limit')+1]=='4'


def test_evaluate_only_calls_test_and_rejects_changed_selected_model(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'validate_training',lambda *a:None)
    adapter=tmp_path/'training/listed/adapter';adapter.mkdir(parents=True)
    for name in ('adapter_config.json','adapter_model.safetensors'):(adapter/name).write_text('synthetic')
    calls=[];monkeypatch.setattr(controller.base,'execute',lambda run,stage,argv:calls.append((stage,argv)))
    c=dict(selection='fixed',model_cache='cache')
    controller.evaluate(tmp_path,c)
    assert len(calls)==1 and calls[0][0]=='eval_test'
    assert 'validation.json' not in ' '.join(calls[0][1])
    (adapter/'adapter_model.safetensors').write_text('different synthetic weights')
    with pytest.raises(ValueError,match='weights changed'):controller.evaluate(tmp_path,c)


def test_pairwise_directions_and_named_columns():
    def preds(flags):return {str(i):dict(question_id=str(i),question=f'q{i}',target=['gold'],split='test',correct=b) for i,b in enumerate(flags)}
    result,pairs=controller.comparison(preds([True,True,False,False]),preds([True,False,True,False]),'full','frequency_matched')
    assert result['delta_left_minus_right_pp']==0 and result['paired_counts']==dict(both_correct=1,rescue=1,regression=1,both_wrong=1)
    assert 'full_correct' in pairs[0] and 'frequency_matched_correct' in pairs[0]


def test_full_report_audits_predictions_model_identity_and_five_method_exports(tmp_path):
    run=tmp_path/'synthetic_run';save=lambda rel,value:controller.write_json(run/rel,value)
    ref=reference();save('reference/run_config.json',ref)
    save('training/run_config.json',dict(ref,listed_negative_source='frequency_matched',negative_sampling_mode='random',random_manifest=str(run/'bank/frequency_matched.jsonl')))
    save('training/trainer_listed/checkpoint-32/trainer_state.json',dict(global_step=32,max_steps=120,log_history=[dict(step=32,loss=.5)]))
    save('training/listed/adapter/run_metadata.json',dict(trainer_global_step=32,trainer_max_steps=120))
    save('training/listed/adapter/adapter_config.json',dict(r=24))
    (run/'training/listed/adapter/adapter_model.safetensors').write_bytes(b'synthetic fixture')
    weights={n:controller.sha256(run/'training/listed/adapter'/n) for n in ('adapter_config.json','adapter_model.safetensors')}
    save('selection.json',dict(adapter_sha256=weights));save('evaluation/test/reuse_identity.json',dict(adapter=weights))
    save('evaluation/test/summary.json',dict(all=dict(count=10000,em=.887)))
    save('inputs/test.json',dict(id='fixture',split='test',graph={},questions=[f'q{i}' for i in range(10000)],answers=['gold']*10000))
    for rel,correct in [('reference/predictions.jsonl',8990),('reference/random9_predictions.jsonl',8648),('evaluation/test/predictions.jsonl',8870)]:
        p=run/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w') as f:
            for i in range(10000):f.write(json.dumps(dict(question_id=f'fixture:test:{i}',question=f'q{i}',target=['gold'],split='test',response='gold' if i<correct else 'wrong',correct=i<correct))+'\n')
    n=run/'code_snapshot/13_base_method/grip-exp/evaluation/utils.py';n.parent.mkdir(parents=True);n.write_text('def normalize_answer(s): return s.lower()\n')
    old=[dict(question_id=f'fixture:test:{i}',random9_correct=int(i<8648),profile_only_correct=int(i<8489),violation_only_correct=int(i<8709),full_correct=int(i<8990)) for i in range(10000)]
    controller.write_csv(run/'reference/component_paired_four_methods.csv',old)
    save('reference/component_result.json',dict(methods=[dict(method=m,n=10000,correct=v,em_percent=v/100) for m,v in [('Random-9',8648),('Profile-only-9',8489),('Violation-only-9',8709),('Full SC-CL-9',8990)]]))
    save('bank/preparation_audit.json',dict(synthetic=True))
    for name in ('diagnostics.csv','frequency_bin_distribution.csv','relation_frequencies.csv'):controller.write_csv(run/'bank'/name,[dict(synthetic=True)])
    c=dict(experiment_id='SYNTHETIC_ONLY',expected_optimizer_steps=32,expected_scheduler_max_steps=120,design=dict(scope='synthetic fixture'),repository_root=str(tmp_path/'fake_repo'),borrowed_source_run='synthetic')
    save('contract.json',c)
    result=controller.report(run,c)
    assert result['full_minus_frequency_matched']['delta_left_minus_right_pp']==1.2
    assert result['frequency_matched_minus_random9']['delta_left_minus_right_pp']==2.22
    assert (run/'paper/table.tex').read_text().count(chr(92)*2)==6
    assert len((run/'paper/paired_five_methods.csv').read_text().splitlines())==10001
    assert not (run/'evaluation/validation').exists()
    save('evaluation/test/reuse_identity.json',dict(adapter={'wrong':'weights'}))
    with pytest.raises(ValueError,match='identity changed'):controller.report(run,c)
