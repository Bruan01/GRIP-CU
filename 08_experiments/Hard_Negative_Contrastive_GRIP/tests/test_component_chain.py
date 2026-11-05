"""Synthetic fixtures for bounded scheduling; no GPU/model is loaded."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

HNG=Path(__file__).parents[1]
sys.path[:0]=[str(HNG/"scripts"),str(HNG/"src"),str(HNG/"tests")]
import run_fb_violation_only_control as component
import watch_fb_component_chain as watch
from test_violation_only_control import fixture,reference
from hard_negative_grip.counterfactuals import sample_structure_counterfactual_group


def test_profile_only_ignores_b1_scores_and_keeps_structure_signal():
    group,profiles,vocab,parent,pool,params=fixture()
    row,_,diag=component.ablate_group(group,profiles,vocab,parent,pool,params,variant='profile_only')
    for r in group:
        r.update(candidate_score=-999,gold_score=777,pairwise_confusion=.999,score_gap=999)
    alt,_=sample_structure_counterfactual_group(group,profiles,total_k=9,seed=2026,
            known_relations={'gold','known'},**component.settings('profile_only')['sampler'])
    assert row['negative_relations']==alt['negative_relations']
    assert [r['sampling_probability'] for r in row['selected_negatives']]==[r['sampling_probability'] for r in alt['selected_negatives']]
    assert row['structure_weight']==1 and row['violation_weight']==0
    assert row['ablation_variant']=='profile_only' and 'profile_only_structure_mean' in diag
    cmd=component.make_training_argv(reference(),Path('run'),'cache',variant='profile_only')
    assert cmd[cmd.index('--counterfactual_violation_weight')+1]=='0.0'
    assert cmd[cmd.index('--counterfactual_structure_weight')+1]=='1.0'
    assert cmd[cmd.index('--counterfactual_manifest')+1].endswith('profile_only.jsonl')


def plan_fixture(tmp_path):
    queue=tmp_path/'queue';queue.mkdir()
    pre=tmp_path/'v';nxt=tmp_path/'p';pre.mkdir();nxt.mkdir()
    watch.write_json(pre/'status.json',{'stage':'train','status':'running','at':'synthetic'})
    watch.write_json(nxt/'status.json',{'stage':'prepared','status':'ready'})
    watch.write_json(pre/'artifact_manifest.json',{'artifacts':[]})
    plan={'prerequisite_run':str(pre),'next_run':str(nxt),'prerequisite_session':'v','next_session':'p',
          'prerequisite_contract_sha256':'v-contract','next_contract_sha256':'p-contract','repository_root':str(tmp_path/'fake_repo'),
          'prerequisite_id':'SYNTHETIC_V','next_id':'SYNTHETIC_P'}
    return queue,pre,nxt,plan


def test_stalled_log_is_flagged_without_killing_or_launching(tmp_path,monkeypatch):
    q,v,p,plan=plan_fixture(tmp_path)
    log=v/'train.log';log.write_text('1/120 [synthetic progress]\n')
    os.utime(log,(time.time()-1900,time.time()-1900))
    watch.write_json(v/'status.json',{'stage':'train','status':'running','log':str(log)})
    monkeypatch.setattr(watch,'session_alive',lambda name:True)
    monkeypatch.setattr(watch,'launch_next',lambda *args:pytest.fail('must not launch'))
    state=watch.tick(q,plan)
    assert state['stage']=='waiting_violation_only' and 'warning' in state['prerequisite']


@pytest.mark.parametrize('stage,status,alive,expected',[
    ('train','running',True,'waiting_violation_only'),
    ('completed','completed',True,'waiting_violation_controller_exit'),
])
def test_never_launches_before_predecessor_complete_and_exited(tmp_path,monkeypatch,stage,status,alive,expected):
    q,v,p,plan=plan_fixture(tmp_path)
    watch.write_json(v/'status.json',{'stage':stage,'status':status})
    monkeypatch.setattr(watch,'session_alive',lambda name:alive)
    monkeypatch.setattr(watch,'audit_completed',lambda *args:pytest.fail('audit too early'))
    monkeypatch.setattr(watch,'launch_next',lambda *args:pytest.fail('launch too early'))
    assert watch.tick(q,plan)['stage']==expected


@pytest.mark.parametrize('status,alive',[('failed',True),('running',False)])
def test_failed_or_disappeared_predecessor_blocks_continuation(tmp_path,monkeypatch,status,alive):
    q,v,p,plan=plan_fixture(tmp_path)
    watch.write_json(v/'status.json',{'stage':'train','status':status})
    monkeypatch.setattr(watch,'session_alive',lambda name:alive)
    monkeypatch.setattr(watch,'launch_next',lambda *args:pytest.fail('must not launch'))
    with pytest.raises(ValueError):watch.tick(q,plan)


def test_integrity_failure_blocks_profile_even_when_score_is_positive(tmp_path,monkeypatch):
    q,v,p,plan=plan_fixture(tmp_path)
    watch.write_json(v/'status.json',{'stage':'completed','status':'completed'})
    monkeypatch.setattr(watch,'session_alive',lambda name:False)
    def bad(*args):raise ValueError('corrupt archive')
    monkeypatch.setattr(watch,'audit_completed',bad)
    monkeypatch.setattr(watch,'launch_next',lambda *args:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='corrupt'):watch.tick(q,plan)


def test_busy_gpu_waits_after_successful_predecessor_audit(tmp_path,monkeypatch):
    q,v,p,plan=plan_fixture(tmp_path)
    watch.write_json(v/'status.json',{'stage':'completed','status':'completed'})
    monkeypatch.setattr(watch,'session_alive',lambda name:False)
    monkeypatch.setattr(watch,'audit_completed',lambda *args:{'contract_sha256':'v-contract','archive_sha256':watch.sha256(v/'artifact_manifest.json')})
    monkeypatch.setattr(watch,'gpu_idle',lambda:False)
    monkeypatch.setattr(watch,'launch_next',lambda *args:pytest.fail('must wait'))
    assert watch.tick(q,plan)['stage']=='waiting_gpu'
    assert (q/'violation_completion_audit.json').exists()


@pytest.mark.parametrize('cli_finished',[False,True])
def test_launch_is_once_and_incomplete_intent_never_retries(tmp_path,monkeypatch,cli_finished):
    q,v,p,plan=plan_fixture(tmp_path)
    watch.write_json(p/'contract.json',{'synthetic':True})
    plan['next_contract_sha256']=watch.sha256(p/'contract.json')
    watch.write_json(p/'checks/cli.json',{'listed_qa':3109,'n_qa':12136,'model_loaded':False})
    if cli_finished:watch.write_json(p/'status.json',{'stage':'cli_preflight','status':'finished'})
    monkeypatch.setattr(watch,'session_alive',lambda name:False)
    monkeypatch.setattr(watch.base,'verify',lambda path:{'design':{'variant':'profile_only'}})
    monkeypatch.setattr(component,'validate_mode',lambda path:None)
    calls=[]
    monkeypatch.setattr(watch.subprocess,'run',lambda argv,**kw:calls.append(argv))
    one=watch.launch_next(q,plan);two=watch.launch_next(q,plan)
    assert one==two and len(calls)==1 and calls[0][0:2]==['tmux','new-session']
    assert one['status']=='launched'
    one['status']='intent';watch.write_json(q/'launch_receipt.json',one)
    with pytest.raises(ValueError,match='incomplete'):watch.launch_next(q,plan)
    assert len(calls)==1


def test_profile_failure_stops_without_new_gpu_attempt(tmp_path,monkeypatch):
    q,v,p,plan=plan_fixture(tmp_path)
    watch.write_json(v/'status.json',{'stage':'completed','status':'completed'})
    watch.write_json(p/'status.json',{'stage':'train','status':'failed'})
    watch.write_json(q/'violation_completion_audit.json',{'contract_sha256':'v-contract','archive_sha256':watch.sha256(v/'artifact_manifest.json')})
    watch.write_json(q/'launch_receipt.json',{'status':'launched','next_contract_sha256':'p-contract'})
    monkeypatch.setattr(watch,'session_alive',lambda name:False)
    with pytest.raises(ValueError,match='no automatic retry'):watch.tick(q,plan)


def test_four_way_table_preserves_counts_and_component_directions(tmp_path):
    q,v,p,plan=plan_fixture(tmp_path)
    records={'immutable_artifacts':{name:{'sha256':'same'} for name in
        ('inputs/task.json','inputs/test.json','inputs/stage1_adapter/adapter_model.safetensors')}}
    for run in (v,p):watch.write_json(run/'contract.json',records)
    watch.write_json(v/'inputs/test.json',{'id':'fixture','split':'test','graph':{},'questions':[f'q{i}' for i in range(10000)],'answers':['gold']*10000})
    norm=v/'code_snapshot/13_base_method/grip-exp/evaluation/utils.py';norm.parent.mkdir(parents=True)
    norm.write_text('def normalize_answer(s): return s.lower()\n')
    for path,correct in ((v/'reference/predictions.jsonl',8990),(v/'reference/random9_predictions.jsonl',8648),
                         (v/'evaluation/test/predictions.jsonl',8800),(p/'evaluation/test/predictions.jsonl',8700)):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            for i in range(10000):f.write(json.dumps({'question_id':f'fixture:test:{i}','split':'test','question':f'q{i}',
                'target':['gold'],'response':'gold' if i<correct else 'wrong','correct':i<correct})+'\n')
    result=watch.four_way_export(q,plan,{'status':'synthetic_pass'},{'status':'synthetic_pass'})
    assert [r['correct'] for r in result['methods']]==[8648,8700,8800,8990]
    assert result['comparisons']['full_minus_violation_only']['delta_pp']==1.9
    assert result['comparisons']['full_minus_profile_only']['delta_pp']==2.9
    assert (q/'paper/table.tex').read_text().count(chr(92)*2)==5
    assert result['next_action'].endswith('no further training is authorized')


def test_completed_audit_recomputes_flags_and_rejects_archive_corruption(tmp_path):
    """End-to-end CPU audit using synthetic weights and synthetic predictions."""
    run=tmp_path/'synthetic_profile'
    save=lambda rel,value:watch.write_json(run/rel,value)
    ref=reference()
    save('reference/run_config.json',ref)
    save('training/run_config.json',dict(ref,listed_negative_source='structure_counterfactual',
         negative_sampling_mode='structure_counterfactual',counterfactual_sampling=component.settings('profile_only')['sampler'],
         counterfactual_manifest=str(run/'bank/profile_only.jsonl')))
    save('training/trainer_listed/checkpoint-32/trainer_state.json',{'global_step':32,'max_steps':120})
    save('training/listed/adapter/run_metadata.json',{'trainer_global_step':32,'trainer_max_steps':120})
    save('training/listed/adapter/adapter_config.json',{'r':24})
    (run/'training/listed/adapter/adapter_model.safetensors').write_bytes(b'synthetic fixture weights')
    weights={name:watch.sha256(run/'training/listed/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    save('selection.json',{'adapter_sha256':weights})
    save('evaluation/test/reuse_identity.json',{'adapter':weights})
    save('evaluation/test/summary.json',{'all':{'count':10000,'em':.88}})
    save('status.json',{'status':'completed'})
    save('inputs/test.json',{'id':'fixture','split':'test','graph':{},'questions':[f'q{i}' for i in range(10000)],'answers':['gold']*10000})
    norm=run/'code_snapshot/13_base_method/grip-exp/evaluation/utils.py';norm.parent.mkdir(parents=True)
    norm.write_text('def normalize_answer(s): return s.lower()\n')
    predictions=[]
    for relative,correct in (('reference/predictions.jsonl',8990),('reference/random9_predictions.jsonl',8648),('evaluation/test/predictions.jsonl',8800)):
        path=run/relative;path.parent.mkdir(parents=True,exist_ok=True)
        rows=[{'question_id':f'fixture:test:{i}','question':f'q{i}','target':['gold'],'split':'test',
               'response':'gold' if i<correct else 'wrong','correct':i<correct} for i in range(10000)]
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        predictions.append({r['question_id']:r for r in rows})
    metrics,_,_=component.three_way_summary(*predictions,variant='profile_only')
    save('paper/result.json',dict(metrics,status='completed',experiment_id='SYNTHETIC_PROFILE'))
    save('contract.json',{'experiment_id':'SYNTHETIC_PROFILE','design':{'variant':'profile_only','dataset':'fb15k237_2','seed':2026},
                         'immutable_artifacts':{'inputs/test.json':{'sha256':watch.sha256(run/'inputs/test.json')}}})
    contract_hash=watch.sha256(run/'contract.json')
    files=[]
    for rel in ('paper/result.json','training/run_config.json','training/listed/adapter/adapter_model.safetensors'):
        path=run/rel;files.append({'path':rel,'bytes':path.stat().st_size,'sha256':watch.sha256(path)})
    save('artifact_manifest.json',{'artifacts':files})
    result=watch.audit_completed(run,'SYNTHETIC_PROFILE',contract_hash,'profile_only')
    assert result['status']=='pass' and result['correctness_flags_recomputed']==30000
    (run/'training/run_config.json').write_text((run/'training/run_config.json').read_text()+' ')
    with pytest.raises(ValueError,match='archive mismatch'):
        watch.audit_completed(run,'SYNTHETIC_PROFILE',contract_hash,'profile_only')
