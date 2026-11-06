"""Synthetic fixtures; no model measurements and no GPU allocations."""
from copy import deepcopy
from pathlib import Path
import json,sys
import pytest
HNG=Path(__file__).parents[1];sys.path[:0]=[str(HNG/'src'),str(HNG/'scripts')]
import run_fb_k_sensitivity as driver
import watch_fb_rerank_to_k as watcher
from hard_negative_grip.k_sensitivity import FULL_SAMPLER,matched_k_parameters,build_k_group
from hard_negative_grip.counterfactuals import RelationProfile,sample_structure_counterfactual_group
from hard_negative_grip.frozen_random import pool_digest
from hard_negative_grip.matched_control import MATCHED_FIELDS


def fixture():
    vocab=['gold','known']+[f'r{i}' for i in range(24)]
    profiles={n:RelationProfile(n,i+1,i+1,i+2,i+3.,i+4.) for i,n in enumerate(vocab)}
    group=[dict(qa_id='task_qa:0',split='train',candidate_relation=n,matched_train_relation='gold',gold_relation='gold',is_gold=n=='gold',is_valid_negative=n not in ('gold','known'),candidate_score=-i,pairwise_confusion=(i+1)/30) for i,n in enumerate(vocab)]
    parent,_=sample_structure_counterfactual_group(group,profiles,total_k=9,seed=2026,known_relations={'gold','known'},**FULL_SAMPLER)
    legal=sorted(set(vocab)-{'gold','known'});pool=dict(question_id='task_qa:0',legal_relations=legal,excluded_train_relations=['gold','known'],pool_sha256=pool_digest(legal))
    return group,profiles,vocab,parent,pool


@pytest.mark.parametrize('k',[3,15])
def test_exact_full_replay_and_nested_draws(k):
    group,p,v,parent,pool=fixture();r,a,d=build_k_group(group,p,v,parent,pool,k)
    assert len(r['negative_relations'])==len(set(r['negative_relations']))==k
    assert r['structure_weight']==r['violation_weight']==1 and r['uniform_mix']==1/3
    assert d['full_k9_replayed'] and d['nested_prefix_verified']
    assert (r['negative_relations']==parent['negative_relations'][:3]) if k==3 else (r['negative_relations'][:9]==parent['negative_relations'])


@pytest.mark.parametrize('bad',['parent','pool','split','gold','shortfall'])
def test_bad_parent_pool_or_scores_cannot_prepare(bad):
    group,p,v,parent,pool=fixture()
    if bad=='parent':parent['negative_relations']=list(reversed(parent['negative_relations']))
    if bad=='pool':pool['pool_sha256']='bad'
    if bad=='split':group[0]['split']='test'
    if bad=='gold':group[1]['matched_train_relation']='wrong'
    if bad=='shortfall':pool['legal_relations']=pool['legal_relations'][:12];pool['pool_sha256']=pool_digest(pool['legal_relations'])
    with pytest.raises(ValueError):build_k_group(group,p,v,parent,pool,15)


def reference():
    r={k:0 for k in MATCHED_FIELDS};r.update(listed_negative_k=9,target_modules=['up_proj'],seed=2026,stop_after_step=32,involve_qa_epochs=5.,training_arguments={'lr_scheduler_type':'linear'});return r


@pytest.mark.parametrize('k',[3,15])
def test_only_intended_K_changes_training_parameters(k):
    ref=reference();actual=dict(ref,listed_negative_k=k)
    rows=matched_k_parameters(ref,actual,k);assert sum(r['intended_change'] for r in rows)==1
    actual['learning_rate']=.01
    with pytest.raises(ValueError):matched_k_parameters(ref,actual,k)
    cmd=driver.make_training_argv(ref,Path('/run'),'cache',k)
    assert cmd[cmd.index('--listed_negative_k')+1]==str(k)
    assert cmd[cmd.index('--stop_after_step')+1]=='32' and cmd[cmd.index('--involve_qa_epochs')+1]=='5.0'
    assert cmd[cmd.index('--counterfactual_structure_weight')+1]=='1.0' and cmd[cmd.index('--counterfactual_violation_weight')+1]=='1.0'
    assert '--defer_eval' in cmd and '--max_steps' not in cmd


def test_training_cannot_start_before_parent_rerank_completion(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir();driver.write_json(parent/'contract.json',{'synthetic':True});driver.write_json(parent/'status.json',{'status':'running'})
    c=dict(parent_rerank_run=str(parent),parent_rerank_contract_sha256=driver.sha256(parent/'contract.json'))
    called=[];monkeypatch.setattr(driver,'cli_check',lambda *a:called.append(True))
    with pytest.raises(ValueError,match='prerequisite'):driver.train(tmp_path/'child',c)
    assert not called


def test_launch_is_idempotent_and_unresolved_intent_blocks(tmp_path,monkeypatch):
    stage=dict(k=3,contract_sha256='abc',run_dir=str(tmp_path/'run'),session='session')
    watcher.write_json(tmp_path/'launch_k3.json',dict(status='launched',contract_sha256='abc'))
    assert watcher.launch_once(tmp_path,stage)['status']=='launched'
    watcher.write_json(tmp_path/'launch_k3.json',dict(status='intent',contract_sha256='abc'))
    with pytest.raises(ValueError,match='unresolved'):watcher.launch_once(tmp_path,stage)


def test_failed_or_unready_child_is_not_relaunched(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();driver.write_json(run/'contract.json',{'design':{'k':3}});driver.write_json(run/'checks/cli.json',dict(n_qa=12136,listed_qa=3109,k=3,model_loaded=False));driver.write_json(run/'status.json',dict(stage='train',status='failed'))
    stage=dict(k=3,contract_sha256=driver.sha256(run/'contract.json'),run_dir=str(run),session='s')
    monkeypatch.setattr(watcher,'alive',lambda *a:False);monkeypatch.setattr(watcher.base,'verify',lambda *a:{'design':{'k':3}})
    with pytest.raises(ValueError,match='already started/failed'):watcher.launch_once(tmp_path,stage)


def test_parent_completion_audit_is_read_only(tmp_path,monkeypatch):
    from hard_negative_grip.rerank_audit import rank_fields,summary,compare,SCORE_RULE
    parent=tmp_path/'parent';(parent/'inputs').mkdir(parents=True)
    models=['full','frequency_matched','random9'];cohort=[];allrows={};metrics=[];comparisons={};sentinels={}
    for i in range(2):cohort.append(dict(dataset_index=i,question_id=f'q{i}',question=f'Q{i}',target=['r2'],candidates=[f'r{j}' for j in range(10)],gold_index=2,frequency_group='low',train_edge_count=10,direct_correct={m:False for m in models}))
    (parent/'inputs/cohort.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cohort))
    c=dict(model_order=models,expected_questions=2,cohort_sha256='cohort',code_digest='code',base_model_identity={'fixture':True},models={m:dict(adapter_sha256={'fixture':'weights'}) for m in models})
    driver.write_json(parent/'contract.json',c);driver.write_json(parent/'status.json',dict(status='completed'))
    for method in models:
        out=parent/'models'/method;out.mkdir(parents=True);rows=[]
        for i,item in enumerate(cohort):
            raw=[-float(j+1) for j in range(10)];raw[0 if method=='random9' and i==1 else 2]=0.;lengths=[3]*10
            rows.append(dict(item,split='test',raw_candidate_scores=raw,candidate_token_lengths=lengths,prompt_tokens=50,prompt_token_sha256='a'*64,score_rule=SCORE_RULE,**rank_fields(item['candidates'],raw,2,lengths)))
        (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));allrows[method]=rows
        identity=dict(adapter_sha256={'fixture':'weights'},cohort_sha256='cohort',code_digest='code',base_model_identity={'fixture':True},score_rule=SCORE_RULE,score_batch_size=2,dtype='bfloat16',seed=2026,max_length=4096,tie_rule='original_candidate_order')
        driver.write_json(out/'identity.json',identity);s=summary(rows,2);driver.write_json(out/'summary.json',s)
        driver.write_json(out/'completion_audit.json',dict(sentinel='must not rewrite'));sentinels[out/'completion_audit.json']=(out/'completion_audit.json').read_bytes()
        metrics.append(dict(method=method,h1_percent=100*s['h1'],mrr_percent=100*s['mrr']))
    for l,r in [('full','frequency_matched'),('full','random9'),('frequency_matched','random9')]:comparisons[l+'_minus_'+r]=compare(allrows[l],allrows[r],cohort)[0]
    driver.write_json(parent/'paper/result.json',dict(metrics=metrics,comparisons=comparisons));driver.write_json(parent/'artifact_manifest.json',dict(artifacts=[]))
    monkeypatch.setattr(watcher.base,'verify',lambda run:c)
    audit=watcher.audit_parent(parent,driver.sha256(parent/'contract.json'))
    assert audit['read_only_parent_audit'] and audit['candidate_scores_verified']==60
    assert all(p.read_bytes()==raw for p,raw in sentinels.items())


def test_successful_launch_creates_one_durable_receipt(tmp_path,monkeypatch):
    from types import SimpleNamespace
    run=tmp_path/'child';run.mkdir();driver.write_json(run/'contract.json',{'design':{'k':3}});driver.write_json(run/'status.json',dict(stage='prepared',status='ready'));driver.write_json(run/'checks/cli.json',dict(n_qa=12136,listed_qa=3109,k=3,model_loaded=False))
    stage=dict(k=3,contract_sha256=driver.sha256(run/'contract.json'),run_dir=str(run),session='synthetic')
    monkeypatch.setattr(watcher,'alive',lambda *a:False);monkeypatch.setattr(watcher.base,'verify',lambda *a:{'design':{'k':3}});monkeypatch.setattr(watcher.kdriver,'validate_mode',lambda *a:None)
    calls=[];monkeypatch.setattr(watcher.subprocess,'run',lambda argv,**kwargs:calls.append(argv) or SimpleNamespace(returncode=0))
    assert watcher.launch_once(tmp_path,stage)['status']=='launched';assert watcher.launch_once(tmp_path,stage)['status']=='launched'
    assert len(calls)==1 and calls[0][:2]==['tmux','new-session']
    assert driver.read_json(tmp_path/'launch_k3.json')['contract_sha256']==stage['contract_sha256']


def test_parent_failure_blocks_without_starting_a_child(tmp_path,monkeypatch):
    queue=tmp_path/'queue';queue.mkdir();parent=tmp_path/'parent';parent.mkdir();driver.write_json(parent/'status.json',dict(stage='rerank_full',status='failed',error='synthetic failure'))
    plan=dict(stages=[{'k':3},{'k':15}],max_additional_training_runs=2,poll_seconds=30,parent_run=str(parent),parent_session='synthetic_parent',parent_contract_sha256='synthetic')
    driver.write_json(queue/'plan.json',plan);driver.write_json(queue/'code_manifest.json',dict(files=[]));monkeypatch.setattr(sys,'argv',['watch','--plan',str(queue/'plan.json')])
    calls=[];monkeypatch.setattr(watcher,'launch_once',lambda *a:calls.append(True))
    with pytest.raises(ValueError,match='parent failed'):watcher.main()
    assert not calls and driver.read_json(queue/'status.json')['status']=='blocked'


def test_three_point_export_preserves_negative_delta_and_missing_k9_cost(tmp_path):
    from hard_negative_grip.rerank_audit import rank_fields,SCORE_RULE
    import csv
    queue=tmp_path/'queue';queue.mkdir();parent=tmp_path/'parent';(parent/'inputs').mkdir(parents=True)
    cohort=[dict(dataset_index=i,question_id=f'q{i}',question=f'Q{i}',target=['r2'],candidates=[f'r{j}' for j in range(10)],gold_index=2,frequency_group='low',train_edge_count=10,direct_correct={'full':True}) for i in range(2)]
    (parent/'inputs/cohort.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cohort))
    def rows(k):
        result=[]
        for i,item in enumerate(cohort):
            raw=[-float(j+1) for j in range(10)];raw[0 if k==3 and i==1 else 2]=0.;lengths=[3]*10
            result.append(dict(item,split='test',raw_candidate_scores=raw,candidate_token_lengths=lengths,prompt_tokens=50,prompt_token_sha256='a'*64,score_rule=SCORE_RULE,**rank_fields(item['candidates'],raw,2,lengths)))
        return result
    (parent/'models/full').mkdir(parents=True);(parent/'models/full/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows(9)))
    stages=[]
    for k in (3,15):
        run=tmp_path/f'k{k}';out=run/f'ranking/models/sccl_k{k}';out.mkdir(parents=True)
        (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows(k)));driver.write_json(run/'contract.json',dict(design={'k':k}));driver.write_json(run/'paper/result.json',dict(metric=dict(reported_training_seconds=100.,candidate_forwards=10,timing_scope='synthetic',training_timing_scope='synthetic')));stages.append(dict(k=k,run_dir=str(run)))
    plan=dict(parent_run=str(parent),parent_contract_sha256='synthetic',stages=stages,repository_root=str(tmp_path/'fake_repo'));driver.write_json(queue/'plan.json',plan)
    result=watcher.export_three(queue,plan)
    assert [r['k'] for r in result['metrics']]==[3,9,15] and result['comparisons']['k3_minus_k9']['delta_h1_pp']==-50
    with (queue/'paper/training_cost.csv').open() as f:cost=list(csv.DictReader(f))
    assert next(r for r in cost if r['k']=='9')['reported_training_seconds']==''
