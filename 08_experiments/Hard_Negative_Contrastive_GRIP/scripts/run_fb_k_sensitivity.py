#!/usr/bin/env python3
"""One registered Full K=3 or K=15 run; default prepare is CPU only.

Fixed 32/120 optimization budget, one complete matched-list ranking evaluation.
K9 ranking is reused from the preceding, separately frozen rerank campaign.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import shutil,sys
HNG=Path(__file__).resolve().parents[1];ROOT=HNG.parents[1]
sys.path[:0]=[str(HNG/'scripts'),str(HNG/'src')]
import run_fb_random9_control as base
from hard_negative_grip.matched_control import read_json,write_json,write_csv,sha256,training_command
from hard_negative_grip.k_sensitivity import FULL_SAMPLER,matched_k_parameters,build_k_group
from hard_negative_grip.counterfactuals import load_train_relation_profiles
from hard_negative_grip.frozen_random import score_groups
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.task_file import build_qa_assets_from_task_texts,load_counterfactual_manifest,validate_structure_counterfactual_manifest,original_question_ids_from_payload
from hard_negative_grip.rerank_audit import read_rows,summary,compare,SCORE_RULE


def make_training_argv(reference,run,cache,k):
    cmd=training_command(str(base.PYTHON),run/'code_snapshot'/base.REL_HNG/'scripts/train_listed_contrastive.py',reference,inputs=run/'inputs',bank=run/'bank/counterfactual_bank.jsonl',output=run/'training',model_cache=Path(cache))
    cmd[cmd.index('--listed_negative_k')+1]=str(k)
    for flag in ('--listed_negative_source','--negative_sampling_mode'):cmd[cmd.index(flag)+1]='structure_counterfactual'
    cmd[cmd.index('--random_manifest')]='--counterfactual_manifest';cmd+=['--counterfactual_metadata',str(run/'bank/metadata.json')]
    for name,value in FULL_SAMPLER.items():cmd+=['--counterfactual_'+name,str(value)]
    return cmd


def validate_mode(run,c):
    actual=read_json(run/'training/run_config.json');k=c['design']['k']
    base.require(actual['listed_negative_source']=='structure_counterfactual' and actual['negative_sampling_mode']=='structure_counterfactual','wrong training loader')
    base.require(actual['counterfactual_sampling']==FULL_SAMPLER and Path(actual['counterfactual_manifest']).resolve()==(run/'bank/counterfactual_bank.jsonl').resolve(),'wrong Full sampler/bank')
    return matched_k_parameters(read_json(run/'reference/run_config.json'),actual,k)


def cli_check(run,c):
    if (run/'checks/cli.json').exists():
        check=read_json(run/'checks/cli.json');base.require(check['n_qa']==12136 and check['listed_qa']==3109 and check['k']==c['design']['k'] and check['model_loaded'] is False,'invalid K CPU preflight');validate_mode(run,c);return
    (run/'training').mkdir(exist_ok=True)
    program=("import sys,json;from pathlib import Path;import train_listed_contrastive as t;"
             "from hard_negative_grip.k_sensitivity import matched_k_parameters;"
             "from hard_negative_grip.matched_control import read_json,write_json;"
             "c=read_json(Path(sys.argv[1]));sys.argv=c['training_argv'][1:];a=t.parse_args();assets=t.resolve_training_assets(a);"
             "t.write_run_config(Path(a.output_dir),a);run=Path(a.output_dir).parent;"
             "matched_k_parameters(read_json(run/'reference/run_config.json'),read_json(run/'training/run_config.json'),c['design']['k']);"
             "write_json(run/'checks/cli.json',{'n_qa':len(assets[2]),'listed_qa':sum(bool(r['listed_relations']) for r in assets[3]),'k':a.listed_negative_k,'model_loaded':False})")
    base.execute(run,'cli_preflight',[str(base.PYTHON),'-c',program,str(run/'contract.json')]);cli_check(run,c)


def prepare(run,config):
    if (run/'contract.json').exists():
        c=base.verify(run);base.require(c['design_sha256']==sha256(config),'design changed');return c
    base.require(not (run/'inputs').exists(),'partial preparation retained; use fresh run')
    design=read_json(config);k=design['k'];base.require(k in (3,15) and design['seed']==2026 and design['optimizer_steps']==32 and design['scheduler_max_steps']==120,'unregistered K budget')
    source=Path(design['source_random_run']);parent=Path(design['parent_rerank_run'])
    base.require(sha256(source/'contract.json')==design['source_contract_sha256'] and sha256(source/'artifact_manifest.json')==design['source_archive_sha256'],'source control changed')
    base.require(sha256(parent/'contract.json')==design['parent_rerank_contract_sha256'],'parent rerank contract changed')
    sc=read_json(source/'contract.json');pc=read_json(parent/'contract.json');artifacts={};base.append_event(run,dict(stage='prepare',status='running'))
    base.freeze(run,config,'inputs/design.json',artifacts)
    for relative,record in sc['immutable_artifacts'].items():
        if relative=='inputs/design.json' or not relative.startswith(('inputs/','reference/')):continue
        target=run/relative;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to((source/relative).resolve())
        base.require(sha256(target)==record['sha256'],'borrowed input changed')
        artifacts[relative]=dict(sha256=record['sha256'],bytes=record['bytes'],source=str(source/relative),borrowed=True)
    for relative,record in pc['immutable_artifacts'].items():
        if relative.startswith('code_snapshot/'):base.freeze(run,parent/relative,relative,artifacts,record['sha256'])
    for relative in ('scripts/run_fb_k_sensitivity.py','src/hard_negative_grip/k_sensitivity.py'):
        base.freeze(run,HNG/relative,Path('code_snapshot')/base.REL_HNG/relative,artifacts)
    for src,dst in ((parent/'inputs/cohort.jsonl','inputs/cohort.jsonl'),(source/'bank/pool_audit.jsonl','reference/pool_audit.jsonl')):
        base.freeze(run,src,dst,artifacts)
    base.require(sha256(run/'inputs/cohort.jsonl')==pc['cohort_sha256'],'parent cohort hash changed')
    cohort=base.jsonl(run/'inputs/cohort.jsonl');base.require(len(cohort)==10000 and sha256(run/'inputs/test.json')==pc['immutable_artifacts']['inputs/test.json']['sha256'],'test cohort differs from parent ranking')
    original=load_counterfactual_manifest(run/'reference/counterfactual_bank.jsonl');pools={r['question_id']:r for r in base.jsonl(run/'reference/pool_audit.jsonl')}
    base.require(set(original)==set(pools) and len(original)==3109,'K9 bank/pool coverage changed')
    source_wiring=read_json(source/'bank/wiring.json');scores=Path(source_wiring['source_scores']);base.require(sha256(scores)==source_wiring['source_scores_sha256'],'frozen B1 scores changed')
    vocabulary=load_train_relation_order(run/'inputs/raw',relation_text_file=run/'inputs/raw/rel2text.json');profiles=load_train_relation_profiles(run/'inputs/raw',relation_text_file=run/'inputs/raw/rel2text.json')
    bank={};diagnostics=[];(run/'bank').mkdir()
    with (run/'bank/counterfactual_bank.jsonl').open('w') as stream,(run/'bank/sampler_audit.jsonl').open('w') as audit_stream:
        for group in score_groups(scores):
            qid=group[0]['qa_id'];base.require(qid in original and qid not in bank,'unexpected score group')
            row,audit,diagnostic=build_k_group(group,profiles,vocabulary,original[qid],pools[qid],k)
            bank[qid]=row;diagnostics.append(diagnostic);stream.write(json.dumps(row,ensure_ascii=False)+'\n');audit_stream.write(json.dumps(audit,ensure_ascii=False)+'\n')
            if len(bank)%100==0:write_json(run/'status.json',dict(stage='prepare_bank',status='running',rows=len(bank),at=base.now()))
    base.require(set(bank)==set(original),'incomplete K bank');validate_structure_counterfactual_manifest(bank,k=k)
    payload=read_json(run/'inputs/task.json');texts=payload['qa_samples'];ids=original_question_ids_from_payload(payload,len(texts))
    common=dict(seed=2026,relation_order=vocabulary,listed_negative_source='structure_counterfactual',question_ids=ids,eligible_question_ids=set(bank))
    _,before=build_qa_assets_from_task_texts(texts,listed_negative_k=9,counterfactual_manifest=original,**common)
    _,after=build_qa_assets_from_task_texts(texts,listed_negative_k=k,counterfactual_manifest=bank,**common)
    base.require(len(texts)==12136 and sum(bool(r['listed_relations']) for r in after)==3109,'QA eligibility changed')
    for left,right in zip(before,after):base.require({n:v for n,v in left.items() if n!='listed_relations'}=={n:v for n,v in right.items() if n!='listed_relations'},'noncandidate QA metadata/loss changed')
    write_json(run/'bank/metadata.json',dict(source=next(iter(bank.values()))['negative_source'],requested_k=k,parameters=dict(FULL_SAMPLER,seed=2026),rows_written=len(bank),rows_failed=0,source_scores_sha256=source_wiring['source_scores_sha256']))
    write_csv(run/'bank/diagnostics.csv',diagnostics)
    prep=dict(k=k,n_qa=12136,listed_qa=3109,all_k9_rows_exactly_replayed=True,nested_seeded_prefix_rows=len(bank),same_non_candidate_metadata=True,train_only=True,min_legal_pool=min(d['pool_size'] for d in diagnostics),test_options=10,new_model_metrics_available=False,new_training_started=False)
    write_json(run/'bank/preparation_audit.json',prep)
    for path in (run/'bank').iterdir():artifacts[str(path.relative_to(run))]=dict(sha256=sha256(path),bytes=path.stat().st_size)
    code={name:record['sha256'] for name,record in artifacts.items() if name.startswith('code_snapshot/')};code_digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    c=dict(schema='fb-k-sensitivity-v1',experiment_id=design['experiment_id'],created_at=base.now(),repository_root=str(ROOT),design=design,design_sha256=sha256(config),immutable_artifacts=artifacts,code_digest=code_digest,
           model_cache=pc['model_cache'],base_model_identity=pc['base_model_identity'],cohort_sha256=pc['cohort_sha256'],parent_rerank_run=str(parent),parent_rerank_contract_sha256=design['parent_rerank_contract_sha256'],
           training_argv=make_training_argv(read_json(run/'reference/run_config.json'),run,pc['model_cache'],k),expected_optimizer_steps=32,expected_scheduler_max_steps=120,selection='predeclared_step32_before_one_full_candidate_rerank',ranking_only=True,full_validation=False)
    write_json(run/'contract.json',c);write_json(run/'execution_plan.json',dict(training_argv=c['training_argv'],stages=['CPU_preflight','train32','fixed_step32_rerank10000','report'],full_validation=False,new_direct_generation=False))
    write_json(run/'environment_prepare.json',base.environment(run));write_json(run/'status.json',dict(stage='prepared',status='ready',at=base.now(),training_complete=False,ranking_complete=False));base.append_event(run,dict(stage='prepare',status='completed',audit=prep));return c


def validate_training(run,c):
    rows=validate_mode(run,c);state=read_json(run/'training/trainer_listed/checkpoint-32/trainer_state.json');meta=read_json(run/'training/listed/adapter/run_metadata.json')
    base.require(state['global_step']==meta['trainer_global_step']==32 and state['max_steps']==meta['trainer_max_steps']==120,'training budget changed')
    for step in (8,16,24,32):
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','trainer_state.json'):base.require((run/f'training/trainer_listed/checkpoint-{step}'/name).is_file(),'missing checkpoint state')
    write_csv(run/'paper/config_match.csv',rows);return state,meta


def train(run,c):
    parent=Path(c['parent_rerank_run']);base.require(sha256(parent/'contract.json')==c['parent_rerank_contract_sha256'] and read_json(parent/'status.json')['status']=='completed','prerequisite reranking not completed; do not start GPU training')
    cli_check(run,c)
    if (run/'training/listed/adapter/run_metadata.json').exists():validate_training(run,c);return
    for checkpoint in (run/'training/trainer_listed').glob('checkpoint-*'):
        base.require(read_json(checkpoint/'trainer_state.json')['global_step']<=32,'resume exceeds fixed budget')
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors'):base.require((checkpoint/name).is_file(),'incomplete recovery checkpoint')
    base.execute(run,'train',c['training_argv']);validate_training(run,c)


def ranking_view(run,c):
    validate_training(run,c);key='sccl_k'+str(c['design']['k']);view=run/'ranking';view.mkdir(exist_ok=True)
    weights={name:sha256(run/'training/listed/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    selected=dict(at=base.now(),step=32,selection=c['selection'],adapter_sha256=weights,validation_used=False,reopen_after_test=False)
    if (run/'selection.json').exists():base.require(read_json(run/'selection.json')['adapter_sha256']==weights,'selected weights changed')
    else:write_json(run/'selection.json',selected)
    if (view/'contract.json').exists():
        vc=base.verify(view);base.require(vc['models'][key]['adapter_sha256']==weights and vc['code_digest']==c['code_digest'] and vc['cohort_sha256']==c['cohort_sha256'],'ranking view identity changed');return view,key
    (view/'inputs').mkdir();(view/'code_snapshot').symlink_to(run/'code_snapshot',target_is_directory=True);artifacts={name:record for name,record in c['immutable_artifacts'].items() if name.startswith('code_snapshot/')}
    for name in ('cohort.jsonl','test.json'):
        target=view/'inputs'/name;target.symlink_to(run/'inputs'/name);artifacts['inputs/'+name]=dict(sha256=sha256(target),bytes=target.stat().st_size)
    adapter=view/'inputs/adapter';adapter.mkdir()
    for name in weights:
        target=adapter/name;target.symlink_to(run/'training/listed/adapter'/name);artifacts['inputs/adapter/'+name]=dict(sha256=weights[name],bytes=target.stat().st_size)
    vc=dict(schema='fb-k-ranking-view-v1',experiment_id=c['experiment_id']+'_RERANK',repository_root=c['repository_root'],models={key:dict(adapter_dir=str(adapter),adapter_sha256=weights)},immutable_artifacts=artifacts,code_digest=c['code_digest'],cohort_sha256=c['cohort_sha256'],model_cache=c['model_cache'],base_model_identity=c['base_model_identity'],score_batch_size=2,max_length=4096,expected_questions=10000)
    write_json(view/'contract.json',vc);return view,key


def evaluate(run,c):
    parent=Path(c['parent_rerank_run']);base.require(sha256(parent/'contract.json')==c['parent_rerank_contract_sha256'] and read_json(parent/'status.json')['status']=='completed','prerequisite reranking not completed')
    view,key=ranking_view(run,c)
    base.execute(run,'rerank',[str(base.PYTHON),'-u',str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_matched_candidate_rerank.py'),'--run-dir',str(view),'--model',key])


def audit_completed(run,expected_contract=None):
    if expected_contract is not None:base.require(sha256(run/'contract.json')==expected_contract,'K contract changed')
    c=base.verify(run);state,meta=validate_training(run,c);key='sccl_k'+str(c['design']['k']);view=run/'ranking';vc=base.verify(view);out=view/'models'/key
    weights={name:sha256(run/'training/listed/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    identity=read_json(out/'identity.json');base.require(read_json(run/'selection.json')['adapter_sha256']==identity['adapter_sha256']==weights,'ranking weights differ from fixed selection')
    base.require(identity['dtype']=='bfloat16' and identity['seed']==2026 and identity['max_length']==4096 and identity['tie_rule']=='original_candidate_order','ranking inference settings changed')
    base.require(identity['cohort_sha256']==c['cohort_sha256'] and identity['score_rule']==SCORE_RULE and identity['score_batch_size']==2 and identity['code_digest']==c['code_digest'] and identity['base_model_identity']==c['base_model_identity'],'ranking rule/code identity changed')
    cohort=base.jsonl(run/'inputs/cohort.jsonl');rows=read_rows(out/'predictions.jsonl',cohort);base.require(len(rows)==10000,'incomplete K ranking');measured=summary(rows,len(cohort));saved=read_json(out/'summary.json')
    for name,value in measured.items():base.require(saved[name]==value,'K ranking summary mismatch')
    parent=Path(c['parent_rerank_run']);parentrows=read_rows(parent/'models/full/predictions.jsonl',cohort);delta,pairs=compare(rows,parentrows,cohort)
    return c,rows,dict(status='pass',at=base.now(),k=c['design']['k'],rows_verified=len(rows),candidate_scores_verified=10*len(rows),metrics=measured,delta_vs_k9=delta,adapter_sha256=weights),pairs,state,meta


def report(run,c):
    c,rows,audit,pairs,state,meta=audit_completed(run)
    k=c['design']['k'];paper=run/'paper';write_json(paper/'completion_audit.json',audit);write_csv(paper/'paired_k_vs_k9.csv',pairs)
    metric=dict(k=k,n=10000,h1_percent=100*audit['metrics']['h1'],mrr_percent=100*audit['metrics']['mrr'],h3_percent=100*audit['metrics']['h3'],optimizer_steps=32,scheduler_steps=120,reported_training_seconds=meta['seconds'],candidate_forwards=meta.get('candidate_forwards'),training_timing_scope='adapter reported process time; recovery history separately retained')
    write_csv(paper/'metrics.csv',[metric]);fields=sorted({name for r in state['log_history'] for name in r});write_csv(paper/'training_history.csv',[{name:r.get(name,'') for name in fields} for r in state['log_history']])
    result=dict(status='completed',at=base.now(),experiment_id=c['experiment_id'],k=k,seed=2026,metric=metric,delta_vs_k9=audit['delta_vs_k9'],ranking_only=True,new_direct_generation=False,paper_role='supporting',section_id='experiments',item_id='SC_SENS_K',claim_links=['C7'],limitations=['single seed at previously Full-K9-selected hyperparameters','fixed update/schedule budget, not equal candidate FLOPs','ten evaluation options fixed regardless of training K','no direct-generation robustness or across-K frequency-control superiority claim'],next_action='retain all K points and stop')
    write_json(paper/'result.json',result)
    # Full checkpoints, ranking scores and completed child logs are archived; mutable root control logs remain separately retained.
    entries=[]
    for folder in ('inputs','reference','bank','code_snapshot','training','ranking/models','ranking/inputs','paper','logs','checks'):
        for p in sorted((run/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='run.lock':entries.append(dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha256(p)))
    write_json(run/'artifact_manifest.json',dict(at=base.now(),artifacts=entries,policy='retain all models/checkpoints/optimizer/RNG; mutable control logs separate'))
    dest=Path(c['repository_root'])/f'11_paper/artifacts/fb_k{k}_sensitivity_seed2026';dest.mkdir(parents=True,exist_ok=True)
    for p in paper.iterdir():
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'.gitattributes').write_text('* -text\n');write_json(dest/'evidence_index.json',dict(run_dir=str(run),contract_sha256=sha256(run/'contract.json'),archive_sha256=sha256(run/'artifact_manifest.json'),borrowed_sources_must_be_retained=True))
    write_json(run/'status.json',dict(stage='completed',status='completed',at=base.now(),result=result));base.append_event(run,dict(stage='report',status='completed',result=result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','train','evaluate','report','run'],nargs='?',default='prepare');p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path);a=p.parse_args();design=read_json(a.config);run=(a.run_dir or Path(design['default_run_dir'])).resolve()
    base.require(Path(sys.executable).resolve()==base.PYTHON.resolve(),'use guardenv Python')
    with base.locked(run):
        try:
            c=prepare(run,a.config) if a.action in ('prepare','run') else base.verify(run)
            print(json.dumps(dict(action=a.action,run_dir=str(run),k=c['design']['k'])),flush=True)
            if a.action in ('check','run'):cli_check(run,c)
            if a.action in ('train','run'):train(run,c)
            if a.action in ('evaluate','run'):evaluate(run,c)
            if a.action in ('report','run'):print(json.dumps(report(run,c),ensure_ascii=False,indent=2),flush=True)
        except BaseException as error:
            write_json(run/'status.json',dict(stage=a.action,status='failed',at=base.now(),error=repr(error)));base.append_event(run,dict(stage=a.action,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
