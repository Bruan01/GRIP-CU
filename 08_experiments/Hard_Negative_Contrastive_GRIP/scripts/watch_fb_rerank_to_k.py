#!/usr/bin/env python3
"""Monitor one existing rerank campaign, then run K3 and K15 once and stop."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shlex,subprocess,sys,time
HNG=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HNG/'scripts'),str(HNG/'src')]
import run_fb_k_sensitivity as kdriver
from hard_negative_grip.matched_control import read_json,write_json,write_csv,sha256
from hard_negative_grip.rerank_audit import read_rows,summary,compare
base=kdriver.base


def record_status(queue,state):
    write_json(queue/'status.json',state)
    with (queue/'observations.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(state,ensure_ascii=False)+'\n')


def alive(name):return subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0


def idle():
    apps=base.capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']);mem=base.capture(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'])
    return apps.get('returncode')==mem.get('returncode')==0 and not apps['stdout'].strip() and max(float(v) for v in mem['stdout'].splitlines())<4500


def parent_observation(run):
    state=read_json(run/'status.json');out={key:state.get(key) for key in ('stage','status','at','error')};progress={}
    for method in ('full','frequency_matched','random9'):
        p=run/'models'/method/'heartbeat.json'
        if p.exists():
            h=read_json(p);progress[method]={key:h.get(key) for key in ('status','completed','expected','questions_per_second','remaining_seconds_this_model','at')}
            if h.get('status')=='running' and time.time()-p.stat().st_mtime>1800:progress[method]['warning']='No scoring progress for >30min; no auto-retry or next launch.'
        else:progress[method]={'status':'not_started'}
    out['models']=progress;return out


def audit_parent(run,expected_contract):
    base.require(read_json(run/'status.json')['status']=='completed','parent incomplete');base.require(sha256(run/'contract.json')==expected_contract,'parent contract changed')
    c=base.verify(run);cohort=base.jsonl(run/'inputs/cohort.jsonl');rows={}
    for name in c['model_order']:
        out=run/'models'/name;identity=read_json(out/'identity.json')
        base.require(identity['adapter_sha256']==c['models'][name]['adapter_sha256'] and identity['cohort_sha256']==c['cohort_sha256'] and identity['code_digest']==c['code_digest'],'parent result identity mismatch')
        base.require(identity['base_model_identity']==c['base_model_identity'] and identity['score_rule']==kdriver.SCORE_RULE and identity['score_batch_size']==2 and identity['dtype']=='bfloat16' and identity['seed']==2026 and identity['max_length']==4096 and identity['tie_rule']=='original_candidate_order','parent scoring settings changed')
        rows[name]=read_rows(out/'predictions.jsonl',cohort);base.require(len(rows[name])==c['expected_questions'],'parent model incomplete')
        measured=summary(rows[name],len(cohort));saved=read_json(out/'summary.json')
        for field,value in measured.items():base.require(saved[field]==value,'parent summary mismatch')
    r=read_json(run/'paper/result.json')
    for left,right in (('full','frequency_matched'),('full','random9'),('frequency_matched','random9')):
        measured,_=compare(rows[left],rows[right],cohort);base.require(measured==r['comparisons'][left+'_minus_'+right],'parent paired ranking result changed')
    for metric in r['metrics']:
        measured=summary(rows[metric['method']],len(cohort));base.require(abs(metric['h1_percent']-100*measured['h1'])<1e-12 and abs(metric['mrr_percent']-100*measured['mrr'])<1e-12,'parent metrics not reproduced')
    manifest=read_json(run/'artifact_manifest.json')
    for entry in manifest['artifacts']:
        p=run/entry['path'];base.require(p.is_file() and p.stat().st_size==entry['bytes'],'parent archive missing/size changed: '+entry['path'])
    return dict(status='pass',at=base.now(),contract_sha256=expected_contract,archive_sha256=sha256(run/'artifact_manifest.json'),questions_verified=sum(map(len,rows.values())),candidate_scores_verified=10*sum(map(len,rows.values())),metrics=r['metrics'],artifact_sizes_verified=len(manifest['artifacts']),read_only_parent_audit=True)


def launch_once(queue,stage):
    run=Path(stage['run_dir']);receipt_path=queue/f"launch_k{stage['k']}.json"
    if receipt_path.exists():
        receipt=read_json(receipt_path);base.require(receipt['contract_sha256']==stage['contract_sha256'] and receipt['status']=='launched','unresolved/conflicting launch intent; no blind retry');return receipt
    base.require(not alive(stage['session']),'unowned K session exists')
    base.require(sha256(run/'contract.json')==stage['contract_sha256'],'K source contract changed');c=base.verify(run);check=read_json(run/'checks/cli.json')
    state=read_json(run/'status.json');ready=(state.get('stage')=='prepared' and state.get('status')=='ready') or (state.get('stage')=='cli_preflight' and state.get('status')=='finished')
    base.require(ready and not list((run/'training/trainer_listed').glob('checkpoint-*')) and not (run/'training/listed/adapter').exists(),'K run already started/failed; do not duplicate')
    base.require(c['design']['k']==stage['k'] and check==dict(n_qa=12136,listed_qa=3109,k=stage['k'],model_loaded=False),'CPU K preflight mismatch');kdriver.validate_mode(run,c)
    argv=[str(base.PYTHON),'-u',str(run/'code_snapshot'/base.REL_HNG/'scripts/run_fb_k_sensitivity.py'),'run','--config',str(run/'inputs/design.json'),'--run-dir',str(run)]
    receipt=dict(at=base.now(),status='intent',contract_sha256=stage['contract_sha256'],session=stage['session'],argv=argv);write_json(receipt_path,receipt)
    command='exec '+shlex.join(argv)+' > '+shlex.quote(str(run/'controller.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',stage['session'],'-c',str(run),command],check=True)
    receipt.update(status='launched',launched_at=base.now());write_json(receipt_path,receipt);base.append_event(queue,dict(stage='launch_k'+str(stage['k']),**receipt));return receipt


def audit_k(run,contract_sha):
    base.require(read_json(run/'status.json')['status']=='completed','K run incomplete')
    c,rows,audit,pairs,state,meta=kdriver.audit_completed(run,contract_sha)
    result=read_json(run/'paper/result.json');base.require(result['delta_vs_k9']==audit['delta_vs_k9'],'K result comparison changed')
    manifest=read_json(run/'artifact_manifest.json')
    for entry in manifest['artifacts']:
        p=run/entry['path'];base.require(p.is_file() and p.stat().st_size==entry['bytes'],'K archive missing or changed size')
    audit.update(contract_sha256=contract_sha,archive_sha256=sha256(run/'artifact_manifest.json'),artifact_sizes_verified=len(manifest['artifacts']));return audit


def export_three(queue,plan):
    parent=Path(plan['parent_run']);cohort=base.jsonl(parent/'inputs/cohort.jsonl');points={9:read_rows(parent/'models/full/predictions.jsonl',cohort)};metrics=[];comparisons={};cost=[]
    for stage in plan['stages']:
        run=Path(stage['run_dir']);c=read_json(run/'contract.json');key='sccl_k'+str(stage['k']);points[stage['k']]=read_rows(run/'ranking/models'/key/'predictions.jsonl',cohort)
        result=read_json(run/'paper/result.json');meta=result['metric'];cost.append(dict(k=stage['k'],reported_training_seconds=meta['reported_training_seconds'],candidate_forwards=meta['candidate_forwards'],timing_scope=meta['training_timing_scope']))
    cost.append(dict(k=9,reported_training_seconds=None,candidate_forwards=None,timing_scope='reused historical K9; total resumed training cost not remeasured'))
    paper=queue/'paper';paper.mkdir(exist_ok=True)
    for k in (3,9,15):
        s=summary(points[k],len(cohort));base.require(s['status']=='completed','K point incomplete');metrics.append(dict(k=k,n=s['n'],h1_percent=100*s['h1'],mrr_percent=100*s['mrr'],h3_percent=100*s['h3']))
        if k!=9:
            comp,pairs=compare(points[k],points[9],cohort);comparisons[f'k{k}_minus_k9']=comp;write_csv(paper/f'paired_k{k}_vs_k9.csv',pairs)
    write_csv(paper/'metrics.csv',metrics);write_csv(paper/'training_cost.csv',sorted(cost,key=lambda r:r['k']));write_csv(paper/'paired_three_k.csv',[dict(question_id=item['question_id'],frequency_group=item['frequency_group'],**{f'k{k}_gold_rank':points[k][i]['gold_rank'] for k in (3,9,15)}) for i,item in enumerate(cohort)])
    result=dict(status='completed',at=base.now(),experiment_id='SC_FB_K_SENSITIVITY_SEED2026',seed=2026,metrics=metrics,comparisons=comparisons,training_runs=2,reused_k9=True,evaluation_options=10,ranking_only=True,section_id='experiments',item_id='SC_SENS_K',paper_role='supporting',claim_links=['C7'],limitations=['one seed, fixed Full-K9-selected hyperparameters','fixed optimizer updates, not equal candidate FLOPs','no generation robustness or cross-K superiority over quota-matched controls','K9 training cost not remeasured; report missing cost explicitly'],next_action='write all three K points and stop; no NELL or additional seed/HPO run')
    write_json(paper/'result.json',result);end=chr(92)*2
    (paper/'table.tex').write_text('\n'.join(['\\begin{tabular}{rrrrr}','\\toprule','$K$ & $n$ & H@1 (\\%) & MRR (\\%) & H@3 (\\%) '+end,'\\midrule']+[f"{r['k']} & {r['n']} & {r['h1_percent']:.4f} & {r['mrr_percent']:.4f} & {r['h3_percent']:.4f} "+end for r in metrics]+['\\bottomrule','\\end{tabular}'])+'\n')
    (paper/'RESULTS.md').write_text('# Full SC-CL training-negative K sensitivity\n\nK3 and K15 each use one seed and fixed32/120 optimization, evaluated on the same original ten test options and scoring rule as reused K9. All K points and negative/null differences are retained. This is ranking sensitivity only, not direct-generation robustness or superiority to frequency controls across K. K9 total resumed training cost is not remeasured.\n')
    dest=Path(plan['repository_root'])/'11_paper/artifacts/fb_k_sensitivity_seed2026';dest.mkdir(parents=True,exist_ok=True)
    for p in paper.iterdir():
        if p.is_file():shutil_copy(p,dest/p.name)
    (dest/'.gitattributes').write_text('* -text\n');write_json(dest/'evidence_index.json',dict(queue_dir=str(queue),plan_sha256=sha256(queue/'plan.json'),parent_contract_sha256=plan['parent_contract_sha256'],new_training_runs=2,all_models_logs_checkpoints_scores_retained=True))
    return result


def shutil_copy(src,dst):
    import shutil
    shutil.copy2(src,dst)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();queue=a.plan.parent
    with base.locked(queue):
        plan=read_json(a.plan);base.require([s['k'] for s in plan['stages']]==[3,15] and plan['max_additional_training_runs']==2 and plan['poll_seconds']==30,'unbounded/changed K plan')
        for entry in read_json(queue/'code_manifest.json')['files']:base.require(sha256(queue/entry['path'])==entry['sha256'],'watcher code changed')
        parent=Path(plan['parent_run']);parent_audited=(queue/'parent_completion_audit.json').exists()
        if parent_audited:base.require(read_json(queue/'parent_completion_audit.json')['contract_sha256']==plan['parent_contract_sha256'],'cached parent audit belongs to another contract')
        try:
            while True:
                observation=parent_observation(parent)
                if not parent_audited:
                    base.require(observation['status']!='failed','parent failed; no K continuation')
                    if observation['status']=='completed' and not alive(plan['parent_session']):
                        record_status(queue,dict(stage='audit_parent',status='running',at=base.now()));audit=audit_parent(parent,plan['parent_contract_sha256']);write_json(queue/'parent_completion_audit.json',audit);parent_audited=True
                    else:
                        base.require(alive(plan['parent_session']) or observation['status']=='completed','parent session vanished before completion')
                        record_status(queue,dict(stage='watching_rerank',status='running',at=base.now(),parent=observation));time.sleep(30);continue
                completed=[];waiting=False
                for stage in plan['stages']:
                    run=Path(stage['run_dir']);state=read_json(run/'status.json');audit_path=queue/f"k{stage['k']}_completion_audit.json"
                    if state.get('status')=='completed' and not alive(stage['session']):
                        if not audit_path.exists():write_json(audit_path,audit_k(run,stage['contract_sha256']))
                        base.require(read_json(audit_path)['contract_sha256']==stage['contract_sha256'],'cached K audit belongs to another contract')
                        completed.append(stage['k']);continue
                    base.require(state.get('status')!='failed','K experiment failed; no retry/continuation')
                    if state.get('status')=='running' and state.get('log') and Path(state['log']).exists() and time.time()-Path(state['log']).stat().st_mtime>1800:
                        state['warning']='No child log progress >30min; no automatic retry or budget change.'
                    receipt=queue/f"launch_k{stage['k']}.json"
                    if receipt.exists():
                        base.require(read_json(receipt)['status']=='launched','unresolved launch intent; inspect before retry')
                        base.require(alive(stage['session']) or state.get('status')=='completed','K session vanished before completion')
                    elif idle():launch_once(queue,stage)
                    record_status(queue,dict(stage='running_k'+str(stage['k']) if receipt.exists() else 'waiting_gpu_for_k'+str(stage['k']),status='running',at=base.now(),completed_k=completed,experiment_state=state));waiting=True;break
                if not waiting:
                    result=export_three(queue,plan);record_status(queue,dict(stage='completed',status='completed',at=base.now(),result=result));base.append_event(queue,dict(stage='report',status='completed',result=result));return
                time.sleep(30)
        except BaseException as error:
            record_status(queue,dict(stage='blocked',status='blocked',at=base.now(),error=repr(error),policy='no automatic retry or budget expansion'));base.append_event(queue,dict(stage='blocked',status='blocked',error=repr(error)));raise


if __name__=='__main__':main()
