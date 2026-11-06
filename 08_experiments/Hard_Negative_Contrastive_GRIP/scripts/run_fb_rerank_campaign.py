#!/usr/bin/env python3
"""Freeze and execute exactly three matched FB reranking evaluations; no training."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import shutil,sys,time
HNG=Path(__file__).resolve().parents[1];ROOT=HNG.parents[1]
sys.path[:0]=[str(HNG/'scripts'),str(HNG/'src'),str(ROOT/'13_base_method/grip-exp')]
import run_fb_random9_control as base
from hard_negative_grip.matched_control import read_json,write_json,write_csv,sha256,audit_predictions
from hard_negative_grip.task_file import load_graph_record
from hard_negative_grip.rerank_audit import SCORE_RULE,read_rows,summary,compare,validate_choices
from grip.relation_cecr import parse_relation_candidates

ORDER=['full','frequency_matched','random9']
LABELS=dict(full='Full SC-CL-9',frequency_matched='Full-quota-matched Random-9',random9='Uniform Random-9')


def cache_identity(cache):
    directory=Path(cache)/'Qwen--Qwen2.5-7B-Instruct'
    base.require(directory.is_dir(),'missing offline model cache')
    small={p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.model') and p.stat().st_size<100*1024**2}
    weights={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(directory.glob('*.safetensors'))}
    base.require(small and weights,'incomplete base-model cache identity')
    return dict(path=str(directory.resolve()),config_tokenizer_sha256=small,weight_file_stats=weights,weight_identity_scope='file size/mtime; full base shard hashes not newly computed')


def prepare(run,config):
    if (run/'contract.json').exists():
        c=base.verify(run);base.require(c['design_sha256']==sha256(config),'design changed');return c
    base.require(not (run/'inputs').exists(),'partial preparation preserved; use fresh run directory')
    design=read_json(config);base.require(design['model_order']==ORDER and design['expected_questions']==10000 and design['score_batch_size']==2,'unregistered queue')
    source=Path(design['source_frequency_run']);random=Path(design['source_random_run'])
    for directory,contract_key,archive_key in [(source,'frequency_contract_sha256','frequency_archive_sha256'),(random,'random_contract_sha256','random_archive_sha256')]:
        base.require(sha256(directory/'contract.json')==design[contract_key] and sha256(directory/'artifact_manifest.json')==design[archive_key],'source experiment identity changed')
        base.require(read_json(directory/'status.json')['status']=='completed','source experiment incomplete')
    sc=read_json(source/'contract.json');artifacts={}
    base.append_event(run,dict(stage='prepare',status='running'))
    base.freeze(run,config,'inputs/design.json',artifacts)
    # Exact evaluated model/data/scoring code is reused from the completed control.
    for relative,record in sc['immutable_artifacts'].items():
        if relative.startswith('code_snapshot/'):
            base.freeze(run,source/relative,relative,artifacts,record['sha256'])
    for relative in ('scripts/run_fb_rerank_campaign.py','scripts/evaluate_matched_candidate_rerank.py','src/hard_negative_grip/rerank_audit.py'):
        base.freeze(run,HNG/relative,Path('code_snapshot')/base.REL_HNG/relative,artifacts)
    for relative in ('grip/training/candidate_scorer.py','grip/tasks/eval_tasks/grip_eval.py','grip/tasks/eval_tasks/base.py','grip/relation_cecr.py','constants.py'):
        base.require(sha256(ROOT/base.REL_BASE/relative)==sha256(run/'code_snapshot'/base.REL_BASE/relative),'scorer/prompt code drifted since completed controls')
    base.freeze(run,source/'inputs/test.json','inputs/test.json',artifacts,sc['immutable_artifacts']['inputs/test.json']['sha256'])
    slice_map=ROOT/'11_paper/artifacts/fb_frequency_slices_seed2026/question_slices.csv'
    base.freeze(run,slice_map,'reference/frequency_slices.csv',artifacts)
    import csv
    with slice_map.open() as f:slices={r['question_id']:r for r in csv.DictReader(f)}
    evidence=read_json(source/'reference/evidence.json');full=Path(evidence['retained_adapter'])
    random_archive={r['path']:r for r in read_json(random/'artifact_manifest.json')['artifacts']}
    freq_archive={r['path']:r for r in read_json(source/'artifact_manifest.json')['artifacts']}
    specs={};direct={}
    adapters=dict(full=full,frequency_matched=source/'training/listed/adapter',random9=random/'training/listed/adapter')
    prediction_paths=dict(full=source/'reference/predictions.jsonl',frequency_matched=source/'evaluation/test/predictions.jsonl',random9=random/'evaluation/test/predictions.jsonl')
    normalize=base.normalization(run/'code_snapshot');samples=base.eval_samples(load_graph_record(run/'inputs/test.json'))
    base.require(len(samples)==10000 and {s['question_id'] for s in samples}==set(slices),'cohort/slice map mismatch')
    for method,adapter in adapters.items():
        expected=evidence['audit_identity']['selected_weights'] if method=='full' else {n:(freq_archive if method=='frequency_matched' else random_archive)['training/listed/adapter/'+n]['sha256'] for n in ('adapter_config.json','adapter_model.safetensors')}
        for name in ('adapter_config.json','adapter_model.safetensors'):
            source_file=adapter/name;base.require(sha256(source_file)==expected[name],'source adapter hash changed')
            rel='inputs/adapters/'+method+'/'+name;target=run/rel;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source_file.resolve())
            artifacts[rel]=dict(sha256=expected[name],bytes=target.stat().st_size,source=str(source_file),borrowed=True)
        base.freeze(run,prediction_paths[method],'reference/direct_'+method+'.jsonl',artifacts)
        direct[method]=audit_predictions(base.jsonl(prediction_paths[method]),samples,normalize)
        specs[method]=dict(label=LABELS[method],adapter_dir=str(run/'inputs/adapters'/method),adapter_sha256=expected,borrowed_adapter_source=str(adapter))
    base.require({m:sum(r['correct'] for r in rows.values()) for m,rows in direct.items()}==dict(full=8990,frequency_matched=8983,random9=8648),'direct-generation reference changed')
    cohort=[];collision_audit=[]
    for i,sample in enumerate(samples):
        row=direct['full'][sample['question_id']];candidates=parse_relation_candidates(row['question']);names=[normalize(n) for n in candidates]
        gold_index,collisions=validate_choices(candidates,row['target'],normalize)
        if collisions:collision_audit.append(dict(question_id=sample['question_id'],collisions=collisions,gold_index=gold_index,original_choices_preserved=True))
        sl=slices[sample['question_id']]
        cohort.append(dict(dataset_index=i,question_id=sample['question_id'],question=row['question'],target=row['target'],candidates=candidates,gold_index=gold_index,frequency_group=sl['frequency_group'],train_edge_count=int(sl['train_edge_count']),direct_correct={m:direct[m][sample['question_id']]['correct'] for m in ORDER}))
    (run/'inputs/cohort.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cohort))
    artifacts['inputs/cohort.jsonl']=dict(sha256=sha256(run/'inputs/cohort.jsonl'),bytes=(run/'inputs/cohort.jsonl').stat().st_size)
    write_json(run/'checks/normalized_distractor_collisions.json',dict(count=len(collision_audit),rows=collision_audit,policy='Preserve all original choices; non-gold normalized aliases are not merged; gold unique on every question.'))
    code={key:value['sha256'] for key,value in artifacts.items() if key.startswith('code_snapshot/')}
    digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    model_cache=sc['model_cache'];model_id=cache_identity(model_cache)
    c=dict(schema='fb-matched-rerank-v1',experiment_id=design['experiment_id'],created_at=base.now(),repository_root=str(ROOT),design=design,design_sha256=sha256(config),
           model_order=ORDER,models=specs,expected_questions=10000,cohort_sha256=artifacts['inputs/cohort.jsonl']['sha256'],immutable_artifacts=artifacts,code_digest=digest,
           model_cache=model_cache,base_model_identity=model_id,score_batch_size=2,max_length=4096,score_rule=SCORE_RULE,training_runs=0,
           pilot_policy='first32_of_Full_are_benchmarked_and_reused; no score-based ordering or selection',stop_after_three_models=True)
    write_json(run/'contract.json',c);write_json(run/'environment_prepare.json',base.environment(run))
    write_json(run/'execution_plan.json',dict(order=ORDER,worker_argv={m:worker_argv(run,m) for m in ORDER},additional_training_runs=0,stop_after_report=True))
    prep=dict(n=10000,candidates_per_question=10,models=ORDER,direct_flags_reaudited=30000,same_prompt_option_order_target=True,cohort_frozen=True,pilot_reused=True,scorer_matches_existing_training_and_historical_evaluation=True,normalized_non_gold_collision_questions=len(collision_audit),all_original_options_preserved=True,new_ranking_metrics_available=False,new_training_runs=0)
    write_json(run/'checks/preparation_audit.json',prep);write_json(run/'status.json',dict(stage='prepared',status='ready',at=base.now()))
    base.append_event(run,dict(stage='prepare',status='completed',audit=prep));return c


def worker_argv(run,model):
    return [str(base.PYTHON),'-u',str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_matched_candidate_rerank.py'),'--run-dir',str(run),'--model',model]


def wait_gpu(run):
    while True:
        check=base.capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
        base.require(check.get('returncode')==0,'GPU process check failed')
        if not check['stdout'].strip():return
        write_json(run/'status.json',dict(stage='waiting_gpu',status='running',at=base.now(),processes=check['stdout']));time.sleep(30)


def audit_model(run,c,method,cohort):
    out=run/'models'/method
    identity=read_json(out/'identity.json')
    base.require(identity['adapter_sha256']==c['models'][method]['adapter_sha256'] and identity['cohort_sha256']==c['cohort_sha256'] and identity['code_digest']==c['code_digest'],'result model/cohort/code identity mismatch')
    base.require(identity['base_model_identity']==c['base_model_identity'] and identity['score_rule']==SCORE_RULE and identity['score_batch_size']==2,'result score settings changed')
    base.require(identity['max_length']==4096 and identity['dtype']=='bfloat16' and identity['seed']==2026 and identity['tie_rule']=='original_candidate_order','result inference settings changed')
    rows=read_rows(out/'predictions.jsonl',cohort);base.require(len(rows)==c['expected_questions'],'incomplete model result')
    measured=summary(rows,len(cohort));saved=read_json(out/'summary.json')
    for key,value in measured.items():base.require(saved[key]==value,'result summary disagrees with raw ranks: '+key)
    write_json(out/'completion_audit.json',dict(status='pass',at=base.now(),rows_verified=len(rows),candidate_scores_verified=10*len(rows),metrics=measured,identity=identity))
    return rows


def report(run,c):
    base.verify(run)
    cohort=base.jsonl(run/'inputs/cohort.jsonl');rows={m:audit_model(run,c,m,cohort) for m in ORDER};metrics=[];slices=[];pairs_result={}
    paper=run/'paper';paper.mkdir(exist_ok=True)
    for method in ORDER:
        s=summary(rows[method],len(cohort));metrics.append(dict(method=method,label=LABELS[method],n=s['n'],h1_percent=100*s['h1'],mrr_percent=100*s['mrr'],h3_percent=100*s['h3'],direct_em_percent=100*sum(r['direct_correct'][method] for r in cohort)/len(cohort)))
        for group in ('low','middle','high'):
            subset=[r for r in rows[method] if r['frequency_group']==group];ss=summary(subset,len(subset));slices.append(dict(method=method,frequency_group=group,n=ss['n'],h1_percent=100*ss['h1'] if ss['n'] else None,mrr_percent=100*ss['mrr'] if ss['n'] else None,h3_percent=100*ss['h3'] if ss['n'] else None))
    write_csv(paper/'metrics.csv',metrics);write_csv(paper/'frequency_slice_metrics.csv',slices)
    for left,right in [('full','frequency_matched'),('full','random9'),('frequency_matched','random9')]:
        measured,paired=compare(rows[left],rows[right],cohort);pairs_result[left+'_minus_'+right]=measured;write_csv(paper/('paired_'+left+'_vs_'+right+'.csv'),paired)
    ranks=[dict(question_id=item['question_id'],frequency_group=item['frequency_group'],**{m+'_gold_rank':rows[m][i]['gold_rank'] for m in ORDER},**{m+'_direct_correct':int(item['direct_correct'][m]) for m in ORDER}) for i,item in enumerate(cohort)]
    write_csv(paper/'paired_ranks_and_direct.csv',ranks)
    from collections import Counter
    write_csv(paper/'rank_distribution.csv',[dict(method=m,gold_rank=rank,count=sum(r['gold_rank']==rank for r in rows[m])) for m in ORDER for rank in range(1,11)])
    result=dict(status='completed',at=base.now(),experiment_id=c['experiment_id'],n=len(cohort),seed=2026,score_rule=SCORE_RULE,metrics=metrics,comparisons=pairs_result,
                paper_role='claim-carrying',section_id='experiments',item_id='SC_FB_MATCHED_RERANK',claim_links=['C2'],training_runs=0,
                limitations=['same frozen adapters, one seed and previously Full-selected hyperparameters','reranking and direct generation are separate answer-selection rules','quota-matched random inherits Full per-question training-frequency quotas','ranking evidence cannot erase direct-EM near-null or low-frequency negative results','no score-based metric/protocol/adapter selection'],next_action='write all three ranking rows and stop; NELL and K studies remain planning only')
    write_json(paper/'result.json',result)
    end=chr(92)*2;tex=['\\begin{tabular}{lrrrr}','\\toprule','Method & $n$ & H@1 (\\%) & MRR (\\%) & H@3 (\\%) '+end,'\\midrule']+[f"{r['label']} & {r['n']} & {r['h1_percent']:.4f} & {r['mrr_percent']:.4f} & {r['h3_percent']:.4f} "+end for r in metrics]+['\\bottomrule','\\end{tabular}'];(paper/'table.tex').write_text('\n'.join(tex)+'\n')
    (paper/'RESULTS.md').write_text('# Matched full-test FB candidate reranking\n\nAll three frozen adapters score the same ten candidate continuations on all 10,000 questions. H@1/MRR/H@3 are distinct from prior direct-generation EM. All positive, null, and negative paired differences are retained; no new training or model selection. See result.json and paired CSVs.\n')
    # Completed worker logs and model outputs are fixed; controller stdout/events/status remain live and retained separately.
    files=[]
    for folder in ('inputs','reference','code_snapshot','models','paper','logs','checks'):
        for path in sorted((run/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name!='run.lock':files.append(dict(path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=sha256(path)))
    write_json(run/'artifact_manifest.json',dict(at=base.now(),artifacts=files,policy='no cleanup; mutable root controller/status/events/resources retained outside fixed manifest'))
    destination=Path(c['repository_root'])/'11_paper/artifacts/fb_matched_rerank_seed2026';destination.mkdir(parents=True,exist_ok=True)
    for path in paper.iterdir():
        if path.is_file():shutil.copy2(path,destination/path.name)
    (destination/'.gitattributes').write_text('* -text\n')
    write_json(destination/'evidence_index.json',dict(run_dir=str(run),contract_sha256=sha256(run/'contract.json'),artifact_manifest_sha256=sha256(run/'artifact_manifest.json'),models=c['models'],all_scores_and_token_counts_retained=True,borrowed_sources_must_be_retained=True))
    write_json(run/'status.json',dict(stage='completed',status='completed',at=base.now(),result=result));base.append_event(run,dict(stage='report',status='completed',result=result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','run','report'],default='prepare',nargs='?');p.add_argument('--config',type=Path,default=HNG/'configs/fb_rerank_campaign.json');p.add_argument('--run-dir',type=Path);a=p.parse_args()
    base.require(Path(sys.executable).resolve()==base.PYTHON.resolve(),'use guardenv Python')
    design=read_json(a.config);run=(a.run_dir or Path(design['default_run_dir'])).resolve()
    with base.locked(run):
        try:
            c=prepare(run,a.config) if a.action in ('prepare','run') else base.verify(run)
            print(json.dumps(dict(action=a.action,run_dir=str(run),experiment_id=c['experiment_id'])),flush=True)
            if a.action=='check':
                base.require(len(base.jsonl(run/'inputs/cohort.jsonl'))==10000,'cohort incomplete');print('CPU identities/cohort verified; model not loaded.',flush=True)
            if a.action=='run':
                base.require(cache_identity(c['model_cache'])==c['base_model_identity'],'base cache identity changed')
                for method in ORDER:
                    wait_gpu(run);base.execute(run,'rerank_'+method,worker_argv(run,method));audit_model(run,c,method,base.jsonl(run/'inputs/cohort.jsonl'))
                print(json.dumps(report(run,c),ensure_ascii=False,indent=2),flush=True)
            elif a.action=='report':print(json.dumps(report(run,c),ensure_ascii=False,indent=2),flush=True)
        except BaseException as error:
            write_json(run/'status.json',dict(stage=a.action,status='failed',at=base.now(),error=repr(error)));base.append_event(run,dict(stage=a.action,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
