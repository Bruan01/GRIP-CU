#!/usr/bin/env python3
"""One fixed-budget frequency-bin matched control; default prepare is CPU only."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys

HNG=Path(__file__).resolve().parents[1]
ROOT=HNG.parents[1]
sys.path[:0]=[str(HNG/'scripts'),str(HNG/'src')]
import run_fb_random9_control as base
from hard_negative_grip.counterfactuals import load_train_relation_profiles, sample_structure_counterfactual_group, score_violation, relation_profile_similarity
from hard_negative_grip.frequency_matched import matched_bank_row, frequency_bin, validate_frequency_manifest
from hard_negative_grip.frozen_random import score_groups, pool_digest
from hard_negative_grip.matched_control import audit_predictions, paired_result, parameter_diff, read_json, sha256, training_command, write_csv, write_json
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.task_file import build_qa_assets_from_task_texts, load_counterfactual_manifest, load_graph_record, original_question_ids_from_payload

require=base.require
VARIANT='frequency_matched'
LABEL='Frequency-matched Random-9'


def make_training_argv(reference,run,model_cache):
    cmd=training_command(str(base.PYTHON),run/'code_snapshot'/base.REL_HNG/'scripts/train_listed_contrastive.py',reference,
                         inputs=run/'inputs',bank=run/'bank/frequency_matched.jsonl',output=run/'training',model_cache=Path(model_cache))
    cmd[cmd.index('--listed_negative_source')+1]=VARIANT
    return cmd


def validate_mode(run):
    actual=read_json(run/'training/run_config.json')
    require(actual['listed_negative_source']==VARIANT and actual['negative_sampling_mode']=='random','wrong frequency control loader')
    require(Path(actual['random_manifest']).resolve()==(run/'bank/frequency_matched.jsonl').resolve(),'wrong frozen frequency bank')
    parameter_diff(read_json(run/'reference/run_config.json'),actual)


def prepare(run,config):
    if (run/'contract.json').exists():
        c=base.verify(run)
        require(c['design_sha256']==sha256(config),'design changed after freeze')
        return c
    require(not (run/'inputs').exists(),'partial preparation retained; use fresh directory')
    design=read_json(config)
    require(design['variant']==VARIANT and design['seed']==2026 and design['k']==9
            and design['optimizer_steps']==32 and design['scheduler_max_steps']==120
            and design['frequency_bin_rule']=='floor_log2_train_edge_count','unregistered experiment design')
    source=Path(design['source_control_run'])
    require(sha256(source/'contract.json')==design['source_contract_sha256'],'parent contract changed')
    require(sha256(source/'artifact_manifest.json')==design['source_archive_sha256'],'parent archive changed')
    require(read_json(source/'status.json')['status']=='completed','parent control incomplete')
    sc=base.verify(source)
    base.append_event(run,{'stage':'prepare','status':'running'})
    artifacts={}
    base.freeze(run,config,'inputs/design.json',artifacts)
    # Reuse immutable data, and copy the exact parent executable layout.
    overrides={str(base.REL_HNG/p) for p in ('scripts/train_listed_contrastive.py','src/hard_negative_grip/task_file.py')}
    for relative,record in sc['immutable_artifacts'].items():
        if relative=='inputs/design.json' or not relative.startswith(('inputs/','reference/','code_snapshot/')): continue
        if relative.startswith('code_snapshot/') and str(Path(relative).relative_to('code_snapshot')) in overrides: continue
        target=run/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if relative.startswith('code_snapshot/'): shutil.copy2(source/relative,target)
        else: target.symlink_to((source/relative).resolve())
        artifacts[relative]={'sha256':record['sha256'],'bytes':record['bytes'],'source':str(source/relative),'borrowed':not relative.startswith('code_snapshot/')}
    # These changes add a new manifest loader; sampler/loss/optimizer code is unchanged.
    for relative in ('scripts/train_listed_contrastive.py','src/hard_negative_grip/task_file.py',
                     'src/hard_negative_grip/frequency_matched.py','scripts/run_fb_frequency_matched_control.py'):
        base.freeze(run,HNG/relative,Path('code_snapshot')/base.REL_HNG/relative,artifacts)
    archive={r['path']:r for r in read_json(source/'artifact_manifest.json')['artifacts']}
    for src,dst in (('evaluation/test/predictions.jsonl','reference/random9_predictions.jsonl'),
                    ('bank/pool_audit.jsonl','reference/pool_audit.jsonl')):
        target=run/dst;target.symlink_to((source/src).resolve())
        require(sha256(target)==archive[src]['sha256'],'parent comparator changed')
        artifacts[dst]={'sha256':archive[src]['sha256'],'bytes':target.stat().st_size,'source':str(source/src),'borrowed':True}
    for name in ('result.json','paired_four_methods.csv'):
        base.freeze(run,ROOT/'11_paper/artifacts/fb_component_ablation_seed2026'/name,'reference/component_'+name,artifacts,
                    design['component_reference_sha256'][name])
    parent=load_counterfactual_manifest(run/'reference/counterfactual_bank.jsonl')
    pools={r['question_id']:r for r in base.jsonl(run/'reference/pool_audit.jsonl')}
    require(set(pools)==set(parent) and len(parent)==design['listed_qa_count'],'parent cohort changed')
    params=read_json(run/'reference/metadata.json')['parameters'].copy();params.pop('seed')
    require(params==dict(uniform_mix=1/3,structure_weight=1.,violation_weight=1.,temperature=.25),'parent sampler changed')
    wiring=read_json(source/'bank/wiring.json');scores=Path(wiring['source_scores'])
    require(sha256(scores)==wiring['source_scores_sha256'],'B1 audit scores changed')
    vocabulary=load_train_relation_order(run/'inputs/raw',relation_text_file=run/'inputs/raw/rel2text.json')
    profiles=load_train_relation_profiles(run/'inputs/raw',relation_text_file=run/'inputs/raw/rel2text.json')
    frequencies={n:p.edge_count for n,p in profiles.items()}
    bank={};diagnostics=[];distribution=Counter();selected_plot=[]
    (run/'bank').mkdir()
    write_csv(run/'bank/relation_frequencies.csv',[dict(relation=n,train_edge_count=frequencies[n],frequency_bin=frequency_bin(frequencies[n])) for n in sorted(frequencies)])
    with (run/'bank/frequency_matched.jsonl').open('w') as output,(run/'bank/frequency_audit.jsonl').open('w') as audit_out:
        for group in score_groups(scores):
            qid=group[0]['qa_id'];require(qid in parent and qid not in bank,'unexpected/duplicate score group')
            pool=pools[qid];known=set(pool['excluded_train_relations'])
            require(pool_digest(pool['legal_relations'])==pool['pool_sha256'],'parent pool identity mismatch')
            replay,fa=sample_structure_counterfactual_group(group,profiles,total_k=9,seed=2026,known_relations=known,**params)
            require(replay['negative_relations']==parent[qid]['negative_relations'],'Full bank failed exact replay')
            row,audit=matched_bank_row(group,relation_order=vocabulary,known_relations=known,frequencies=frequencies,parent_row=parent[qid],seed=2026,k=9)
            require(row['pool_sha256']==pool['pool_sha256'],'legal pool changed')
            bank[qid]=row;output.write(json.dumps(row,ensure_ascii=False)+'\n');audit_out.write(json.dumps(audit,ensure_ascii=False)+'\n')
            score_index={r['candidate_relation']:r for r in group};gold=row['matched_train_relation']
            diagnostic={'question_id':qid,'same_set':set(row['negative_relations'])==set(parent[qid]['negative_relations']),
                        'overlap_k':len(set(row['negative_relations'])&set(parent[qid]['negative_relations'])),
                        'exact_slot_bins_match':audit['exact_bin_match'],'fallback_count':audit['fallback_count'],
                        'forced_slot_count':audit['forced_slot_count'],'pool_size':row['pool_size']}
            for method,names in (('full',parent[qid]['negative_relations']),('frequency_matched',row['negative_relations'])):
                vals=[]
                for slot,name in enumerate(names):
                    values=dict(question_id=qid,method=method,slot=slot,relation=name,train_edge_count=frequencies[name],
                                frequency_bin=frequency_bin(frequencies[name]),log2_train_frequency=math.log2(frequencies[name]),
                                structure_similarity=relation_profile_similarity(profiles[gold],profiles[name]),
                                b1_score_violation=score_violation(score_index[name],temperature=.25))
                    selected_plot.append(values);vals.append(values);distribution[(method,values['frequency_bin'])]+=1
                for metric in ('train_edge_count','log2_train_frequency','structure_similarity','b1_score_violation'):
                    diagnostic[method+'_mean_'+metric]=sum(v[metric] for v in vals)/9
            diagnostics.append(diagnostic)
            if len(bank)%100==0:write_json(run/'status.json',{'stage':'prepare_bank','status':'running','rows':len(bank),'at':base.now()})
    require(set(bank)==set(parent),'incomplete matched bank')
    validate_frequency_manifest(bank,k=9)
    changed=sum(not d['same_set'] for d in diagnostics)
    require(changed>0,'frequency control changed no candidate sets; no GPU run')
    write_csv(run/'bank/diagnostics.csv',diagnostics)
    write_csv(run/'bank/selected_candidates.csv',selected_plot)
    write_csv(run/'bank/frequency_bin_distribution.csv',[dict(method=m,frequency_bin=b,selected_count=n) for (m,b),n in sorted(distribution.items())])
    payload=read_json(run/'inputs/task.json');texts=payload['qa_samples'];ids=original_question_ids_from_payload(payload,len(texts))
    common=dict(seed=2026,relation_order=vocabulary,question_ids=ids,eligible_question_ids=set(bank))
    before_text,before=build_qa_assets_from_task_texts(texts,listed_negative_source='structure_counterfactual',counterfactual_manifest=parent,**common)
    after_text,after=build_qa_assets_from_task_texts(texts,listed_negative_source=VARIANT,random_manifest=bank,**common)
    require(before_text==after_text and len(texts)==design['qa_count'],'training QA changed')
    for left,right in zip(before,after):
        require({k:v for k,v in left.items() if k not in ('listed_relations','listed_negative_source')}==
                {k:v for k,v in right.items() if k not in ('listed_relations','listed_negative_source')},'noncandidate QA/loss metadata changed')
    prep=dict(n_qa=len(texts),listed_qa=len(bank),k=9,seed=2026,exact_full_replayed_rows=len(bank),exact_slot_bin_match_rows=len(bank),
              frequency_bin_rule=design['frequency_bin_rule'],matching_scope='exact bins; not exact train counts or B1 difficulty',
              train_only=True,same_legal_pool=True,same_non_candidate_metadata=True,fallback_count=0,
              changed_candidate_set_questions=changed,mean_shared_negatives_of_9=sum(d['overlap_k'] for d in diagnostics)/len(bank),
              forced_slot_count=sum(d['forced_slot_count'] for d in diagnostics),total_selected_slots=9*len(bank),
              source_scores_sha256=wiring['source_scores_sha256'],new_training_started=False,performance_results_available=False)
    write_json(run/'bank/preparation_audit.json',prep)
    for path in (run/'bank').iterdir():artifacts[str(path.relative_to(run))]={'sha256':sha256(path),'bytes':path.stat().st_size}
    cmd=make_training_argv(read_json(run/'reference/run_config.json'),run,sc['model_cache'])
    c=dict(schema='fb-frequency-matched-v1',experiment_id=design['experiment_id'],created_at=base.now(),repository_root=str(ROOT),
           design=design,design_sha256=sha256(config),immutable_artifacts=artifacts,training_argv=cmd,model_cache=sc['model_cache'],
           expected_optimizer_steps=32,expected_scheduler_max_steps=120,selection='fixed_step32_before_one_full_test',
           validation_policy='not_run_by_predeclared_budget_plan',borrowed_source_run=str(source),borrowed_inputs_must_be_retained=True)
    write_json(run/'contract.json',c)
    write_json(run/'execution_plan.json',dict(training_argv=cmd,stages=['cli_check','train32','one_full_test','report'],stop_after_report=True))
    write_json(run/'environment_prepare.json',base.environment(run))
    write_json(run/'status.json',dict(stage='prepared',status='ready',at=base.now(),training_complete=False,test_complete=False))
    base.append_event(run,dict(stage='prepare',status='completed',audit=prep))
    return c


def validate_training(run,c):
    validate_mode(run)
    state=read_json(run/'training/trainer_listed/checkpoint-32/trainer_state.json')
    meta=read_json(run/'training/listed/adapter/run_metadata.json')
    require(state['global_step']==meta['trainer_global_step']==c['expected_optimizer_steps']==32
            and state['max_steps']==meta['trainer_max_steps']==c['expected_scheduler_max_steps']==120,'training budget changed')
    write_csv(run/'paper/config_match.csv',parameter_diff(read_json(run/'reference/run_config.json'),read_json(run/'training/run_config.json')))
    return state


def train(run,c):
    base.cli_check(run,c);validate_mode(run)
    if (run/'training/listed/adapter/run_metadata.json').exists():validate_training(run,c);return
    checkpoints=list((run/'training/trainer_listed').glob('checkpoint-*'))
    if checkpoints:
        latest=max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))
        require(read_json(latest/'trainer_state.json')['global_step']<=32,'resume exceeds budget')
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors'):
            require((latest/name).is_file(),'incomplete recovery checkpoint')
    base.execute(run,'train',c['training_argv']);validate_training(run,c)


def evaluate(run,c):
    validate_training(run,c)
    weights={name:sha256(run/'training/listed/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    if (run/'selection.json').exists():require(read_json(run/'selection.json')['adapter_sha256']==weights,'frozen weights changed')
    else:write_json(run/'selection.json',dict(selected_at=base.now(),selection_rule=c['selection'],step=32,adapter_sha256=weights,validation_run=False,search_budget=0,reopen_after_test=False))
    base.execute(run,'eval_test',[str(base.PYTHON),str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_adapter.py'),
                  '--adapter',str(run/'training/listed/adapter'),'--eval_file',str(run/'inputs/test.json'),
                  '--output_dir',str(run/'evaluation/test'),'--model_name','qwen-7b','--model_cache_dir',c['model_cache'],
                  '--dtype','bfloat16','--gen_max_length','32','--progress_every','100','--resume','--label','fb_frequency_matched_seed2026_fixed32_test'])


def comparison(left,right,left_name,right_name):
    result,pairs=paired_result(left,right)
    return dict(n=result['n'],left=left_name,right=right_name,delta_left_minus_right_pp=result['delta_sccl_minus_random9_pp'],net_correct=result['net_correct'],paired_counts=result['paired_counts']),[
        dict(question_id=p['question_id'],**{left_name+'_correct':p['sccl_correct'],right_name+'_correct':p['random9_correct']},outcome=p['outcome_sccl_vs_random9'],question_sha256=p['question_sha256']) for p in pairs]


def report(run,c):
    state=validate_training(run,c);samples=base.eval_samples(load_graph_record(run/'inputs/test.json'));normalize=base.normalization(run/'code_snapshot')
    full,random,freq=[audit_predictions(base.jsonl(run/p),samples,normalize) for p in ('reference/predictions.jsonl','reference/random9_predictions.jsonl','evaluation/test/predictions.jsonl')]
    counts=[sum(r['correct'] for r in idx.values()) for idx in (full,random,freq)]
    require(len(freq)==10000 and counts[:2]==[8990,8648],'reference cohort/metrics changed')
    weights={name:sha256(run/'training/listed/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    require(read_json(run/'selection.json')['adapter_sha256']==read_json(run/'evaluation/test/reuse_identity.json')['adapter']==weights,'test model identity changed')
    summary=read_json(run/'evaluation/test/summary.json')['all'];require(summary['count']==10000 and abs(summary['em']-counts[2]/10000)<1e-12,'summary not reproduced')
    first,pairs=comparison(full,freq,'full','frequency_matched');second,second_pairs=comparison(freq,random,'frequency_matched','random9')
    result=dict(status='completed',at=base.now(),experiment_id=c['experiment_id'],dataset='fb15k237_2',seed=2026,n=10000,
                full_correct=counts[0],random9_correct=counts[1],frequency_matched_correct=counts[2],frequency_matched_em=counts[2]/10000,
                full_minus_frequency_matched=first,frequency_matched_minus_random9=second,preparation_audit=read_json(run/'bank/preparation_audit.json'),
                section_id='experiments',item_id='SC_CTRL_FREQMATCH',claim_links=['C2'],paper_role='claim-carrying',
                scope=c['design']['scope'],limitations=['one seed at previously Full-selected configuration; no equal-HPO or stability claim',
                    'exact log2-frequency bins, not exact counts; score difficulty is not matched',
                    'Full versus frequency-matched random does not isolate structure from B1 feedback',
                    '464 QA repairs and historical test exposure retained'],next_action='write positive/null/negative result and stop; no further training')
    write_json(run/'paper/result.json',result);write_csv(run/'paper/paired_full_vs_frequency_matched.csv',pairs);write_csv(run/'paper/paired_frequency_matched_vs_random.csv',second_pairs)
    prior=read_json(run/'reference/component_result.json')['methods']
    metrics=[dict(model=r['method'],seed=2026,n=r['n'],correct=r['correct'],em_percent=r['em_percent']) for r in prior]
    metrics.insert(1,dict(model=LABEL,seed=2026,n=10000,correct=counts[2],em_percent=counts[2]/100))
    write_csv(run/'paper/metrics.csv',metrics)
    with (run/'reference/component_paired_four_methods.csv').open() as f: old=list(csv.DictReader(f))
    require(len(old)==10000 and len({r['question_id'] for r in old})==10000 and {r['question_id'] for r in old}==set(freq),'prior paired IDs mismatch')
    expected={'full_correct':8990,'random9_correct':8648,'profile_only_correct':8489,'violation_only_correct':8709}
    for key,total in expected.items():require(sum(int(r[key]) for r in old)==total,'prior method metrics changed')
    for row in old:
        qid=row['question_id'];require(int(row['full_correct'])==int(full[qid]['correct']) and int(row['random9_correct'])==int(random[qid]['correct']),'prior Full/Random flags mismatch')
        row['frequency_matched_correct']=int(freq[qid]['correct'])
    write_csv(run/'paper/paired_five_methods.csv',old)
    fields=sorted({k for row in state['log_history'] for k in row});write_csv(run/'paper/training_history.csv',[{k:r.get(k,'') for k in fields} for r in state['log_history']])
    for name in ('diagnostics.csv','frequency_bin_distribution.csv','relation_frequencies.csv','preparation_audit.json'):shutil.copy2(run/'bank'/name,run/'paper'/name)
    end=chr(92)*2
    (run/'paper/table.tex').write_text('\n'.join(['\\begin{tabular}{lrr}','\\toprule','Method & Correct / n & EM (\\%) '+end,'\\midrule']+[f"{r['model']} & {r['correct']}/{r['n']} & {r['em_percent']:.4f} "+end for r in metrics]+['\\bottomrule','\\end{tabular}'])+'\n')
    (run/'paper/RESULTS.md').write_text(f"# Frequency-bin matched random control\n\nFull minus frequency-matched random: {first['delta_left_minus_right_pp']:+.4f} pp. Frequency-matched minus uniform random: {second['delta_left_minus_right_pp']:+.4f} pp.\n\nExact per-question log2-frequency-bin quotas; not exact train counts or B1 difficulty. One seed at previously Full-selected hyperparameters. This comparison tests whether bin-level frequency matching alone reproduces the Full result; it does not isolate a semantic structure mechanism. Positive, null and negative outcomes are retained. No new full validation or search.\n")
    base.inventory(run)
    destination=Path(c['repository_root'])/'11_paper/artifacts/fb_frequency_matched_control_seed2026';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'result.json').exists():require(read_json(destination/'result.json')['experiment_id']==c['experiment_id'],'export belongs to other run')
    for p in (run/'paper').iterdir():
        if p.is_file():shutil.copy2(p,destination/p.name)
    (destination/'.gitattributes').write_text('* -text\n')
    write_json(destination/'evidence_index.json',dict(run_dir=str(run),experiment_id=c['experiment_id'],contract_sha256=sha256(run/'contract.json'),archive_sha256=sha256(run/'artifact_manifest.json'),borrowed_source_run=c['borrowed_source_run'],borrowed_inputs_must_be_retained=True,raw_selected_candidates=str(run/'bank/selected_candidates.csv')))
    write_json(run/'status.json',dict(stage='completed',status='completed',at=base.now(),training_complete=True,test_complete=True,result=result))
    base.append_event(run,dict(stage='report',status='completed',result=result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','train','evaluate','report','run'],nargs='?',default='prepare')
    p.add_argument('--config',type=Path,default=HNG/'configs/fb_frequency_matched.json');p.add_argument('--run-dir',type=Path);a=p.parse_args()
    require(Path(sys.executable).resolve()==base.PYTHON.resolve(),'use project guardenv Python')
    design=read_json(a.config);run=(a.run_dir or Path(design['default_run_dir'])).resolve()
    with base.locked(run):
        try:
            c=prepare(run,a.config) if a.action in ('prepare','run') else base.verify(run)
            print(json.dumps(dict(action=a.action,run_dir=str(run),experiment_id=c['experiment_id'])),flush=True)
            if a.action in ('check','run'):base.cli_check(run,c);validate_mode(run)
            if a.action in ('train','run'):train(run,c)
            if a.action in ('evaluate','run'):evaluate(run,c)
            if a.action in ('report','run'):print(json.dumps(report(run,c),ensure_ascii=False,indent=2),flush=True)
        except BaseException as error:
            write_json(run/'status.json',dict(stage=a.action,status='failed',at=base.now(),error=repr(error)));base.append_event(run,dict(stage=a.action,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
