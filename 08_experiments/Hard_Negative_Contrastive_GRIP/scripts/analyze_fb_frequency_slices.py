#!/usr/bin/env python3
"""One CPU-only, train-defined frequency slice analysis of frozen predictions."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import shutil,sys

SOURCE=Path('/home/kieran/grip-cu-runs/fb_frequency_matched_control_20261106_seed2026')
FROZEN_HNG=SOURCE/'code_snapshot/08_experiments/Hard_Negative_Contrastive_GRIP'
sys.path[:0]=[str(FROZEN_HNG/'scripts'),str(FROZEN_HNG/'src')]
import run_fb_random9_control as base
from hard_negative_grip.counterfactuals import load_train_relation_profiles
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.task_file import load_graph_record,train_relation_alias_index,match_train_relation
from hard_negative_grip.matched_control import audit_predictions,sha256,read_json,write_json,write_csv

METHODS=['random9','profile_only','violation_only','frequency_matched','full']
EXPECTED=dict(random9=8648,profile_only=8489,violation_only=8709,frequency_matched=8983,full=8990)
LABELS=dict(random9='Uniform Random-9',profile_only='Profile-only-9',violation_only='Violation-only-9',frequency_matched='Full-quota-matched Random-9',full='Full SC-CL-9')
GROUPS=['low','middle','high']
now=lambda:datetime.now(timezone.utc).isoformat()


def train_cutoffs(counts):
    values=sorted(counts)
    if len(values)<3 or any(type(v) is not int or v<1 for v in values):raise ValueError('invalid train counts')
    return values[math.ceil(len(values)/3)-1],values[math.ceil(2*len(values)/3)-1]


def frequency_group(count,cutoffs):
    return 'low' if count<=cutoffs[0] else 'middle' if count<=cutoffs[1] else 'high'


def paired_counts(rows,left='full',right='frequency_matched'):
    c=dict(both_correct=0,rescue=0,regression=0,both_wrong=0)
    for row in rows:
        l=bool(row[left+'_correct']);r=bool(row[right+'_correct'])
        c['both_correct' if l and r else 'rescue' if l else 'regression' if r else 'both_wrong']+=1
    return c


def prepare(run,root):
    run.mkdir(parents=True,exist_ok=True)
    if (run/'plan.json').exists():return read_json(run/'plan.json')
    audit=read_json(root/'11_paper/artifacts/fb_frequency_matched_control_seed2026/completion_audit.json')
    assert audit['status']=='pass' and audit['counts']==EXPECTED
    assert sha256(SOURCE/'contract.json')==audit['contract_sha256']
    profiles=load_train_relation_profiles(SOURCE/'inputs/raw',relation_text_file=SOURCE/'inputs/raw/rel2text.json')
    frequencies={n:p.edge_count for n,p in profiles.items()};cutoffs=train_cutoffs(list(frequencies.values()))
    paths=read_json(SOURCE/'reference/component_result.json')['raw_predictions'].copy();paths['frequency_matched']=str(SOURCE/'evaluation/test/predictions.jsonl')
    sources={'predictions_'+m:Path(p) for m,p in paths.items()}
    sources.update(train=SOURCE/'inputs/raw/train.txt',relation_text=SOURCE/'inputs/raw/rel2text.json',test=SOURCE/'inputs/test.json',source_contract=SOURCE/'contract.json',source_archive=SOURCE/'artifact_manifest.json')
    code=run/'code_snapshot.py';shutil.copy2(__file__,code)
    plan=dict(experiment_id='SC_SLICE_FREQUENCY_FB_SEED2026',created_at=now(),repository_root=str(root),source_run=str(SOURCE),
              research_question='Does near-zero overall Full minus quota-matched EM mask train-frequency-dependent rescue/regression?',
              cutoffs=list(cutoffs),cutoff_rule='train vocabulary relation-count order statistics ceil(R/3),ceil(2R/3); ties never split',
              groups=GROUPS,methods=METHODS,selection='all test questions; no outcome-based cuts or exclusions',
              observed_after_overall_results=True,seed=2026,gpu_training_runs=0,code_sha256=sha256(code),
              sources={name:dict(path=str(p),sha256=sha256(p)) for name,p in sources.items()},
              paper_role='supporting',section_id='experiments',item_id='SC_SLICE_FREQUENCY',claim_links=['C2'],
              next_action='write all slices including negative/null; stop; no new GPU run')
    write_json(run/'plan.json',plan)
    write_csv(run/'train_relation_frequencies.csv',[dict(relation=n,train_edge_count=count,frequency_group=frequency_group(count,cutoffs)) for n,count in sorted(frequencies.items())])
    write_json(run/'status.json',dict(stage='prepared',status='ready',at=now(),test_predictions_inspected_for_cutoffs=False))
    return plan


def analyze(run,plan):
    assert sha256(run/'code_snapshot.py')==plan['code_sha256'],'analysis executable changed'
    assert sha256(Path(__file__))==plan['code_sha256'],'use the frozen code_snapshot.py for an existing plan'
    for record in plan['sources'].values():assert sha256(Path(record['path']))==record['sha256'],'source changed: '+record['path']
    write_json(run/'status.json',dict(stage='analyze',status='running',at=now()))
    samples=base.eval_samples(load_graph_record(Path(plan['sources']['test']['path'])))
    normalizer=base.normalization(SOURCE/'code_snapshot')
    predictions={m:audit_predictions(base.jsonl(Path(plan['sources']['predictions_'+m]['path'])),samples,normalizer) for m in METHODS}
    counts={m:sum(r['correct'] for r in idx.values()) for m,idx in predictions.items()};assert counts==EXPECTED
    profiles=load_train_relation_profiles(SOURCE/'inputs/raw',relation_text_file=SOURCE/'inputs/raw/rel2text.json')
    vocab=load_train_relation_order(SOURCE/'inputs/raw',relation_text_file=SOURCE/'inputs/raw/rel2text.json');aliases=train_relation_alias_index(vocab)
    frequencies={n:p.edge_count for n,p in profiles.items()};assert list(train_cutoffs(list(frequencies.values())))==plan['cutoffs']
    rows=[];by_relation=defaultdict(list);by_group=defaultdict(list)
    for sample in samples:
        qid=sample['question_id'];targets=predictions['full'][qid]['target']
        matched={match_train_relation(t,aliases) for t in targets}
        if None in matched or len(matched)!=1:raise ValueError('unmapped/ambiguous test target; do not silently exclude: '+qid)
        relation=matched.pop();count=frequencies[relation];group=frequency_group(count,plan['cutoffs'])
        row=dict(question_id=qid,relation=relation,train_edge_count=count,frequency_group=group,question_sha256=hashlib.sha256(predictions['full'][qid]['question'].encode()).hexdigest(),**{m+'_correct':int(predictions[m][qid]['correct']) for m in METHODS})
        rows.append(row);by_relation[relation].append(row);by_group[group].append(row)
    assert len(rows)==10000 and sum(map(len,by_group.values()))==10000
    write_csv(run/'question_slices.csv',rows)
    per_relation=[]
    for relation,items in sorted(by_relation.items()):
        per_relation.append(dict(relation=relation,train_edge_count=frequencies[relation],frequency_group=items[0]['frequency_group'],n=len(items),**{m+'_correct':sum(r[m+'_correct'] for r in items) for m in METHODS}))
    write_csv(run/'per_relation.csv',per_relation)
    metrics=[];pairs=[]
    for group in [*GROUPS,'all']:
        items=rows if group=='all' else by_group[group];rels={r['relation'] for r in items}
        for method in METHODS:
            correct=sum(r[method+'_correct'] for r in items)
            macro=sum(sum(r[method+'_correct'] for r in by_relation[n])/len(by_relation[n]) for n in rels)/len(rels) if rels else None
            metrics.append(dict(frequency_group=group,method=method,label=LABELS[method],n=len(items),observed_relations=len(rels),correct=correct,em_percent=100*correct/len(items) if items else None,macro_relation_em_percent=100*macro if macro is not None else None))
        for comparator in METHODS[:-1]:
            c=paired_counts(items,right=comparator);net=c['rescue']-c['regression']
            pairs.append(dict(frequency_group=group,left='full',right=comparator,n=len(items),**c,net_correct=net,delta_pp=100*net/len(items) if items else None))
    for m in METHODS:
        assert sum(r['correct'] for r in metrics if r['method']==m and r['frequency_group']!='all')==EXPECTED[m]
    assert sum(r['net_correct'] for r in pairs if r['right']=='frequency_matched' and r['frequency_group']!='all')==7
    assert sum(r['rescue'] for r in pairs if r['right']=='frequency_matched' and r['frequency_group']!='all')==348
    assert sum(r['regression'] for r in pairs if r['right']=='frequency_matched' and r['frequency_group']!='all')==341
    write_csv(run/'metrics.csv',metrics);write_csv(run/'paired_transitions.csv',pairs)
    end=chr(92)*2
    table=['\\begin{tabular}{lrrrrr}','\\toprule','训练关系频率组 & $n$ & Full EM & 配额匹配 EM & 净变化 & $\\Delta$ (pp) '+end,'\\midrule']
    for group in GROUPS:
        a=next(r for r in metrics if r['method']=='full' and r['frequency_group']==group);b=next(r for r in metrics if r['method']=='frequency_matched' and r['frequency_group']==group);p=next(r for r in pairs if r['right']=='frequency_matched' and r['frequency_group']==group)
        label={'low':'低频','middle':'中频','high':'高频'}[group]
        table.append(f"{label} & {a['n']} & {a['em_percent']:.2f} & {b['em_percent']:.2f} & {p['net_correct']:+d} & {p['delta_pp']:+.2f} "+end)
    table+=['\\bottomrule','\\end{tabular}'];(run/'table.tex').write_text('\n'.join(table)+'\n')
    result=dict(status='completed',completed_at=now(),experiment_id=plan['experiment_id'],cutoffs=plan['cutoffs'],counts=counts,metrics=metrics,paired_transitions=pairs,correctness_flags_recomputed=50000,question_count=10000,observed_relations=len(by_relation),all_slices_reported=True,macro_definition='unweighted mean of per-relation EM over test-observed relations; no missing-relation imputation',limitations=['one seed and one fixed Full-selected configuration','descriptive slices selected after observing the overall comparison; cuts determined by train counts only','frequency measures train graph relation edges, not QA count or entity sparsity','no statistical equivalence, causal mechanism, or cross-seed claim'],next_action=plan['next_action'])
    write_json(run/'result.json',result)
    write_json(run/'artifact_manifest.json',dict(at=now(),files=[dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(run.iterdir()) if p.is_file() and p.name not in ('status.json','artifact_manifest.json','controller.log','run.lock')]))
    destination=Path(plan['repository_root'])/'11_paper/artifacts/fb_frequency_slices_seed2026';destination.mkdir(parents=True,exist_ok=True)
    for p in run.iterdir():
        if p.is_file() and p.name not in ('status.json','controller.log','run.lock'):shutil.copy2(p,destination/p.name)
    (destination/'.gitattributes').write_text('* -text\n')
    write_json(destination/'evidence_index.json',dict(run_dir=str(run),plan_sha256=sha256(run/'plan.json'),manifest_sha256=sha256(run/'artifact_manifest.json'),source_experiment='SC_CTRL_FREQMATCH_FB_SEED2026_FIXED32',gpu_training_runs=0))
    write_json(run/'status.json',dict(stage='completed',status='completed',at=now(),n=10000,gpu_training_runs=0))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'],default='prepare',nargs='?');p.add_argument('--run-dir',type=Path,default=Path('/home/kieran/grip-cu-runs/fb_frequency_slice_20261106_seed2026'));p.add_argument('--repository-root',type=Path,default=Path.cwd());a=p.parse_args()
    with base.locked(a.run_dir):
        plan=prepare(a.run_dir,a.repository_root.resolve())
        if a.action=='run':print(json.dumps(analyze(a.run_dir,plan),ensure_ascii=False,indent=2))
        else:print(json.dumps(dict(status='prepared',cutoffs=plan['cutoffs'],run_dir=str(a.run_dir))))


if __name__=='__main__':main()
