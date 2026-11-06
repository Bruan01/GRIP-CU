"""Independent standard-library validation of the fixed frequency slices."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,math,hashlib
ART=Path(__file__).resolve().parent
SRC=Path('/home/kieran/grip-cu-runs/fb_frequency_matched_control_20261106_seed2026')
readcsv=lambda p:list(csv.DictReader(p.open()))
rawmap=json.loads((SRC/'inputs/raw/rel2text.json').read_text());counts=Counter()
with (SRC/'inputs/raw/train.txt').open() as f:
    for line in f:
        if line.strip():
            h,r,t=line.split();counts[str(rawmap.get(r,r)).strip()]+=1
freq=readcsv(ART/'train_relation_frequencies.csv')
assert {r['relation']:int(r['train_edge_count']) for r in freq}==dict(counts)
v=sorted(counts.values());assert [v[math.ceil(len(v)/3)-1],v[math.ceil(2*len(v)/3)-1]]==[215,689]
rows=readcsv(ART/'question_slices.csv');original={r['question_id']:r for r in readcsv(SRC/'paper/paired_five_methods.csv')}
assert len(rows)==len(original)==10000 and len({r['question_id'] for r in rows})==10000
methods=['random9','profile_only','violation_only','frequency_matched','full'];groups=defaultdict(list)
for row in rows:
    count=counts[row['relation']];group='low' if count<=215 else 'middle' if count<=689 else 'high'
    assert row['frequency_group']==group and int(row['train_edge_count'])==count
    for method in methods:assert row[method+'_correct']==original[row['question_id']][method+'_correct']
    groups[group].append(row)
metrics=readcsv(ART/'metrics.csv');pairs=readcsv(ART/'paired_transitions.csv')
for metric in metrics:
    items=rows if metric['frequency_group']=='all' else groups[metric['frequency_group']];method=metric['method'];by=defaultdict(list)
    for row in items:by[row['relation']].append(int(row[method+'_correct']))
    correct=sum(sum(flags) for flags in by.values());macro=100*sum(sum(flags)/len(flags) for flags in by.values())/len(by)
    assert int(metric['n'])==len(items) and int(metric['observed_relations'])==len(by) and int(metric['correct'])==correct
    assert abs(float(metric['em_percent'])-100*correct/len(items))<1e-10 and abs(float(metric['macro_relation_em_percent'])-macro)<1e-10
for pair in pairs:
    items=rows if pair['frequency_group']=='all' else groups[pair['frequency_group']];l=pair['left']+'_correct';r=pair['right']+'_correct'
    vals=dict(both_correct=sum(int(x[l])*int(x[r]) for x in items),rescue=sum(int(x[l])*(1-int(x[r])) for x in items),regression=sum((1-int(x[l]))*int(x[r]) for x in items),both_wrong=sum((1-int(x[l]))*(1-int(x[r])) for x in items))
    for key,value in vals.items():assert int(pair[key])==value
    assert int(pair['net_correct'])==vals['rescue']-vals['regression']
manifest=json.loads((ART/'artifact_manifest.json').read_text())
for record in manifest['files']:assert hashlib.sha256((ART/record['path']).read_bytes()).hexdigest()==record['sha256']
plan=json.loads((ART/'plan.json').read_text())
assert hashlib.sha256((ART/'code_snapshot.py').read_bytes()).hexdigest()==plan['code_sha256']
for record in plan['sources'].values():assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
validation=dict(status='pass',raw_train_counts_recomputed=True,cutoffs=[215,689],train_relations=len(counts),test_rows_verified=10000,method_flags_verified=50000,metric_rows_recomputed=len(metrics),transition_rows_recomputed=len(pairs),fixed_output_hashes_verified=len(manifest['files']),all_slices_included=True,gpu_training_runs=0)
(ART/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
