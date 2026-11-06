from pathlib import Path
import csv,json,sys
from datetime import datetime,timezone
run=Path('/home/kieran/grip-cu-runs/fb_frequency_matched_control_20261106_seed2026')
root=Path('/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU');dest=root/'11_paper/artifacts/fb_frequency_matched_control_seed2026'
h=run/'code_snapshot/08_experiments/Hard_Negative_Contrastive_GRIP'
sys.path[:0]=[str(h/'scripts'),str(h/'src')]
import run_fb_random9_control as base
from hard_negative_grip.matched_control import audit_predictions,parameter_diff,sha256,read_json,write_json
from hard_negative_grip.task_file import load_graph_record,load_counterfactual_manifest
from hard_negative_grip.frequency_matched import matched_bank_row,validate_frequency_manifest,frequency_bin
from hard_negative_grip.counterfactuals import load_train_relation_profiles
from hard_negative_grip.official_lists import load_train_relation_order
c=base.verify(run);result=read_json(run/'paper/result.json');state=read_json(run/'status.json')
assert state['status']=='completed' and state['training_complete'] and state['test_complete']
assert sha256(run/'contract.json')=='831d165cdba04af4fa155c71f99bad6e38e607c3a7fcb16c1b2c27f9b301169c'
index=read_json(dest/'evidence_index.json');assert index['archive_sha256']==sha256(run/'artifact_manifest.json')
archive=read_json(run/'artifact_manifest.json')['artifacts']
for record in archive:
 p=run/record['path'];assert p.is_file() and p.stat().st_size==record['bytes'] and sha256(p)==record['sha256'],str(p)
print('Archive PASS',len(archive),flush=True)
actual=read_json(run/'training/run_config.json');reference=read_json(run/'reference/run_config.json')
parameter_diff(reference,actual)
assert actual['listed_negative_source']=='frequency_matched' and actual['negative_sampling_mode']=='random'
assert Path(actual['random_manifest']).resolve()==(run/'bank/frequency_matched.jsonl').resolve()
for step in (8,16,24,32):
 checkpoint=run/f'training/trainer_listed/checkpoint-{step}'
 for name in ('adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'):
  assert (checkpoint/name).is_file(),str(checkpoint/name)
 trainer=read_json(checkpoint/'trainer_state.json');assert trainer['global_step']==step and trainer['max_steps']==120
meta=read_json(run/'training/listed/adapter/run_metadata.json');assert meta['trainer_global_step']==32 and meta['trainer_max_steps']==120
weights={name:sha256(run/'training/listed/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}
assert read_json(run/'selection.json')['adapter_sha256']==read_json(run/'evaluation/test/reuse_identity.json')['adapter']==weights
samples=base.eval_samples(load_graph_record(run/'inputs/test.json'));normalize=base.normalization(run/'code_snapshot')
component=read_json(run/'reference/component_result.json');paths=component['raw_predictions'].copy()
paths['frequency_matched']=str(run/'evaluation/test/predictions.jsonl')
indexes={name:audit_predictions(base.jsonl(Path(p)),samples,normalize) for name,p in paths.items()}
counts={name:sum(r['correct'] for r in rows.values()) for name,rows in indexes.items()}
assert counts==dict(full=8990,random9=8648,profile_only=8489,violation_only=8709,frequency_matched=8983),counts
with (run/'paper/paired_five_methods.csv').open() as f:paired=list(csv.DictReader(f))
assert len(paired)==10000 and len({r['question_id'] for r in paired})==10000
assert {r['question_id'] for r in paired}==set(indexes['full'])
for row in paired:
 for method,idx in indexes.items():assert int(row[method+'_correct'])==int(idx[row['question_id']]['correct'])
for left,right,key in [('full','frequency_matched','full_minus_frequency_matched'),('frequency_matched','random9','frequency_matched_minus_random9')]:
 tally=dict(both_correct=0,rescue=0,regression=0,both_wrong=0)
 for qid in indexes[left]:
  l=indexes[left][qid]['correct'];r=indexes[right][qid]['correct']
  outcome='both_correct' if l and r else 'rescue' if l else 'regression' if r else 'both_wrong';tally[outcome]+=1
 assert tally==result[key]['paired_counts']
 assert tally['rescue']-tally['regression']==result[key]['net_correct']
 assert abs(100*(counts[left]-counts[right])/10000-result[key]['delta_left_minus_right_pp'])<1e-12
summary=read_json(run/'evaluation/test/summary.json')['all'];assert summary['count']==10000 and summary['em']==.8983
print('Raw predictions and 5-method pairs PASS',counts,flush=True)
profiles=load_train_relation_profiles(run/'inputs/raw',relation_text_file=run/'inputs/raw/rel2text.json')
frequencies={n:p.edge_count for n,p in profiles.items()}
vocab=load_train_relation_order(run/'inputs/raw',relation_text_file=run/'inputs/raw/rel2text.json')
parent=load_counterfactual_manifest(run/'reference/counterfactual_bank.jsonl');bank=load_counterfactual_manifest(run/'bank/frequency_matched.jsonl')
pools={r['question_id']:r for r in base.jsonl(run/'reference/pool_audit.jsonl')}
validate_frequency_manifest(bank,k=9);assert set(bank)==set(parent)==set(pools) and len(bank)==3109
forced=0
for qid,row in bank.items():
 pool=pools[qid];gold=row['matched_train_relation'];known=set(pool['excluded_train_relations']);legal=set(pool['legal_relations'])
 assert legal==set(vocab)-known-{gold} and set(row['negative_relations']).issubset(legal)
 assert [frequency_bin(frequencies[n]) for n in row['negative_relations']]==[frequency_bin(frequencies[n]) for n in parent[qid]['negative_relations']]
 group=[dict(qa_id=qid,split='train',candidate_relation=n,matched_train_relation=gold,gold_relation=row['positive_relation'],is_gold=n==gold,is_valid_negative=n in legal) for n in vocab]
 replay,audit=matched_bank_row(group,relation_order=vocab,known_relations=known,frequencies=frequencies,parent_row=parent[qid],seed=2026,k=9)
 assert replay==row,qid
 forced+=audit['forced_slot_count']
assert forced==82
print('3109 deterministic train-frequency samples PASS',flush=True)
with (run/'bank/selected_candidates.csv').open() as f: selected=list(csv.DictReader(f))
assert len(selected)==55962
seen=set()
for row in selected:
 qid=row['question_id'];method=row['method'];slot=int(row['slot']);key=(qid,method,slot);assert key not in seen;seen.add(key)
 expected=parent[qid]['negative_relations'][slot] if method=='full' else bank[qid]['negative_relations'][slot]
 assert row['relation']==expected and int(row['train_edge_count'])==frequencies[expected] and int(row['frequency_bin'])==frequency_bin(frequencies[expected])
assert {r['method'] for r in selected}=={'full','frequency_matched'}
exports=[]
for p in (run/'paper').iterdir():
 if p.is_file():assert sha256(p)==sha256(dest/p.name);exports.append(p.name)
audit=dict(verified_at=datetime.now(timezone.utc).isoformat(),status='pass',contract_sha256=sha256(run/'contract.json'),archive_sha256=sha256(run/'artifact_manifest.json'),immutable_files_verified=len(c['immutable_artifacts']),archive_files_verified=len(archive),unique_correctness_flags_recomputed=50000,paired_five_method_rows_verified=10000,counts=counts,adapter_sha256=weights,optimizer_steps=32,scheduler_max_steps=120,full_checkpoints_retained=[8,16,24,32],frequency_bank_rows_replayed_from_train_counts_without_model_scores=3109,selected_candidate_plot_rows_verified=55962,forced_slots=82,paper_exports_byte_identical=exports,limitations=result['limitations'],next_action='narrow C2; preserve near-null comparator result; stop GPU training')
write_json(dest/'completion_audit.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
