#!/usr/bin/env python3
"""Frozen, serial Llama main-table campaign. Every stage fails closed."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
HNG=Path(__file__).resolve().parents[1];ROOT=HNG.parents[1]
sys.path[:0]=[str(HNG/'src'),str(HNG/'scripts')]
from hard_negative_grip.matched_control import read_json,write_json,write_csv,sha256,audit_predictions,paired_result
import run_fb_random9_control as base
from hard_negative_grip.task_file import load_graph_record,load_counterfactual_manifest,build_qa_assets_from_task_texts,original_question_ids_from_payload
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.rerank_audit import validate_choices,SCORE_RULE,read_rows,summary
from run_fb_rerank_campaign import cache_identity
PYTHON=Path('/home/kieran/miniconda3/envs/guardenv/bin/python')
RUN=Path('/home/kieran/grip-cu-runs/llama_four_20261111')
STORAGE=Path('/mnt/e/GRIP-CU/artifacts/llama_four_20261111')
CACHE=Path('/mnt/e/GRIP-CU/model_cache')
EXPORT=ROOT/'11_paper/artifacts/llama_four_seed2026'
DOC=ROOT/'08_experiments/Hard_Negative_Contrastive_GRIP/docs/llama_four_dataset_20261111'

def state(**v):write_json(RUN/'status.json',dict(at=base.now(),**v))
def env():
 e=os.environ.copy();snap=RUN/'code_snapshot'
 e.update(PYTHONPATH=os.pathsep.join([str(snap),str(snap/base.REL_BASE),str(snap/base.REL_HNG/'src'),str(snap/base.REL_HNG/'scripts')]),GRIP_BASELINE_DIR=str(snap/base.REL_BASE),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',GRIP_REUSE_CACHE=str(RUN/'reuse_cache'))
 return e

def execute(dataset,stage,cmd):
 done=RUN/'stages'/f'{dataset}_{stage}.json'
 if done.exists():
  assert read_json(done)['argv']==cmd,'completed stage command changed';return
 assert shutil.disk_usage('/mnt/e').free>4*1024**3,'E storage reserve insufficient'
 assert shutil.disk_usage('/mnt/c').free>1024**3,'C storage reserve insufficient'
 gpu=base.capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
 assert gpu.get('returncode')==0 and not gpu['stdout'].strip(),'another GPU compute process is active'
 log=RUN/'logs'/f'{dataset}_{stage}.log';log.parent.mkdir(exist_ok=True)
 assert not log.exists(),'partial/failed stage retained; audit before resuming'
 with log.open('w') as f:
  f.write(json.dumps(dict(at=base.now(),argv=cmd))+'\n');f.flush()
  p=subprocess.Popen(cmd,cwd=RUN/'code_snapshot',env=env(),stdout=f,stderr=subprocess.STDOUT)
  start=time.monotonic()
  try:
   while True:
    code=p.poll();free=shutil.disk_usage('/mnt/e').free;c_free=shutil.disk_usage('/mnt/c').free
    heartbeat=dict(dataset=dataset,stage=stage,status='running' if code is None else 'finished',pid=p.pid,elapsed_seconds=time.monotonic()-start,log=str(log),log_age_seconds=time.time()-log.stat().st_mtime,free_e_bytes=free,free_c_bytes=c_free,gpu=base.capture(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']))
    state(**heartbeat)
    with (RUN/'resources.jsonl').open('a') as h:h.write(json.dumps(dict(at=base.now(),**heartbeat))+'\n')
    if code is not None:break
    if free<1024**3 or c_free<512*1024**2:
     p.terminate();p.wait(timeout=60);raise RuntimeError('own worker stopped at physical disk safety reserve')
    if heartbeat['log_age_seconds']>1800:write_json(RUN/'attention.json',dict(at=base.now(),reason='no new log for 30 minutes; worker still alive, inspect CPU/loading/checkpoint activity',**heartbeat))
    time.sleep(30)
  except BaseException:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=45)
    except subprocess.TimeoutExpired:p.kill();p.wait()
   raise
 assert code==0,f'{dataset}/{stage} exited {code}; see {log}'
 write_json(done,dict(at=base.now(),argv=cmd,log=str(log),returncode=code,elapsed_seconds=time.monotonic()-start))

def config():
 if (RUN/'contract.json').exists():return read_json(RUN/'contract.json')
 specs=[]
 for d in ['wn18rr','fb15k237_2','codexm','nell23k']:
  src=(HNG/'results/runs/wn18rr_sccl_hpo_s2026_lc0p25/run_config.json' if d=='wn18rr' else HNG/f'results/hpo/stage2_three_dataset_20261030/{d}/search/trials/trial_0001/train_run/run_config.json')
  ref=read_json(src);raw=ROOT/f'04_benchmark/data/{d}/raw'
  task=base.resolve_source(ref['input_file']);valid=base.resolve_source(ref['eval_file']);test=ROOT/f'04_benchmark/data/{d}/processed/processed_test.json'
  mapping=raw/'rel2text.json';entity=ref.get('paths',{}).get('entity_text_file')
  if d=='wn18rr':entity=str(raw/'entity2text.txt')
  spec=dict(dataset=d,reference_config=str(src),reference_config_sha256=sha256(src),reference=ref,source_task=str(task),task_sha256=sha256(task),raw=str(raw),valid=str(valid),test=str(test),valid_sha256=sha256(valid),test_sha256=sha256(test),candidate_count=11 if d=='wn18rr' else 10,gen_max_length=1000 if d=='wn18rr' else 32,expected_test=dict(wn18rr=3134,fb15k237_2=10000,codexm=10311,nell23k=4943)[d],relation_text_file=str(mapping) if mapping.exists() else None,entity_text_file=str(base.resolve_source(entity)) if entity else None)
  spec['raw_train_sha256']=sha256(raw/'train.txt')
  specs.append(spec)
 snap=RUN/'code_snapshot';snap.mkdir()
 sources=[]
 for folder in [HNG/'src',HNG/'scripts',ROOT/base.REL_BASE]:
  for p in folder.rglob('*.py'):
   if any(x in p.parts for x in ['__pycache__','.git','model_cache','outputs','artifacts','results','tests']):continue
   rel=p.relative_to(ROOT);dst=snap/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);sources.append(dict(path=str(rel),sha256=sha256(dst)))
 c=dict(model_name='llama3-8b',model_id='meta-llama/Llama-3.1-8B-Instruct',model_cache=str(CACHE),seed=2026,datasets=specs,source_files=sources,source_digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),stage1_epochs=1.,policy='Qwen-selected fixed Stage2 configurations transferred without new HPO; matched B1 uses same Stage1/steps/optimizer, lambda=0; no test-based selection',retention='one active checkpoint; after final adapter and history identity captured, discard only new campaign checkpoint tensors; retain all final Stage1/B1/SC-CL adapters',wn_scope='historical training pool with potentially unverified entity pairs; count and disclose; no repaired-B1 or clean-pair claim')
 write_json(RUN/'contract.json',c);write_json(DOC/'campaign_contract.json',c);return c

def prepare_data(s):
 work=STORAGE/s['dataset'];work.mkdir(parents=True,exist_ok=True)
 if (work/'data_audit.json').exists():
  a=read_json(work/'data_audit.json');assert sha256(work/'task.json')==a['task_sha256'];return work
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(CACHE/'meta-llama--Llama-3.1-8B-Instruct',local_files_only=True)
 original=read_json(Path(s['source_task']));assert sha256(Path(s['source_task']))==s['task_sha256']
 pattern=re.compile(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',re.S)
 converted=[]
 for text in original['qa_samples']:
  matches=list(pattern.finditer(text));assert matches,'unsupported task chat format'
  assert not pattern.sub('',text).strip(),'unparsed chat content'
  messages=[dict(role=m.group(1),content=m.group(2)) for m in matches]
  out=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
  assert all(m['content'] in out for m in messages),'message content changed'
  converted.append(out)
 original['qa_samples']=converted;original['llama_conversion']=dict(source_task_sha256=s['task_sha256'],model='meta-llama/Llama-3.1-8B-Instruct',message_contents_verified=True)
 write_json(work/'task.json',original)
 eligible=s['reference'].get('relation_eligibility')
 if eligible:
  e=read_json(base.resolve_source(eligible));e['task_sha256']=sha256(work/'task.json');write_json(work/'eligibility.json',e)
 audit=dict(task_sha256=sha256(work/'task.json'),source_task_sha256=s['task_sha256'],qa_count=len(converted),context_count=len(original['context_samples']),all_message_contents_verified=True,model='llama3-8b')
 write_json(work/'data_audit.json',audit);return work

def args_for(s,work,out,stage):
 ref=s['reference'];cmd=[str(PYTHON),str(RUN/'code_snapshot'/base.REL_HNG/'scripts/train_listed_contrastive.py'),'--input_file',str(work/'task.json'),'--eval_file',s['valid'],'--output_dir',str(out),'--stage',stage,'--model_name','llama3-8b','--model_cache_dir',str(CACHE),'--dtype','bfloat16','--raw_dir',s['raw'],'--dataset',s['dataset'],'--seed','2026','--save_steps','16','--save_total_limit','1','--s1_gradient_checkpointing','--swanlab_mode','offline','--defer_eval']
 fields=['lora_r','lora_alpha','learning_rate','weight_decay','adam_epsilon','max_grad_norm','temperature','involve_qa_epochs']
 for k in fields:cmd.extend(['--'+k,str(ref[k])])
 cmd+=['--gradient_accumulation_steps',str(ref['gradient_accumulation_steps']*ref['per_device_train_batch_size']),'--num_train_epochs','1','--per_device_train_batch_size','1','--target_modules',*ref['target_modules'],'--gen_max_length',str(s['gen_max_length'])]
 if stage!='s1':cmd+=['--s1_adapter',str(work/'s1/s1_adapter'),'--stop_after_step',str(ref['stop_after_step'])]
 if s['relation_text_file']:cmd+=['--relation_text_file',s['relation_text_file']]
 if (work/'eligibility.json').exists():cmd+=['--relation_eligibility',str(work/'eligibility.json')]
 if stage=='listed':cmd+=['--listed_negative_source','structure_counterfactual','--negative_sampling_mode','structure_counterfactual','--counterfactual_manifest',str(work/'bank/counterfactual_bank.jsonl'),'--lambda_candidate',str(ref['lambda_candidate'])]
 if stage=='b1':cmd+=['--lambda_candidate','0']
 return cmd

def cli_preflight(s,work,out,stage):
 cmd=args_for(s,work,out,stage)
 program="import sys,json; import train_listed_contrastive as t; sys.argv=json.loads(sys.argv[1])[1:]; a=t.parse_args(); assets=t.resolve_training_assets(a); t.write_run_config(a.output_dir,a); print(json.dumps({'qa':len(assets[2]),'listed':sum(bool(m['listed_relations']) for m in assets[3]),'model_loaded':False}))"
 execute(s['dataset'],'preflight_'+stage,[str(PYTHON),'-c',program,json.dumps(cmd)])

def retain_and_trim(out,variant,expected_steps=None):
 adapter=out/('s1_adapter' if variant=='s1' else f'{variant}/adapter')
 assert (adapter/'adapter_model.safetensors').is_file() and (adapter/'run_metadata.json').is_file()
 if expected_steps is not None:assert read_json(adapter/'run_metadata.json')['trainer_global_step']==expected_steps,'actual optimizer steps differ from frozen budget'
 write_json(out/'retained_adapter_identity.json',{n:sha256(adapter/n) for n in ['adapter_config.json','adapter_model.safetensors']})
 states=[];deleted=[]
 for ck in (out/f'trainer_{variant}').glob('checkpoint-*'):
  if (ck/'trainer_state.json').exists():states.append(read_json(ck/'trainer_state.json'))
  for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:
   p=ck/name
   if p.is_file():deleted.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
 write_json(out/'training_history_and_retention.json',dict(states=states,discarded_new_transient_files=deleted,final_adapter_retained=True))

def ranking(s,work,arm,adapter,predictions,c):
 r=work/'ranking'/arm;r.mkdir(parents=True,exist_ok=True)
 if not (r/'contract.json').exists():
  (r/'inputs').mkdir();(r/'inputs/test.json').symlink_to(s['test']);(r/'code_snapshot').symlink_to(RUN/'code_snapshot',target_is_directory=True)
  normalize=base.normalization(RUN/'code_snapshot');pred=base.jsonl(predictions);samples=base.eval_samples(load_graph_record(Path(s['test'])))
  direct=audit_predictions(pred,samples,normalize);assert len(direct)==s['expected_test']
  sys.path.insert(0,str(RUN/'code_snapshot'/base.REL_BASE));from grip.relation_cecr import parse_relation_candidates
  cohort=[]
  for i,item in enumerate(samples):
   row=direct[item['question_id']];choices=parse_relation_candidates(row['question']);gold,collisions=validate_choices(choices,row['target'],normalize,expected_candidates=s['candidate_count'])
   cohort.append(dict(dataset_index=i,question_id=row['question_id'],question=row['question'],target=row['target'],candidates=choices,gold_index=gold,frequency_group='unstratified',train_edge_count=0,direct_correct={}))
  (r/'inputs/cohort.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cohort))
  ident={n:sha256(adapter/n) for n in ['adapter_config.json','adapter_model.safetensors']}
  rc=dict(model_name='llama3-8b',model_cache=str(CACHE),models={arm:dict(adapter_dir=str(adapter),adapter_sha256=ident)},candidate_count=s['candidate_count'],cohort_sha256=sha256(r/'inputs/cohort.jsonl'),code_digest=c['source_digest'],score_batch_size=2,max_length=4096,score_rule=SCORE_RULE,base_model_identity=cache_identity_llama(),immutable_artifacts={})
  write_json(r/'contract.json',rc)
 execute(s['dataset'],'rank_'+arm,[str(PYTHON),str(RUN/'code_snapshot'/base.REL_HNG/'scripts/evaluate_matched_candidate_rerank.py'),'--run-dir',str(r),'--model',arm])

def cache_identity_llama():
 p=CACHE/'meta-llama--Llama-3.1-8B-Instruct';return dict(path=str(p),config_sha256=sha256(p/'config.json'),weights={x.name:dict(bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns) for x in p.glob('*.safetensors')})

def summarize(s,work):
 samples=base.eval_samples(load_graph_record(Path(s['test'])));normal=base.normalization(RUN/'code_snapshot');metrics=[];direct={}
 for arm in ['b1','sccl']:
  direct[arm]=audit_predictions(base.jsonl(work/f'evaluation/{arm}/predictions.jsonl'),samples,normal)
  r=work/'ranking'/arm;cohort=base.jsonl(r/'inputs/cohort.jsonl');rows=read_rows(r/f'models/{arm}/predictions.jsonl',cohort,expected_candidates=s['candidate_count']);m=summary(rows,len(samples));assert m['status']=='completed'
  metrics.append(dict(dataset=s['dataset'],method=arm,n=len(samples),direct_em_percent=100*sum(x['correct'] for x in direct[arm].values())/len(samples),h1_percent=100*m['h1'],mrr_percent=100*m['mrr'],h3_percent=100*m['h3']))
 paired,pairs=paired_result(direct['sccl'],direct['b1']);dest=EXPORT/s['dataset'];dest.mkdir(parents=True,exist_ok=True)
 write_csv(dest/'metrics.csv',metrics);write_csv(dest/'paired_direct.csv',pairs)
 write_json(dest/'result.json',dict(status='completed',metrics=metrics,sccl_minus_b1=paired,model='meta-llama/Llama-3.1-8B-Instruct',seed=2026,run_dir=str(work),limitations=['one seed; Qwen-selected frozen hyperparameters transferred without independent Llama tuning','WN historical unverified-pair limitations retained where applicable','local task protocol, not strict reproduction of published GRIP numbers']))
 all_metrics=[]
 for p in sorted(EXPORT.glob('*/result.json')):all_metrics.extend(read_json(p)['metrics'])
 write_csv(EXPORT/'main_table.csv',all_metrics)
 end=chr(92)*2
 lines=['\\begin{tabular}{llrrrr}','\\hline','Dataset & Method & EM & H@1 & MRR & H@3 '+end,'\\hline']
 for m in all_metrics:lines.append(f"{m['dataset'].replace('_',chr(92)+'_')} & {m['method']} & {m['direct_em_percent']:.2f} & {m['h1_percent']:.2f} & {m['mrr_percent']:.2f} & {m['h3_percent']:.2f} "+end)
 (EXPORT/'main_table.tex').write_text('\n'.join(lines+['\\hline','\\end{tabular}'])+'\n')


def main():
 assert Path(sys.executable).resolve()==PYTHON.resolve();RUN.mkdir(exist_ok=True);STORAGE.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True);(RUN/'stages').mkdir(exist_ok=True)
 with base.locked(RUN):
  try:
   while True:
    d=read_json(RUN/'download_status.json');state(stage='model_download',status=d['status'],download=d)
    if d['status']=='completed':break
    assert d['status']!='failed',d
    time.sleep(30)
   c=config()
   for f in c['source_files']:assert sha256(RUN/'code_snapshot'/f['path'])==f['sha256'],'frozen source changed'
   for s in c['datasets']:
    assert sha256(Path(s['raw'])/'train.txt')==s['raw_train_sha256']
    assert sha256(Path(s['valid']))==s['valid_sha256'] and sha256(Path(s['test']))==s['test_sha256']
    d=s['dataset'];work=prepare_data(s);state(dataset=d,stage='prepared',status='running')
    for stage,out,variant in [('s1',work/'s1','s1'),('b1',work/'b1','b1')]:
     cli_preflight(s,work,out,stage)
     execute(d,stage,args_for(s,work,out,stage));retain_and_trim(out,variant,None if stage=='s1' else s['reference']['stop_after_step'])
    score=[str(PYTHON),str(RUN/'code_snapshot'/base.REL_HNG/'scripts/score_relation_candidates_offline.py'),'--task_file',str(work/'task.json'),'--raw_dir',s['raw'],'--b1_adapter',str(work/'b1/b1/adapter'),'--output_dir',str(work/'scores'),'--model_name','llama3-8b','--model_cache_dir',str(CACHE),'--candidate_batch_size','2','--limit','0','--filter_splits','train','--temperature','1','--seed','2026','--resume']
    if s['relation_text_file']:score+=['--relation_text_file',s['relation_text_file']]
    if s['entity_text_file']:score+=['--entity_text_file',s['entity_text_file']]
    if (work/'eligibility.json').exists():score+=['--relation_eligibility',str(work/'eligibility.json')]
    execute(d,'score',score)
    bank=[str(PYTHON),str(RUN/'code_snapshot'/base.REL_HNG/'scripts/build_structure_counterfactual_bank.py'),'--scores',str(work/'scores/candidate_scores.jsonl'),'--raw_dir',s['raw'],'--output_dir',str(work/'bank'),'--total_k','9','--seed','2026']
    if s['relation_text_file']:bank+=['--relation_text_file',s['relation_text_file']]
    if d=='wn18rr':bank+=['--strip_first_underscore']
    execute(d,'bank',bank)
    payload=read_json(work/'task.json');manifest=load_counterfactual_manifest(work/'bank/counterfactual_bank.jsonl')
    eligible=set(read_json(work/'eligibility.json')['eligible_question_ids']) if (work/'eligibility.json').exists() else None
    build_qa_assets_from_task_texts(payload['qa_samples'],seed=2026,relation_order=load_train_relation_order(Path(s['raw']),relation_text_file=Path(s['relation_text_file']) if s['relation_text_file'] else None),listed_negative_source='structure_counterfactual',counterfactual_manifest=manifest,eligible_question_ids=eligible)
    cli_preflight(s,work,work/'sccl','listed')
    execute(d,'sccl',args_for(s,work,work/'sccl','listed'));retain_and_trim(work/'sccl','listed',s['reference']['stop_after_step'])
    for arm,adapter in [('b1',work/'b1/b1/adapter'),('sccl',work/'sccl/listed/adapter')]:
     ev=[str(PYTHON),str(RUN/'code_snapshot'/base.REL_HNG/'scripts/evaluate_adapter.py'),'--adapter',str(adapter),'--eval_file',s['test'],'--output_dir',str(work/'evaluation'/arm),'--model_name','llama3-8b','--model_cache_dir',str(CACHE),'--gen_max_length',str(s['gen_max_length']),'--progress_every','100','--resume']
     execute(d,'test_'+arm,ev);ranking(s,work,arm,adapter,work/f'evaluation/{arm}/predictions.jsonl',c)
    summarize(s,work)
   state(stage='completed',status='completed');write_json(EXPORT/'completion.json',read_json(RUN/'status.json'))
  except BaseException as e:state(stage='campaign',status='failed',error=repr(e));raise

if __name__=='__main__':main()
