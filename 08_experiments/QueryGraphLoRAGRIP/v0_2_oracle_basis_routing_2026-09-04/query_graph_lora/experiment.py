from __future__ import annotations
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from .io_utils import append_jsonl,environment_snapshot,set_seed,sha256_file,write_json
from .metrics import normalize_entity,score_predictions
from .modules import adapter_state_dict,inject_lora
from .records import build_prompt,load_jsonl,records_by_depth,validate_splits
from .route_artifacts import build_route_tables
from .routing import route_access
from .runtime_data import CausalCollator,ExactHopDataset
def dtype_of(x):return {'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[x]
def ctx(device,dtype):return torch.autocast(device_type='cuda',dtype=dtype) if device.type=='cuda' and dtype in {torch.bfloat16,torch.float16} else nullcontext()
def load_splits(repo,c):
 paths={k:repo/c['data'][k] for k in ['train','validation']};splits={k:load_jsonl(p) for k,p in paths.items()};audit=validate_splits(splits)
 for k,n in c['data']['expected_sizes'].items():
  if len(splits[k])!=n:raise ValueError(f'{k} size mismatch')
 audit.update({'paths':{k:str(p.relative_to(repo)) for k,p in paths.items()},'evaluation_splits_opened':['validation'],'test_file_opened':False});return splits,audit
def load_runtime(c,method,override=None):
 mc=c['model'];name=override or mc['name_or_path'];device=torch.device(mc['device'])
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 dtype=dtype_of(mc['dtype']);tok=AutoTokenizer.from_pretrained(name,local_files_only=mc.get('local_files_only',False),trust_remote_code=mc.get('trust_remote_code',False),use_fast=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 tok.padding_side='right';model=AutoModelForCausalLM.from_pretrained(name,local_files_only=mc.get('local_files_only',False),trust_remote_code=mc.get('trust_remote_code',False),torch_dtype=dtype,attn_implementation=mc.get('attn_implementation','eager')).to(device);model.config.use_cache=False;l=c['lora'];controller,report=inject_lora(model,method,l['target_modules'],l['total_rank'],l['groups'],l['group_rank'],l['alpha'],l['dropout']);return model,tok,controller,report,device,dtype,name
def move(batch,device):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
def train(model,tok,controller,rows,routes,c,device,dtype):
 t=c['training'];ds=ExactHopDataset(rows,tok,c['data']['max_length'],routes);g=torch.Generator().manual_seed(t['data_order_seed']);loader=DataLoader(ds,batch_size=t['batch_size'],shuffle=True,generator=g,collate_fn=CausalCollator(tok.pad_token_id));opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=t['learning_rate'],weight_decay=t['weight_decay']);model.train();steps=micro=tokens=0;losses=[];started=time.perf_counter();opt.zero_grad(set_to_none=True)
 while steps<t['max_optimizer_steps']:
  for batch in loader:
   batch=move(batch,device);route=batch.pop('route_ids');[batch.pop(k) for k in ['depths','task_ids','answers','composition_status']];controller.set_batch(route.cpu().tolist());tokens+=int((batch['labels']!=-100).sum())
   with ctx(device,dtype):loss=model(**batch).loss/t['gradient_accumulation_steps']
   loss.backward();micro+=1;losses.append(loss.detach().float().item()*t['gradient_accumulation_steps'])
   if micro%t['gradient_accumulation_steps']==0:
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],t['max_grad_norm']);opt.step();opt.zero_grad(set_to_none=True);steps+=1
    if steps>=t['max_optimizer_steps']:break
 return {'optimizer_steps':steps,'microbatches':micro,'answer_tokens_seen':tokens,'mean_microbatch_loss':sum(losses)/len(losses),'elapsed_seconds':time.perf_counter()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'batch_size':t['batch_size'],'gradient_accumulation_steps':t['gradient_accumulation_steps']}
def generate(model,tok,controller,rows,routes,c,device,dtype):
 model.eval();pred=[];started=time.perf_counter()
 with torch.no_grad():
  for r in rows:
   enc=tok(build_prompt(r['text']),return_tensors='pt',truncation=True,max_length=c['data']['max_length']).to(device);controller.set_batch([routes[r['task_id']]])
   with ctx(device,dtype):out=model.generate(**enc,max_new_tokens=c['evaluation']['max_new_tokens'],do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   text=tok.decode(out[0,enc['input_ids'].shape[1]:],skip_special_tokens=True);pred.append({'task_id':r['task_id'],'answer':r['answer'],'prediction_text':text,'normalized_prediction':normalize_entity(text),'depth_label':r['depth_label'],'path_relations':r['path_relations'],'composition_status':r['composition_status'],'route_id':routes[r['task_id']]})
 metric=score_predictions(pred);metric['elapsed_seconds']=time.perf_counter()-started;return pred,metric
def gradient_probe(model,tok,controller,rows,routes,c,device,dtype):
 grouped=records_by_depth(rows);vectors={};collate=CausalCollator(tok.pad_token_id);model.train()
 for d in range(1,5):
  selected=sorted(grouped[d],key=lambda r:r['task_id'])[:c['analysis']['gradient_probe_batch_size']];ds=ExactHopDataset(selected,tok,c['data']['max_length'],routes);batch=move(collate([ds[i] for i in range(len(ds))]),device);route=batch.pop('route_ids');[batch.pop(k) for k in ['depths','task_ids','answers','composition_status']];controller.set_batch(route.cpu().tolist());model.zero_grad(set_to_none=True)
  with ctx(device,dtype):loss=model(**batch).loss
  loss.backward();vectors[d]=torch.cat([p.grad.detach().float().cpu().reshape(-1) for p in model.parameters() if p.requires_grad and p.grad is not None])
 model.zero_grad(set_to_none=True);return {'batch_size_per_depth':c['analysis']['gradient_probe_batch_size'],'cosine':{f'd{a}_d{b}':float(F.cosine_similarity(vectors[a],vectors[b],dim=0,eps=1e-12)) for a in range(1,5) for b in range(a,5)}}
def run_one(repo_root:Path,experiment_root:Path,config_path:Path,config:dict,method:str,seed:int,output_dir:Path,model_override=None):
 output_dir.mkdir(parents=True,exist_ok=False);set_seed(seed);splits,data_audit=load_splits(repo_root,config);art=build_route_tables(splits['train'],splits['validation'],config['lora']['groups'],seed,config['routing']['feature_dim']);write_json(output_dir/'environment.json',environment_snapshot(repo_root));write_json(output_dir/'data_audit.json',data_audit);write_json(output_dir/'route_artifacts.json',art);started=time.perf_counter();model,tok,controller,report,device,dtype,resolved=load_runtime(config,method,model_override);tr={k:v[method] for k,v in art['routes']['train'].items()};va={k:v[method] for k,v in art['routes']['validation'].items()};training=train(model,tok,controller,splits['train'],tr,config,device,dtype);adapter=output_dir/'adapter_model.pt';torch.save(adapter_state_dict(model),adapter);pred,metric=generate(model,tok,controller,splits['validation'],va,config,device,dtype);append_jsonl(output_dir/'predictions_validation.jsonl',pred);probe=gradient_probe(model,tok,controller,splits['train'],tr,config,device,dtype);summary={'format_version':1,'experiment_id':config['experiment_id'],'method':method,'seed':seed,'model':resolved,'config_path':str(config_path.relative_to(repo_root)),'config_sha256':sha256_file(config_path),'protocol':{**route_access(method),'evaluation_splits_opened':['validation'],'test_file_opened':False,'checkpoint_selection_split':None,'selection_policy':'final_step_only_no_checkpoint_selection'},'data':data_audit,'route_audit':art['audit'],'injection':{'target_modules':list(report.target_modules),'replaced_module_count':len(report.replaced_modules),'trainable_parameters':report.trainable_parameters,'total_parameters':report.total_parameters,'adapter_layout':{'groups':config['lora']['groups'],'group_rank':config['lora']['group_rank'],'total_rank':config['lora']['total_rank'],'alpha':config['lora']['alpha'],'dropout':config['lora']['dropout']}},'training':training,'evaluation':{'split':'validation','max_new_tokens':config['evaluation']['max_new_tokens'],'do_sample':False,'num_beams':1,'max_input_length':config['data']['max_length']},'metrics':{'validation':metric},'gradient_probe':probe,'adapter_artifact':{'path':adapter.name,'size_bytes':adapter.stat().st_size,'sha256':sha256_file(adapter)},'total_elapsed_seconds':time.perf_counter()-started};write_json(output_dir/'run_summary.json',summary);return summary
