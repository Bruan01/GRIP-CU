import torch
from torch.utils.data import Dataset
from .records import build_prompt
class ExactHopDataset(Dataset):
 def __init__(self,rows,tokenizer,max_length,route_table):self.rows=list(rows);self.t=tokenizer;self.max=int(max_length);self.routes=route_table
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];p=self.t(build_prompt(r['text']),add_special_tokens=True)['input_ids'];a=self.t(str(r['answer'])+(self.t.eos_token or ''),add_special_tokens=False)['input_ids'];p=p[-(self.max-len(a)):]
  return {'input_ids':p+a,'labels':[-100]*len(p)+a,'route_id':self.routes[r['task_id']],'depth':int(r['depth_label']),'task_id':r['task_id'],'answer':r['answer'],'composition_status':r['composition_status']}
class CausalCollator:
 def __init__(self,pad_token_id):self.pad=int(pad_token_id)
 def __call__(self,examples):
  m=max(len(x['input_ids']) for x in examples);ids=[];labels=[];masks=[]
  for x in examples:
   p=m-len(x['input_ids']);ids.append(x['input_ids']+[self.pad]*p);labels.append(x['labels']+[-100]*p);masks.append([1]*len(x['input_ids'])+[0]*p)
  return {'input_ids':torch.tensor(ids),'labels':torch.tensor(labels),'attention_mask':torch.tensor(masks),'route_ids':torch.tensor([x['route_id'] for x in examples]),'depths':torch.tensor([x['depth'] for x in examples]),'task_ids':[x['task_id'] for x in examples],'answers':[x['answer'] for x in examples],'composition_status':[x['composition_status'] for x in examples]}
