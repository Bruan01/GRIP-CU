from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .routing import build_route_masks
@dataclass
class InjectionReport:method:str;target_modules:tuple;replaced_modules:tuple;trainable_parameters:int;total_parameters:int
class RoutingController:
 def __init__(self,method,groups):self.method=method;self.groups=groups;self.route_ids=None;self.knockout_group=None;self.cache={}
 def set_batch(self,x):self.route_ids=tuple(int(v) for v in x);self.cache={}
 def set_knockout(self,x):self.knockout_group=x;self.cache={}
 def gates(self,device,dtype):
  if self.route_ids is None:raise RuntimeError('route ids not set')
  key=(str(device),str(dtype))
  if key not in self.cache:
   m=build_route_masks(self.method,self.route_ids,self.groups,self.knockout_group);self.cache[key]=(torch.tensor(m.forward,device=device,dtype=dtype),torch.tensor(m.credit,device=device,dtype=dtype))
  return self.cache[key]
def broadcast(g,u):
 if g.shape[0]!=u.shape[0]:
  if g.shape[0]==1:g=g.expand(u.shape[0])
  else:raise RuntimeError('route batch mismatch')
 return g.reshape(g.shape[0],*([1]*(u.ndim-1)))
class StaticLoraLinear(nn.Module):
 def __init__(self,base,rank,alpha,dropout):
  super().__init__();self.base=base;self.scale=float(alpha)/rank;self.drop=nn.Dropout(dropout) if dropout else nn.Identity();self.A=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32,device=base.weight.device));self.B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32,device=base.weight.device));nn.init.kaiming_uniform_(self.A,a=5**.5)
  for p in base.parameters():p.requires_grad=False
 def forward(self,x):
  out=self.base(x);u=F.linear(F.linear(self.drop(x).to(self.A.dtype),self.A),self.B);return out+u.to(out.dtype)*self.scale
class BasisLoraLinear(nn.Module):
 def __init__(self,base,groups,group_rank,total_rank,alpha,dropout,controller):
  super().__init__();self.base=base;self.groups=groups;self.scale=float(alpha)/total_rank;self.drop=nn.Dropout(dropout) if dropout else nn.Identity();self.controller=controller;self.A=nn.Parameter(torch.empty(groups,group_rank,base.in_features,dtype=torch.float32,device=base.weight.device));self.B=nn.Parameter(torch.zeros(groups,base.out_features,group_rank,dtype=torch.float32,device=base.weight.device))
  for g in range(groups):nn.init.kaiming_uniform_(self.A[g],a=5**.5)
  for p in base.parameters():p.requires_grad=False
 def forward(self,x):
  out=self.base(x);z=self.drop(x).to(self.A.dtype);fg,cg=self.controller.gates(z.device,z.dtype);total=torch.zeros_like(out,dtype=self.A.dtype)
  for g in range(self.groups):
   u=F.linear(F.linear(z,self.A[g]),self.B[g]);f=broadcast(fg[:,g],u);c=broadcast(cg[:,g],u);total+=u*c+u.detach()*(f-c)
  return out+total.to(out.dtype)*self.scale
def parent(model,name):
 parts=name.split('.');p=model
 for x in parts[:-1]:p=getattr(p,x)
 return p,parts[-1]
def inject_lora(model,method,target_modules,total_rank,groups,group_rank,alpha,dropout):
 for p in model.parameters():p.requires_grad=False
 controller=RoutingController(method,groups);names=[]
 for name,module in list(model.named_modules()):
  if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in target_modules:
   p,c=parent(model,name);new=StaticLoraLinear(module,4 if method=='static_rank4' else total_rank,alpha,dropout) if method in {'static_rank4','static_rank8'} else BasisLoraLinear(module,groups,group_rank,total_rank,alpha,dropout,controller);setattr(p,c,new);names.append(name)
 if not names:raise ValueError('no target module')
 return controller,InjectionReport(method,tuple(target_modules),tuple(names),sum(p.numel() for p in model.parameters() if p.requires_grad),sum(p.numel() for p in model.parameters()))
def adapter_state_dict(model):return {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
