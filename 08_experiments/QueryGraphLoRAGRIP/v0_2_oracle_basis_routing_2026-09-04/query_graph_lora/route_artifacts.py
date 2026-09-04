import hashlib,math,random
from collections import Counter
def stable_bucket(text,groups,seed):return int.from_bytes(hashlib.blake2b(f'{seed}:{text}'.encode(),digest_size=8).digest(),'big')%groups
def features(row,mode):
 rels=[str(x) for x in row['path_relations']]
 if mode=='relation_family':return sorted(set(rels)) or ['EMPTY']
 return [f'p{i}:{x}' for i,x in enumerate(rels)]+[f'b{i}:{rels[i]}>{rels[i+1]}' for i in range(len(rels)-1)] or ['EMPTY']
def vector(items,dim,seed):
 v=[0.0]*dim
 for x in items:
  h=hashlib.blake2b(f'{seed}:{x}'.encode(),digest_size=8).digest();v[int.from_bytes(h,'big')%dim]+=1.0 if h[0]%2==0 else -1.0
 n=math.sqrt(sum(x*x for x in v)) or 1.0;return [x/n for x in v]
def dist(a,b):return sum((x-y)**2 for x,y in zip(a,b))
def fit(rows,mode,groups,dim,seed):
 ordered=sorted(rows,key=lambda r:r['task_id']);vec=[vector(features(r,mode),dim,seed) for r in ordered];centers=[vec[stable_bucket('start',len(vec),seed)]]
 while len(centers)<groups:centers.append(vec[max(range(len(vec)),key=lambda i:(min(dist(vec[i],c) for c in centers),-i))])
 labels=None
 capacities=[len(vec)//groups+(1 if k < len(vec)%groups else 0) for k in range(groups)]
 for _ in range(20):
  preferences=[]
  for i,v in enumerate(vec):
   ranked=sorted(range(groups),key=lambda k:(dist(v,centers[k]),k))
   margin=dist(v,centers[ranked[1]])-dist(v,centers[ranked[0]])
   preferences.append((-margin,i,ranked))
  remaining=capacities[:];new=[None]*len(vec)
  for _,i,ranked in sorted(preferences):
   choice=next(k for k in ranked if remaining[k]>0);new[i]=choice;remaining[choice]-=1
  if new==labels:break
  labels=new
  for k in range(groups):
   members=[v for v,l in zip(vec,labels) if l==k]
   if members:centers[k]=[sum(v[j] for v in members)/len(members) for j in range(dim)]
 return {'mode':mode,'groups':groups,'dim':dim,'seed':seed,'centers':centers,'train_assignments':{r['task_id']:l for r,l in zip(ordered,labels)},'train_cluster_counts':dict(Counter(labels))}
def predict(model,row):
 v=vector(features(row,model['mode']),model['dim'],model['seed']);return min(range(model['groups']),key=lambda k:(dist(v,model['centers'][k]),k))
def build_route_tables(train,validation,groups,seed,dim=64):
 family=fit(train,'relation_family',groups,dim,seed+101);path=fit(train,'relation_path',groups,dim,seed+211);routes={}
 for split,rows in [('train',train),('validation',validation)]:
  table={};path_labels=[]
  for row in rows:
   tid=row['task_id'];table[tid]={'static_rank4':0,'static_rank8':0,'uniform_basis':0,'random_route':stable_bucket(tid,groups,seed+307),'oracle_hop_route':int(row['depth_label'])-1,'oracle_relation_family_route':family['train_assignments'][tid] if split=='train' else predict(family,row),'oracle_relation_path_route':path['train_assignments'][tid] if split=='train' else predict(path,row)};path_labels.append(table[tid]['oracle_relation_path_route'])
  shuffled=list(path_labels);random.Random(seed+401+(split=='validation')).shuffle(shuffled)
  for row,label in zip(rows,shuffled):table[row['task_id']]['shuffled_oracle_route']=label
  routes[split]=table
 audit={'fit_splits':['train'],'application_splits':['train','validation'],'test_opened':False,'groups':groups,'feature_dim':dim,'cluster_counts':{split:{m:dict(Counter(x[m] for x in table.values())) for m in ['random_route','shuffled_oracle_route','oracle_hop_route','oracle_relation_family_route','oracle_relation_path_route']} for split,table in routes.items()}}
 return {'models':{'relation_family':family,'relation_path':path},'routes':routes,'audit':audit}
