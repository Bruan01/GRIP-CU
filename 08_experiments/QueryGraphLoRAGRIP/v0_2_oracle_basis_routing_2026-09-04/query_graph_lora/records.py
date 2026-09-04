import json
from collections import Counter,defaultdict
REQUIRED={'task_id','text','answer','depth_label','path_relations','split'}
def load_jsonl(path):
 rows=[]
 with path.open(encoding='utf-8') as f:
  for n,line in enumerate(f,1):
   if not line.strip():continue
   row=json.loads(line);missing=REQUIRED-set(row)
   if missing:raise ValueError(f'{path}:{n} missing {sorted(missing)}')
   if int(row['depth_label']) not in range(1,5):raise ValueError('bad depth')
   rows.append(row)
 if not rows:raise ValueError('empty split')
 return rows
def validate_splits(splits):
 ids={k:{r['task_id'] for r in v} for k,v in splits.items()}
 if ids['train']&ids['validation']:raise ValueError('split leakage')
 train_paths={tuple(r['path_relations']) for r in splits['train']}
 for r in splits['train']:r['composition_status']='train'
 for r in splits['validation']:r['composition_status']='seen' if tuple(r['path_relations']) in train_paths else 'novel'
 return {'split_sizes':{k:len(v) for k,v in splits.items()},'overlaps':{},'depth_counts':{k:{str(d):sum(int(r['depth_label'])==d for r in v) for d in range(1,5)} for k,v in splits.items()},'validation_composition_counts':dict(Counter(r['composition_status'] for r in splits['validation']))}
def build_prompt(q):return 'You are answering a graph path query. Follow the ordered relation chain exactly.\nReturn only the final entity identifier and no explanation.\n\nQuestion: '+q+'\nAnswer:'
def records_by_depth(rows):
 out=defaultdict(list)
 for r in rows:out[int(r['depth_label'])].append(r)
 return out
