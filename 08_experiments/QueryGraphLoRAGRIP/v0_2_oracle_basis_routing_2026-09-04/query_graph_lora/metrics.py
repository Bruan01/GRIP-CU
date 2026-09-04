import re
from collections import defaultdict
P=re.compile(r'concept_[A-Za-z0-9_:-]+')
def normalize_entity(x):
 x=str(x).strip();m=P.search(x);return m.group(0).rstrip('.,;:!?)]}\"') if m else (x.splitlines()[0].strip("`'\".,;:!?()[]{}") if x else '')
def score_predictions(rows):
 rows=list(rows);byd=defaultdict(list);byc=defaultdict(list);allv=[]
 for r in rows:
  ok=int(normalize_entity(r['prediction_text'])==normalize_entity(r['answer']));allv.append(ok);byd[int(r['depth_label'])].append(ok);byc[r['composition_status']].append(ok)
 acc=lambda x:sum(x)/len(x) if x else None;depth={str(d):acc(byd[d]) for d in range(1,5)};obs=[x for x in depth.values() if x is not None]
 return {'count':len(rows),'correct':sum(allv),'canonical_em':acc(allv),'canonical_em_by_depth':depth,'macro_depth_em':sum(obs)/len(obs),'worst_depth_em':min(obs),'canonical_em_by_composition':{k:acc(v) for k,v in sorted(byc.items())},'composition_counts':{k:len(v) for k,v in sorted(byc.items())}}
