#!/usr/bin/env python3
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];sys.path.insert(0,str(ROOT))
from query_graph_lora.config import load_config
from query_graph_lora.io_utils import write_json
from query_graph_lora.records import load_jsonl,validate_splits
from query_graph_lora.route_artifacts import build_route_tables
c=load_config(ROOT/'configs/phase_a_oracle.json')
s={k:load_jsonl(REPO/c['data'][k]) for k in ['train','validation']}
a=validate_splits(s)
r=build_route_tables(s['train'],s['validation'],c['lora']['groups'],43,c['routing']['feature_dim'])
path_counts=r['models']['relation_path']['train_cluster_counts']
family_counts=r['models']['relation_family']['train_cluster_counts']
validation_routes=r['routes']['validation']
required={'static_rank8','uniform_basis','shuffled_oracle_route','oracle_hop_route','oracle_relation_path_route'}
payload={
 'status':'LOCAL_IMPLEMENTATION_READY_FOR_WSL_VALIDATION',
 'config':str((ROOT/'configs/phase_a_oracle.json').relative_to(REPO)),
 'data':a|{'paths':{k:c['data'][k] for k in ['train','validation']}},
 'route_audit':r['audit'],
 'invariants':{
  'validation_only':c['evaluation_splits']==['validation'],
  'test_path_absent':'test' not in c['data'],
  'test_not_opened':r['audit']['test_opened'] is False,
  'expected_train_size':len(s['train'])==c['data']['expected_sizes']['train'],
  'expected_validation_size':len(s['validation'])==c['data']['expected_sizes']['validation'],
  'train_validation_disjoint':not ({x['task_id'] for x in s['train']} & {x['task_id'] for x in s['validation']}),
  'train_depth_balanced':set(a['depth_counts']['train'].values())=={179},
  'validation_depth_balanced':set(a['depth_counts']['validation'].values())=={38},
  'seen_and_novel_composition':set(a['validation_composition_counts'])=={'seen','novel'},
  'route_fit_train_only':r['audit']['fit_splits']==['train'],
  'path_clusters_balanced':sorted(path_counts.values())==[179]*4,
  'family_clusters_balanced':sorted(family_counts.values())==[179]*4,
  'shuffle_preserves_path_distribution':Counter(x['shuffled_oracle_route'] for x in validation_routes.values())==Counter(x['oracle_relation_path_route'] for x in validation_routes.values()),
  'minimal_matrix_exact':set(c['minimal_methods'])==required,
  'rank_budget_exact':c['lora']['groups']*c['lora']['group_rank']==c['lora']['total_rank']==8,
  'registered_seeds_exact':c['training']['seeds']==[43,44],
 }
}
if not all(payload['invariants'].values()):raise AssertionError(json.dumps(payload['invariants'],ensure_ascii=False))
write_json(ROOT/'artifacts/setup_audit.json',payload)
print(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True))
