from dataclasses import dataclass
METHODS=('static_rank4','static_rank8','uniform_basis','random_route','shuffled_oracle_route','oracle_hop_route','oracle_relation_family_route','oracle_relation_path_route')
ORACLE_METHODS=METHODS[5:]
@dataclass(frozen=True)
class RouteMasks: forward:tuple;credit:tuple
def build_route_masks(method,route_ids,groups=4,knockout_group=None):
 if method not in METHODS:raise ValueError(f'unknown method {method}')
 ids=tuple(int(x) for x in route_ids)
 if not ids or any(x<0 or x>=groups for x in ids):raise ValueError('invalid route ids')
 rows=[]
 for route in ids:
  row=tuple(1.0 for _ in range(groups)) if method in {'static_rank4','static_rank8','uniform_basis'} else tuple(1.0 if i==route else 0.0 for i in range(groups))
  if knockout_group is not None:row=tuple(0.0 if i==knockout_group else x for i,x in enumerate(row))
  rows.append(row)
 return RouteMasks(tuple(rows),tuple(rows))
def route_access(method):
 oracle=method in ORACLE_METHODS or method=='shuffled_oracle_route'
 return {'oracle_route_access':oracle,'question_only_router':False,'inference_graph_access':False,'gold_path_used_for_oracle_diagnostic':oracle,'training_gold_route_access':oracle,'validation_gold_route_access':oracle}
