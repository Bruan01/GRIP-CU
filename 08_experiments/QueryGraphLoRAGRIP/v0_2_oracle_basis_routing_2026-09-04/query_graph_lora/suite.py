import json
from pathlib import Path
from .io_utils import write_csv,write_json
def discover_runs(root):return [json.loads(p.read_text()) for p in sorted(Path(root).glob('*/seed_*/run_summary.json'))]
def mean(xs):xs=list(xs);return sum(xs)/len(xs)
def aggregate_runs(runs):
 grouped={}
 for r in runs:grouped.setdefault(r['method'],[]).append(r)
 out={}
 for m,items in grouped.items():
  out[m]={'seed_count':len(items),'seeds':sorted(x['seed'] for x in items),'canonical_em':mean(x['metrics']['validation']['canonical_em'] for x in items),'novel_em':mean((x['metrics']['validation']['canonical_em_by_composition'].get('novel') or 0) for x in items),'seen_em':mean((x['metrics']['validation']['canonical_em_by_composition'].get('seen') or 0) for x in items),'trainable_parameters':sorted({x['injection']['trainable_parameters'] for x in items}),'optimizer_steps':sorted({x['training']['optimizer_steps'] for x in items}),'answer_tokens_seen':sorted(x['training']['answer_tokens_seen'] for x in items)}
 return out
def apply_gate(a,c):
 required=['static_rank8','uniform_basis','shuffled_oracle_route','oracle_hop_route','oracle_relation_path_route'];missing=[m for m in required if m not in a]
 if missing:return {'decision':'INCOMPLETE_PHASE_A','all_checks_pass':False,'missing_methods':missing}
 base=a['static_rank8'];name=max(['oracle_hop_route','oracle_relation_path_route'],key=lambda m:a[m]['canonical_em']);best=a[name];control=max(a['uniform_basis']['canonical_em'],a['shuffled_oracle_route']['canonical_em']);g=c['gate'];param=[a[m]['trainable_parameters'] for m in required]
 checks={'minimum_seeds':{'value':min(a[m]['seed_count'] for m in required),'threshold':g['minimum_seeds']},'oracle_canonical_gain_pp':{'value':100*(best['canonical_em']-base['canonical_em']),'threshold':g['canonical_gain_pp']},'oracle_novel_gain_pp':{'value':100*(best['novel_em']-base['novel_em']),'threshold':g['novel_gain_pp']},'seen_drop_within_pp':{'value':100*(best['seen_em']-base['seen_em']),'threshold':-g['maximum_seen_drop_pp']},'oracle_over_control_pp':{'value':100*(best['canonical_em']-control),'threshold':g['oracle_over_control_pp']},'equal_parameter_budget':{'value':int(len({tuple(x) for x in param})==1),'threshold':1}}
 for x in checks.values():x['pass']=x['value']>=x['threshold']
 passed=all(x['pass'] for x in checks.values());return {'decision':'GO_LEARNED_QUERY_ROUTER' if passed else 'STOP_GRAPH_CONDITIONAL_LORA','all_checks_pass':passed,'best_oracle':name,'checks':checks}
def render(a,g):
 lines=['# QueryGraph-LoRA Phase-A Validation Report','',f"Decision: **{g['decision']}**",'','| Method | Seeds | Canonical EM | Novel EM | Seen EM | Parameters |','|---|---:|---:|---:|---:|---:|']
 for m,v in sorted(a.items()):lines.append(f"| {m} | {v['seed_count']} | {v['canonical_em']:.4f} | {v['novel_em']:.4f} | {v['seen_em']:.4f} | {v['trainable_parameters']} |")
 lines+=['','## Gate','']+[f"- [{'x' if x['pass'] else ' '}] `{n}`: {x['value']:.4f} >= {x['threshold']:.4f}" for n,x in g.get('checks',{}).items()]+['','Test split was not opened. STOP forbids learned/Bayesian router development.',''];return '\n'.join(lines)
def write_suite_outputs(root,c):
 runs=discover_runs(root)
 if not runs:raise FileNotFoundError('no runs')
 a=aggregate_runs(runs);g=apply_gate(a,c);payload={'run_count':len(runs),'aggregate':a,'gate':g};write_json(Path(root)/'suite_summary.json',payload);rows=[]
 for m,v in sorted(a.items()):rows.append({'method':m,'seed_count':v['seed_count'],'seeds':';'.join(map(str,v['seeds'])),'canonical_em':v['canonical_em'],'novel_em':v['novel_em'],'seen_em':v['seen_em'],'trainable_parameters':';'.join(map(str,v['trainable_parameters']))})
 write_csv(Path(root)/'suite_metrics.csv',rows);(Path(root)/'REPORT.md').write_text(render(a,g),encoding='utf-8');return payload
