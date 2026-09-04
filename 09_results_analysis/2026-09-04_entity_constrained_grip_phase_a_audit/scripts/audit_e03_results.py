#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
def mcnemar(up,down):
 n=up+down
 if not n:return 1.0
 k=min(up,down);return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def audit(run,expected_path):
 exp=read(expected_path);suite=read(run/'suite_summary.json');primary=exp['primary_checkpoints'];checks={}
 tests=sorted(str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and 'test' in p.name.lower() and 'prediction' in p.name.lower())
 checks['no_test_predictions']=not tests;checks['decision_matches']=suite['gate']['decision']==exp['expected_decision'];checks['next_action_matches']=suite['gate']['next_action']==exp['expected_next_action'];checks['primary_matches']=suite['gate']['primary_checkpoints']==primary
 per={};up=down=total=0
 for name in primary:
  d=run/'phase_a'/name;t=read(d/'d0_to_d1_validation_transitions.json');c=t['counts'];n=int(t['count'])
  d0=c['d0_correct_to_d1_correct']+c['d0_correct_to_d1_wrong'];d1=c['d0_correct_to_d1_correct']+c['d0_invalid_to_d1_correct']+c['d0_valid_wrong_to_d1_correct'];gain=100*(d1-d0)/n
  u=c['d0_invalid_to_d1_correct']+c['d0_valid_wrong_to_d1_correct'];q=c['d0_correct_to_d1_wrong'];up+=u;down+=q;total+=n
  r=read(d/'runtime_audit.json');protocol=r.get('evaluation_splits_opened')==exp['expected_evaluation_splits_opened'] and all(r.get(k) is False for k in ['test_file_opened','inference_graph_access','query_specific_candidate_access','gold_path_in_prompt'])
  reported=suite['gate']['per_checkpoint'][name]['canonical_gain_pp'];per[name]={'count':n,'d0_correct':d0,'d1_correct':d1,'gain_pp':gain,'reported_gain_pp':reported,'gain_matches':math.isclose(gain,reported,abs_tol=1e-9),'improvements':u,'regressions':q,'protocol_ok':protocol}
 mean=sum(x['gain_pp'] for x in per.values())/len(per);reported=suite['gate']['aggregate']['mean_canonical_gain_pp']
 metric={'validation_total':total,'improvements':up,'regressions':down,'net_correct_gain':up-down,'net_gain_pp':100*(up-down)/total,'mean_checkpoint_gain_pp':mean,'reported_mean_checkpoint_gain_pp':reported,'mean_matches':math.isclose(mean,reported,abs_tol=1e-9),'exact_mcnemar_two_sided_p':mcnemar(up,down),'per_checkpoint':per}
 checks['checkpoint_metrics_reproduced']=all(x['gain_matches'] for x in per.values());checks['aggregate_reproduced']=metric['mean_matches'];checks['runtime_protocol_pass']=all(x['protocol_ok'] for x in per.values());checks['validation_count_matches']=all(x['count']==exp['expected_validation_count'] for x in per.values())
 return {'audit_status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'unexpected_test_predictions':tests,'metric_reproduction':metric,'interpretation':'D1主要修复实体合法性，semantic EM增益小且配对检验不显著；保留E03作为机制边界证据，停止D2/D3/Phase-B。'}
def render(a):
 r=a['metric_reproduction'];lines=['# E03 Phase-A Reproduction Audit','',f"- Audit: **{a['audit_status']}**",f"- Improvements/regressions: **{r['improvements']}/{r['regressions']}**",f"- Net gain: **{r['net_gain_pp']:.3f} pp**",f"- Exact McNemar p: **{r['exact_mcnemar_two_sided_p']:.4f}**",'','## Checks','']+[f"- [{'x' if v else ' '}] `{k}`" for k,v in a['checks'].items()]+['','## Decision','',a['interpretation'],''];return '\n'.join(lines)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--expected',type=Path,default=Path(__file__).resolve().parents[1]/'EXPECTED_PROTOCOL.json');p.add_argument('--output-dir',type=Path);a=p.parse_args();run=a.run_dir.resolve();out=(a.output_dir or run/'e03_audit').resolve();out.mkdir(parents=True,exist_ok=False);result=audit(run,a.expected.resolve());write(out/'E03_RESULT_AUDIT.json',result);write(out/'metric_reproduction.json',result['metric_reproduction']);(out/'E03_RESULT_AUDIT.md').write_text(render(result),encoding='utf-8');files=[{'path':str(x.relative_to(run)),'size_bytes':x.stat().st_size,'sha256':digest(x)} for x in sorted(run.rglob('*')) if x.is_file() and out not in x.parents];write(out/'artifact_manifest.json',{'file_count':len(files),'files':files});print('E03_AUDIT_'+result['audit_status'],out);raise SystemExit(0 if result['audit_status']=='PASS' else 2)
if __name__=='__main__':main()
