import importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'scripts/audit_e03_results.py';s=importlib.util.spec_from_file_location('a',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class T(unittest.TestCase):
 def test_mcnemar(self):self.assertEqual(m.mcnemar(4,2),.6875)
 def test_fixture(self):
  with tempfile.TemporaryDirectory() as td:
   r=Path(td);run=r/'run';run.mkdir();exp={'expected_decision':'STOP','expected_next_action':'HALT','primary_checkpoints':['a','b'],'expected_validation_count':4,'expected_evaluation_splits_opened':['validation']};(r/'e.json').write_text(json.dumps(exp));g={}
   for name,c in [('a',{'d0_correct_to_d1_correct':1,'d0_correct_to_d1_wrong':0,'d0_invalid_to_d1_correct':1,'d0_invalid_to_d1_wrong':1,'d0_valid_wrong_to_d1_correct':0,'d0_valid_wrong_to_d1_wrong':1}),('b',{'d0_correct_to_d1_correct':1,'d0_correct_to_d1_wrong':1,'d0_invalid_to_d1_correct':1,'d0_invalid_to_d1_wrong':0,'d0_valid_wrong_to_d1_correct':0,'d0_valid_wrong_to_d1_wrong':1})]:
    d=run/'phase_a'/name;d.mkdir(parents=True);(d/'d0_to_d1_validation_transitions.json').write_text(json.dumps({'count':4,'counts':c}));(d/'runtime_audit.json').write_text(json.dumps({'evaluation_splits_opened':['validation'],'test_file_opened':False,'inference_graph_access':False,'query_specific_candidate_access':False,'gold_path_in_prompt':False}));d0=c['d0_correct_to_d1_correct']+c['d0_correct_to_d1_wrong'];d1=c['d0_correct_to_d1_correct']+c['d0_invalid_to_d1_correct']+c['d0_valid_wrong_to_d1_correct'];g[name]=100*(d1-d0)/4
   (run/'suite_summary.json').write_text(json.dumps({'gate':{'decision':'STOP','next_action':'HALT','primary_checkpoints':['a','b'],'per_checkpoint':{k:{'canonical_gain_pp':v} for k,v in g.items()},'aggregate':{'mean_canonical_gain_pp':sum(g.values())/2}}}));self.assertEqual(m.audit(run,r/'e.json')['audit_status'],'PASS')
if __name__=='__main__':unittest.main()
