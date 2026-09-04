import unittest
from query_graph_lora.suite import apply_gate
class T(unittest.TestCase):
 def m(self,e,n,s):return {'seed_count':2,'canonical_em':e,'novel_em':n,'seen_em':s,'trainable_parameters':[100]}
 def c(self):return {'gate':{'minimum_seeds':2,'canonical_gain_pp':2,'novel_gain_pp':2,'maximum_seen_drop_pp':1,'oracle_over_control_pp':1}}
 def test_go(self):
  a={'static_rank8':self.m(.30,.20,.40),'uniform_basis':self.m(.30,.20,.40),'shuffled_oracle_route':self.m(.30,.20,.40),'oracle_hop_route':self.m(.31,.21,.40),'oracle_relation_path_route':self.m(.34,.24,.40)};self.assertEqual(apply_gate(a,self.c())['decision'],'GO_LEARNED_QUERY_ROUTER')
 def test_stop(self):
  a={m:self.m(.3,.2,.4) for m in ['static_rank8','uniform_basis','shuffled_oracle_route','oracle_hop_route','oracle_relation_path_route']};self.assertEqual(apply_gate(a,self.c())['decision'],'STOP_GRAPH_CONDITIONAL_LORA')
if __name__=='__main__':unittest.main()
