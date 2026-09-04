import unittest
from collections import Counter
from query_graph_lora.route_artifacts import build_route_tables
class T(unittest.TestCase):
 def rows(self,p,n):return [{'task_id':f'{p}{i}','depth_label':i%4+1,'path_relations':[f'r{i%5}',f'r{(i+1)%7}']} for i in range(n)]
 def test_deterministic(self):
  a=build_route_tables(self.rows('t',20),self.rows('v',8),4,43,16);b=build_route_tables(self.rows('t',20),self.rows('v',8),4,43,16);self.assertEqual(a['routes'],b['routes']);self.assertFalse(a['audit']['test_opened'])
 def test_balanced_train_clusters(self):
  x=build_route_tables(self.rows('t',716),self.rows('v',12),4,43,16)
  self.assertEqual(sorted(x['models']['relation_path']['train_cluster_counts'].values()),[179,179,179,179])
  self.assertEqual(sorted(x['models']['relation_family']['train_cluster_counts'].values()),[179,179,179,179])
 def test_shuffle_distribution(self):
  x=build_route_tables(self.rows('t',24),self.rows('v',12),4,43,16)['routes']['validation'];self.assertEqual(Counter(v['shuffled_oracle_route'] for v in x.values()),Counter(v['oracle_relation_path_route'] for v in x.values()))
if __name__=='__main__':unittest.main()
