import unittest
from query_graph_lora.routing import build_route_masks,route_access
class T(unittest.TestCase):
 def test_uniform(self):self.assertEqual(build_route_masks('uniform_basis',[2]).forward[0],(1.,1.,1.,1.))
 def test_onehot(self):self.assertEqual(build_route_masks('oracle_hop_route',[2]).forward[0],(0.,0.,1.,0.))
 def test_access(self):self.assertTrue(route_access('oracle_relation_path_route')['oracle_route_access']);self.assertFalse(route_access('static_rank8')['oracle_route_access'])
if __name__=='__main__':unittest.main()
