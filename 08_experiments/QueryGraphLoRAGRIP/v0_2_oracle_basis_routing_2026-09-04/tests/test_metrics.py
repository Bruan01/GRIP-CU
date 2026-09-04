import unittest
from query_graph_lora.metrics import score_predictions
class T(unittest.TestCase):
 def test_composition(self):
  m=score_predictions([{'answer':'concept_a','prediction_text':'concept_a','depth_label':1,'composition_status':'seen'},{'answer':'concept_b','prediction_text':'concept_x','depth_label':2,'composition_status':'novel'}]);self.assertEqual(m['canonical_em'],.5);self.assertEqual(m['canonical_em_by_composition']['novel'],0)
if __name__=='__main__':unittest.main()
