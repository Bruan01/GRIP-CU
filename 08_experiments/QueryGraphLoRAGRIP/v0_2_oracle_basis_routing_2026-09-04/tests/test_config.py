import json,unittest
from pathlib import Path
from query_graph_lora.config import load_config,validate_config
P=Path(__file__).resolve().parents[1]/'configs/phase_a_oracle.json'
class T(unittest.TestCase):
 def test_ok(self):load_config(P)
 def test_reject_test(self):
  c=json.loads(P.read_text());c['evaluation_splits']=['validation','test']
  with self.assertRaises(ValueError):validate_config(c)
if __name__=='__main__':unittest.main()
