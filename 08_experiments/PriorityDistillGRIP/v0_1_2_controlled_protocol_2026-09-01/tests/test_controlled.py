import json
import unittest
from pathlib import Path
from priority_distill.config import load_config

ROOT = Path(__file__).resolve().parents[1]

class ControlledConfigTests(unittest.TestCase):
    def test_direct_and_oracle_configs_are_distinct(self):
        direct = load_config(ROOT / 'configs/direct_answer_only.json')
        oracle = load_config(ROOT / 'configs/oracle_two_stage.json')
        self.assertEqual(direct['protocol'], 'direct_answer_only')
        self.assertEqual(oracle['protocol'], 'oracle_two_stage')
        self.assertEqual(direct['training']['stage2_epochs'], oracle['training']['stage2_epochs'])
        self.assertEqual(direct['training']['tokens_per_optimizer_step'], oracle['training']['tokens_per_optimizer_step'])
        self.assertEqual(direct['training']['first_segment_learning_rate'], oracle['training']['first_segment_learning_rate'])
        self.assertEqual(direct['training']['answer_only_learning_rate'], oracle['training']['answer_only_learning_rate'])

    def test_test_selection_is_not_in_config(self):
        config = json.loads((ROOT / 'configs/direct_answer_only.json').read_text())
        self.assertNotIn('test_checkpoint', config)
        self.assertEqual(config['diagnostic']['splits'], ['validation', 'test'])

if __name__ == '__main__':
    unittest.main()
