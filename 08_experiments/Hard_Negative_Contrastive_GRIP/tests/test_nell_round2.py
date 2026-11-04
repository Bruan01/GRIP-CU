"""CPU checks of stricter validation gating and locked Stage2 search resumption."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import search_nell_stage2_round2 as r
import watch_research_completion as watcher


class Round2Tests(unittest.TestCase):
    def fixture(self, out, screen=.87):
        record = {'valid': str(out/'valid.json'), 'test': str(out/'test.json'), 'search_dir': str(out/'parent')}
        c = {'record': record, 'trial_budget': 2, 'rungs': [16,32,48], 'resource_cap': 48,
             'prune_after': 48, 'initial_trials': [{},{}], 'screen_size': 1000,
             'screen_gate_em': .864, 'full_validation_incumbent_em': .8814141414141414,
             'full_validation_target_em': .8834141414141414, 'matched_b1_sha256': {},
             'asset_sha256': {record['test']: 'test-hash'}, 'scope': 'adaptive exploration'}
        for trial in range(2):
            folder = out/f'trials/trial_{trial:04d}/step_048'
            r.guard.hpo.write_json(folder/'metrics.json', {'trial': trial, 'step': 48,
                'screen_em': screen, 'adapter': str(folder/'adapter'), 'adapter_sha256': {}})
        return c

    def test_gain_over_original_but_not_validation_incumbent_does_not_open_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp);c = self.fixture(out)
            with patch.object(r.guard, 'fb_search'), patch.object(r, 'prepare', return_value=c), \
                 patch.object(r.guard, 'evaluate', return_value=.88) as evaluate, \
                 patch.object(r, 'finish') as finish:
                result = r.run(out, out, c)
            self.assertFalse(result['promote'])
            self.assertTrue((out/'no_go.json').exists())
            self.assertTrue(all(call.args[1]==Path(c['record']['valid']) for call in evaluate.call_args_list))
            finish.assert_not_called()

    def test_below_screen_gate_skips_full_validation_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp);c = self.fixture(out, screen=.86)
            with patch.object(r.guard, 'fb_search'), patch.object(r, 'prepare', return_value=c), \
                 patch.object(r.guard, 'evaluate') as evaluate, patch.object(r, 'finish') as finish:
                r.run(out, out, c)
            evaluate.assert_not_called();finish.assert_not_called()

    def test_selection_is_frozen_before_test_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp);c = self.fixture(out)
            def dispatch(contract, destination, selection):
                self.assertEqual(r.guard.read(destination/'selection.json'), selection)
                self.assertTrue(selection['search_stopped_before_test'])
                self.assertFalse(selection['test_used_for_checkpoint_selection'])
                return 'locked'
            with patch.object(r.guard, 'fb_search'), patch.object(r, 'prepare', return_value=c), \
                 patch.object(r.guard, 'evaluate', return_value=.89), patch.object(r, 'finish', side_effect=dispatch):
                self.assertEqual(r.run(out, out, c), 'locked')

    def test_locked_resume_does_not_reopen_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            r.guard.hpo.write_json(out/'selection.json', {'winner': 'fixed'})
            with patch.object(r.guard, 'fb_search') as train, patch.object(r, 'finish', return_value='resumed'):
                self.assertEqual(r.run(out, out, {}), 'resumed')
            train.assert_not_called()

    def test_monitor_restarts_the_configured_script_and_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(watcher.subprocess, 'run') as launch:
                launch.return_value.returncode=1
                self.assertTrue(watcher.launch_nell(root, root, Path(r.__file__), 'round2-test-session'))
            args = launch.call_args_list[-1].args[0]
            self.assertIn('round2-test-session', args)
            self.assertIn('search_nell_stage2_round2.py', args[-1])


if __name__=='__main__':
    unittest.main()
