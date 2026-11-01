"""CPU-only tests for validation-gated early confirmation and test isolation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import confirm_stage2_candidates as confirm
guard = confirm.guard


def adapter(path, weights):
    path.mkdir(parents=True)
    guard.hpo.write_json(path / 'adapter_config.json', {'r': 4, 'lora_alpha': 8})
    (path / 'adapter_model.safetensors').write_bytes(weights)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.out = self.root / 'search'
        self.out.mkdir()
        adapter(self.root / 's1', b'stage1')
        adapter(self.root / 'b1', b'baseline')
        for source in ['valid', 'test']:
            guard.hpo.write_json(self.root / f'{source}.json', {'questions': ['q'], 'answers': ['a']})
        self.record = {'dataset': 'nell23k', 'search_dir': str(self.out),
            's1_adapter': str(self.root / 's1'), 'b1_adapter': str(self.root / 'b1'),
            'valid': str(self.root / 'valid.json'), 'test': str(self.root / 'test.json'),
            'paper_reference': .8774, 'input': str(self.root / 'tasks.json'),
            'raw': str(self.root / 'raw'), 'sampler': str(self.root / 'bank'),
            'expected_listed': 1, 'epochs': 10, 'rungs': [8, 16, 32]}
        guard.hpo.write_json(self.out / 'fixed_stage1_identity.json', guard.fingerprint(self.root / 's1'))
        guard.hpo.write_json(self.out / 'search_progress.json', {'trials_recorded': 2, 'resource_cap': 32})
        guard.hpo.write_json(self.out / 'b1_screen/summary.json', {'all': {'em': .849}})
        self.candidates = []
        for trial, step, em in [(0, 16, .862), (1, 32, .864)]:
            path = self.out / f'trials/trial_{trial:04d}/step_{step:03d}/adapter'
            adapter(path, f'trial{trial}'.encode())
            self.candidates.append(path)
            guard.hpo.write_json(path.parent / 'metrics.json', {'trial': trial, 'step': step,
                'screen_em': em, 'b1_screen_em': .849, 'adapter': str(path),
                'adapter_sha256': guard.fingerprint(path)})

    def tearDown(self):
        self.temporary.cleanup()

    def evaluator(self, candidate_validation=.87, candidate_test=.89, interrupt=False):
        calls = []

        def fake(model, source, destination, label):
            calls.append((model, source))
            if source == Path(self.record['test']):
                selection = guard.read(self.out / 'selection.json')
                self.assertTrue(selection['search_stopped_before_test'])
                self.assertFalse(selection['test_used_in_this_search_selection'])
                if model != self.root / 'b1' and interrupt:
                    raise InterruptedError('test inference interrupted')
                return .88 if model == self.root / 'b1' else candidate_test
            return .85 if model == self.root / 'b1' else candidate_validation

        return calls, fake

    def test_small_full_validation_gain_does_not_open_test_or_end_hpo(self):
        calls, evaluate = self.evaluator(candidate_validation=.854)
        with patch.object(guard, 'evaluate', side_effect=evaluate):
            decision = confirm.confirm(self.record)
        self.assertFalse(decision['promote'])
        self.assertFalse(any(source == Path(self.record['test']) for _, source in calls))
        self.assertFalse((self.out / 'selection.json').exists())
        self.assertFalse((self.out / 'no_go.json').exists())
        self.assertFalse(confirm.eligible(self.record))

    def test_test_loss_cannot_reopen_search_or_change_selection(self):
        calls, evaluate = self.evaluator(candidate_test=.87)
        with patch.object(guard, 'evaluate', side_effect=evaluate):
            result = confirm.confirm(self.record)
        self.assertFalse(result['target_achieved'])
        self.assertTrue(result['search_stopped_before_test'])
        self.assertTrue(confirm.locked_selection(self.out))
        with patch.object(guard.hpo, 'train_rung', side_effect=AssertionError('training forbidden')):
            guard.fb_search(self.root, self.record)
        with patch.object(guard, 'evaluate', side_effect=AssertionError('finished test must be reused')):
            self.assertEqual(confirm.confirm(self.record)['selected_test_em'], .87)

    def test_interrupted_test_resumes_locked_winner_without_reselecting(self):
        calls, evaluate = self.evaluator(interrupt=True)
        with patch.object(guard, 'evaluate', side_effect=evaluate):
            with self.assertRaises(InterruptedError):
                confirm.confirm(self.record)
        winner = guard.read(self.out / 'selection.json')['winner']['adapter']
        with self.assertRaisesRegex(RuntimeError, 'locked before test'):
            guard.fb_search(self.root, self.record)
        calls2, evaluate2 = self.evaluator()
        with patch.object(guard, 'evaluate', side_effect=evaluate2):
            result = confirm.confirm(self.record)
        self.assertEqual(result['winner']['adapter'], winner)
        self.assertTrue(all(source == Path(self.record['test']) for _, source in calls2))

    def test_locked_checkpoint_mutation_fails_before_another_test(self):
        _, evaluate = self.evaluator(interrupt=True)
        with patch.object(guard, 'evaluate', side_effect=evaluate):
            with self.assertRaises(InterruptedError):
                confirm.confirm(self.record)
        winner = Path(guard.read(self.out / 'selection.json')['winner']['adapter'])
        (winner / 'adapter_model.safetensors').write_bytes(b'changed')
        with patch.object(guard, 'evaluate', side_effect=AssertionError('mutated model cannot test')):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                confirm.confirm(self.record)

    def test_partial_first_round_is_not_eligible(self):
        guard.hpo.write_json(self.out / 'search_progress.json', {'trials_recorded': 1, 'resource_cap': 16})
        self.assertFalse(confirm.eligible(self.record))

    def test_validation_stop_closes_optimizer_trials_before_test(self):
        import optuna
        study = optuna.create_study(study_name='nell23k_sccl_validation_checkpoint_v1',
            storage=f'sqlite:///{self.out / "study.sqlite3"}')
        for _ in range(2):
            study.ask().report(.86, 32)
        _, evaluate = self.evaluator()

        def checked(model, source, destination, label):
            if source == Path(self.record['test']):
                self.assertFalse(study.get_trials(states=(optuna.trial.TrialState.RUNNING,)))
            return evaluate(model, source, destination, label)

        with patch.object(guard, 'evaluate', side_effect=checked):
            confirm.confirm(self.record)
        self.assertEqual(len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,))), 2)


if __name__ == '__main__':
    unittest.main()
