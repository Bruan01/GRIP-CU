"""CPU checks for test gating, queue safety and locked refinement resumption."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refine_nell_stage2 as r


class RefinementTests(unittest.TestCase):
    def test_queue_needs_terminal_state_and_no_unresolved_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r.write(root / 'status.json', {'action': 'codexm_search_early_confirmation'})
            self.assertFalse(r.queue_finished(root))
            r.write(root / 'status.json', {'action': r.TERMINAL[0]})
            self.assertTrue(r.queue_finished(root))
            r.write(root / 'fb15k237_2/failure.json', {'error': 'operational failure'})
            self.assertFalse(r.queue_finished(root))
            r.write(root / 'fb15k237_2/failure.json', {'resolved': True})
            self.assertTrue(r.queue_finished(root))

    def test_changed_assets_cannot_resume_a_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(r, 'contract', return_value={'weights': 'a'}):
                r.freeze_contract(out, out)
            with patch.object(r, 'contract', return_value={'weights': 'b'}):
                with self.assertRaisesRegex(ValueError, 'identities changed'):
                    r.freeze_contract(out, out)

    def test_contract_accepts_real_aliases_without_rewriting_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real, alias = root / 'real', root / 'alias'
            real.mkdir()
            alias.symlink_to(real, target_is_directory=True)
            asset = real / 'data.json'
            asset.write_text('same input')
            old = {'parent_campaign': str(real), 'parent_selection': str(asset),
                   'parent_adapter': str(real), 'record': {'input': str(asset), 'seed': 2026},
                   'asset_sha256': {str(asset): 'fixed-hash'}, 'params': {'learning_rate': .0002}}
            new = {**old, 'parent_campaign': str(alias), 'parent_selection': str(alias / 'data.json'),
                   'parent_adapter': str(alias), 'record': {'input': str(alias / 'data.json'), 'seed': 2026},
                   'asset_sha256': {str(alias / 'data.json'): 'fixed-hash'}}
            r.write(root / 'contract.json', old)
            before = (root / 'contract.json').read_bytes()
            with patch.object(r, 'contract', return_value=new):
                self.assertEqual(r.freeze_contract(root, root), old)
            self.assertEqual((root / 'contract.json').read_bytes(), before)
            for changed in ({**new, 'asset_sha256': {str(alias / 'data.json'): 'changed'}},
                            {**new, 'params': {'learning_rate': .0003}},
                            {**new, 'record': {**new['record'], 'seed': 2027}}):
                self.assertFalse(r.contracts_match(old, changed))
            other = root / 'different.json'
            other.write_bytes(asset.read_bytes())
            self.assertFalse(r.contracts_match(old, {**new, 'asset_sha256': {str(other): 'fixed-hash'}}))

    def fixture(self, out, screen=0.87):
        record = {key: str(out / key) for key in
                  ('input', 'valid', 'test', 'raw', 'relation_eligibility', 'b1_adapter', 'sampler')}
        record.update(epochs=10, expected_listed=2974)
        parent = out / 'parent'
        parent.mkdir()
        (parent / 'adapter_config.json').write_text('{"r": 4, "lora_alpha": 8}')
        (parent / 'adapter_model.safetensors').write_bytes(b'preserved-parent-fixture')
        c = {'record': record, 'parent_adapter': str(parent), 'parent_adapter_sha256': r.guard.fingerprint(parent), 'params': {},
             'additional_steps': [8, 16, 32], 'parent_screen_em': 0.864,
             'parent_full_validation_em': 0.8765656565656565,
             'minimum_full_validation_gain': 0.002, 'scope': 'adaptive exploration'}
        for step in c['additional_steps']:
            r.write(out / f'trials/trial_0000/step_{step:03d}/metrics.json',
                    {'trial': 0, 'step': step, 'screen_em': screen,
                     'adapter': str(out / f'trials/trial_0000/step_{step:03d}/adapter'),
                     'adapter_sha256': {'weights': 'fixed'}})
        return c

    def run_gate(self, out, c, full_em):
        with patch.object(r.hpo, 'validate_assets'), patch.object(r.hpo, 'prepare_validation_screen'), \
             patch.object(r.hpo, 'train_rung') as train, patch.object(r.guard, 'evaluate', return_value=full_em) as evaluate, \
             patch.object(r, 'cleanup'), patch.object(r, 'finish_selected', return_value='tested') as finish:
            result = r.refine(c, out)
            train.assert_not_called()
            return result, evaluate, finish

    def test_no_test_if_complete_validation_fails_to_beat_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c = self.fixture(out)
            result, evaluate, finish = self.run_gate(out, c, 0.877)
            self.assertFalse(result['promote'])
            evaluate.assert_called_once()
            self.assertEqual(evaluate.call_args.args[1], Path(c['record']['valid']))
            finish.assert_not_called()
            self.assertTrue((out / 'no_go.json').exists())
            self.assertFalse((out / 'selection.json').exists())

    def test_no_full_validation_or_test_if_all_screen_scores_decline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c = self.fixture(out, screen=0.86)
            _, evaluate, finish = self.run_gate(out, c, 0.99)
            evaluate.assert_not_called()
            finish.assert_not_called()

    def test_selection_is_persisted_before_any_test_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c = self.fixture(out)

            def inspect_locked(c, target, decision):
                self.assertEqual(json.loads((target / 'selection.json').read_text()), decision)
                self.assertTrue(decision['search_stopped_before_test'])
                self.assertTrue(decision['adaptive_after_test_exposure'])
                self.assertFalse(decision['test_used_for_checkpoint_selection'])
                return 'locked'

            with patch.object(r.hpo, 'validate_assets'), patch.object(r.hpo, 'prepare_validation_screen'), \
                 patch.object(r, 'cleanup'), patch.object(r.guard, 'evaluate', return_value=0.88), \
                 patch.object(r, 'finish_selected', side_effect=inspect_locked):
                self.assertEqual(r.refine(c, out), 'locked')

    def test_locked_test_resume_cannot_reopen_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            r.write(out / 'selection.json', {'winner': 'fixed'})
            with patch.object(r, 'finish_selected', return_value='resumed') as finish, \
                 patch.object(r.hpo, 'train_rung') as train:
                self.assertEqual(r.refine({}, out), 'resumed')
                finish.assert_called_once()
                train.assert_not_called()

    def test_training_launch_uses_warmstart_and_explicit_resume_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.fixture(root)
            params = dict(lambda_candidate=0.11229873660012951, temperature=0.8863961214238169,
                          learning_rate=0.0002, weight_decay=8.889902164812924e-05)
            with patch.object(r.hpo, 'run_logged') as launch:
                for resume in (None, root / 'checkpoint-8'):
                    r.hpo.train_rung(python=r.guard.PYTHON, dataset='nell23k',
                        paths={**c['record'], 'save_steps': 16}, adapter=Path(c['parent_adapter']),
                        sampler=root / 'bank', model_cache=root / 'models', trial_dir=root / 'trial',
                        trial_number=0, rung_step=16, fresh=resume is None,
                        resume_checkpoint=resume, params=params)
                    env = launch.call_args.args[1]
                    self.assertEqual(env['S1_ADAPTER'], c['parent_adapter'])
                    self.assertEqual(env['STEP'], 'train')
                    self.assertEqual(env['STOP_AFTER_STEP'], '16')
                    self.assertEqual(env['TRAIN_LEARNING_RATE'], '0.0002')
                    self.assertIn('13_base_method/grip-exp', env['PYTHONPATH'])
                    if resume is None:
                        self.assertEqual(env['NO_RESUME'], '1')
                        self.assertNotIn('RESUME_FROM_CHECKPOINT', env)
                    else:
                        self.assertEqual(env['RESUME_FROM_CHECKPOINT'], str(resume))

    def test_locked_adapter_identity_mismatch_blocks_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.fixture(root)
            (Path(c['parent_adapter']) / 'adapter_model.safetensors').write_bytes(b'changed')
            with patch.object(r.guard, 'evaluate') as evaluate:
                with self.assertRaisesRegex(ValueError, 'parent adapter changed'):
                    r.finish_selected(c, root, {'winner': {}})
                evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
