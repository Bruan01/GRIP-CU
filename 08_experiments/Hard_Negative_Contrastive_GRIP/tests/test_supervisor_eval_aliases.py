"""Cached evaluations survive filesystem aliases without weakening identities."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import supervise_nell_fb_hpo as guard


class AliasCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.real = self.root / 'real'
        self.real.mkdir()
        self.alias = self.root / 'alias'
        self.alias.symlink_to(self.real, target_is_directory=True)
        self.adapter = self.real / 'adapter'
        self.adapter.mkdir()
        (self.adapter / 'adapter_config.json').write_text('{}')
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'fixed')
        self.source = self.real / 'valid.json'
        guard.hpo.write_json(self.source, {'questions': ['q'], 'answers': ['a']})
        self.out = self.root / 'evaluation'
        self.identity = {'adapter': str(self.adapter), 'adapter_sha256': guard.fingerprint(self.adapter),
            'eval_file': str(self.source), 'eval_sha256': guard.hpo.sha256_file(self.source),
            'gen_max_length': 32, 'dtype': 'bfloat16', 'label': 'fixture'}
        guard.hpo.write_json(self.out / 'cache_identity.json', self.identity)
        guard.hpo.write_json(self.out / 'summary.json', {'all': {'count': 1, 'em': 1.},
            'adapter': str(self.adapter), 'eval_file': str(self.source)})
        (self.out / 'predictions.jsonl').write_text('{"correct":true}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_alias_cache_returns_without_gpu_or_identity_rewrite(self):
        before = (self.out / 'cache_identity.json').read_bytes()
        with patch.object(guard.hpo, 'run_logged') as run:
            self.assertEqual(guard.evaluate(self.alias / 'adapter', self.alias / 'valid.json', self.out, 'fixture'), 1.)
            run.assert_not_called()
        self.assertEqual((self.out / 'cache_identity.json').read_bytes(), before)

    def test_changed_weights_at_same_alias_are_rejected(self):
        (self.adapter / 'adapter_model.safetensors').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            guard.evaluate(self.alias / 'adapter', self.alias / 'valid.json', self.out, 'fixture')

    def test_different_path_cannot_pass_by_matching_contents(self):
        other = self.root / 'other.json'
        other.write_bytes(self.source.read_bytes())
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            guard.evaluate(self.adapter, other, self.out, 'fixture')

    def test_changed_decoder_is_rejected_even_for_same_file(self):
        changed = {**self.identity, 'gen_max_length': 64}
        self.assertFalse(guard.evaluation_identity_matches(self.identity, changed))

    def test_partial_prediction_count_still_rejected(self):
        (self.out / 'predictions.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            guard.evaluate(self.alias / 'adapter', self.alias / 'valid.json', self.out, 'fixture')


if __name__ == '__main__':
    unittest.main()
