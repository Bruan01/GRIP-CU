"""CPU-only evidence gates and bounded recovery for the completion monitor."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import completion_evidence as e
import watch_research_completion as watcher


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out, self.baseline, self.archive = (self.root / n for n in ('search', 'baseline', 'archive'))
        self.selected, self.b1 = (self.root / n for n in ('selected_adapter', 'baseline_adapter'))
        for folder in (self.selected, self.b1):
            folder.mkdir()
            (folder / 'adapter_config.json').write_text('{}')
            (folder / 'adapter_model.safetensors').write_bytes(folder.name.encode())
        self.test = self.root / 'test.json'
        e.reuse.write(self.test, {'questions': ['q1', 'q2'], 'answers': ['a', 'b']})
        self.test_hash, self.b1_hash = e.reuse.sha256(self.test), e.reuse.fingerprint(self.b1)
        self.record = {'dataset': 'fb15k237_2', 'test': str(self.test), 'b1_adapter': str(self.b1),
                       'paper_reference': .8875}
        winner = {'adapter': str(self.selected), 'adapter_sha256': e.reuse.fingerprint(self.selected)}
        self.selection = {'winner': winner, 'search_stopped_before_test': True,
                          'test_used_in_this_search_selection': False}
        e.reuse.write(self.out / 'selection.json', self.selection)
        e.reuse.write(self.out / 'final_result.json', {'winner': winner, 'seed': 2026,
            'selected_test_em': 1., 'b1_test_em': .5, 'test_count': 2})
        for folder, adapter, responses in ((self.baseline, self.b1, ['a', 'x']),
                                           (self.out / 'selected_test', self.selected, ['a', 'b'])):
            rows = [{'question_id': str(i), 'question': f'q{i+1}', 'target': [answer],
                     'response': response, 'correct': answer == response}
                    for i, (answer, response) in enumerate(zip(['a', 'b'], responses))]
            e.reuse.write(folder / 'summary.json', {'all': {'count': 2, 'em': sum(r['correct'] for r in rows)/2}})
            e.reuse.write(folder / 'cache_identity.json', {'adapter_sha256': e.reuse.fingerprint(adapter),
                'eval_sha256': self.test_hash, 'gen_max_length': 32, 'dtype': 'bfloat16'})
            (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self):
        return e.audit_selected(self.out, self.record, self.baseline, self.archive, self.b1_hash, self.test_hash)

    def test_complete_evidence_is_recomputed_retained_and_reused(self):
        report = self.audit()
        self.assertEqual(report['paired_counts'], {'both_correct': 1, 'rescue': 1})
        self.assertTrue(report['target_achieved'])
        self.assertTrue((self.archive/'adapter/adapter_model.safetensors').samefile(self.selected/'adapter_model.safetensors'))
        self.assertEqual(self.audit(), report)

    def test_incomplete_predictions_cannot_be_marked_complete(self):
        (self.out/'selected_test/predictions.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.audit()
        self.assertFalse((self.archive/'ARCHIVE.json').exists())

    def test_changed_candidates_cannot_be_called_matched(self):
        path = self.out/'selected_test/predictions.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['question'] = 'different candidate order'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'Paired prompts'):
            self.audit()

    def test_saved_correctness_is_not_trusted(self):
        path = self.out/'selected_test/predictions.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['response'] = 'wrong'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'correctness flag'):
            self.audit()

    def test_test_guided_selection_is_rejected(self):
        e.reuse.write(self.out/'selection.json', {**self.selection, 'test_used_in_this_search_selection': True})
        with self.assertRaisesRegex(ValueError, 'Test was used'):
            self.audit()

    def test_changed_weight_and_decoder_identities_are_rejected(self):
        path = self.out/'selected_test/cache_identity.json'
        identity = e.reuse.read(path)
        for changed, message in (({'adapter_sha256': {}}, 'identity mismatch'),
                                 ({'gen_max_length': 64}, 'Decoder changed')):
            e.reuse.write(path, {**identity, **changed})
            with self.assertRaisesRegex(ValueError, message):
                self.audit()

    def test_retained_evidence_corruption_is_not_silently_reused(self):
        report = self.audit()
        name = next(iter(report['retained_evidence_sha256']))
        (self.archive/name).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Retained evidence changed'):
            self.audit()

    def test_restart_waits_for_gpu_and_live_workers_and_stops_at_budget(self):
        for alive, locked, gpu, restarts, expected in (
            (True, False, [], 0, 'monitoring'),
            (False, True, [], 0, 'waiting_for_existing_work'),
            (False, False, ['17'], 0, 'waiting_for_existing_work'),
            (False, False, [], 0, 'resume_nell'),
            (False, False, [], 3, 'retry_limit_reached')):
            self.assertEqual(watcher.recovery_decision({}, alive, locked, gpu, restarts), expected)
        self.assertEqual(watcher.recovery_decision({'phase': 'completed'}, False, False, [], 0), 'finished')
        self.assertEqual(watcher.recovery_decision({'phase': 'no_validation_improvement'}, False, False, [], 0), 'finished')

    def test_exhausted_main_recovery_is_reported_as_needing_attention(self):
        campaign, nell = self.root/'campaign', self.root/'nell'
        nell.mkdir()
        e.reuse.write(campaign/'watchdog.json', {'action': 'retry_limit_reached'})
        with patch.object(watcher.main_watch, 'gpu_processes', return_value=['17']):
            report = watcher.inspect(campaign, nell, 0)
        self.assertEqual(report['phase'], 'needs_attention')
        self.assertIn('Main queue', report['error'])
        self.assertFalse(report['all_three_test_targets_achieved'])


if __name__ == '__main__':
    unittest.main()
