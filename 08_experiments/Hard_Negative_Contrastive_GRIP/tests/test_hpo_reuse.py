"""Behavioral checks for reuse identities, GPU skipping, and disk retention."""
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HNG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HNG / 'scripts'), str(HNG / 'src'),
               str(HNG.parents[1] / '13_base_method/grip-exp')]
import hpo_reuse as reuse


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'GRIP_REUSE_CACHE': str(self.root / 'cache')})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_checksum_reuses_bytes_and_detects_same_size_replacement(self):
        weight = self.root / 'weight'
        weight.write_bytes(b'aaaa')
        before = weight.stat()
        old = reuse.sha256(weight)
        actual_open = Path.open
        reads = []

        def tracked(path, mode='r', *args, **kwargs):
            if path == weight and mode == 'rb':
                reads.append(path)
            return actual_open(path, mode, *args, **kwargs)

        with patch.object(Path, 'open', tracked), \
             patch.object(reuse.time, 'time_ns', return_value=before.st_ctime_ns + 3_000_000_000):
            self.assertEqual(reuse.sha256(weight), old)
        self.assertEqual(reads, [])
        weight.write_bytes(b'bbbb')
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertNotEqual(reuse.sha256(weight), old)

    def test_running_bash_uses_immutable_snapshot_when_original_is_edited(self):
        source = self.root / 'project/configs/run_error_memory_wsl.sh'
        source.parent.mkdir(parents=True)
        ready, release = self.root / 'ready', self.root / 'release'
        payload = '''#!/bin/bash
set -eu
"$PYTHON" -c 'import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); exec("while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)")' "$1" "$2"
printf 'snapshot-ok\\n%s\\n' "$GRIP_HNG_ROOT"
'''
        source.write_text(payload)
        command, environment = reuse.freeze_shell_launch(['bash', str(source), str(ready), str(release)],
            {**os.environ, 'PYTHON': sys.executable})
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            source.write_text('broken "quote\n')
            release.touch()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn('snapshot-ok', stdout)
            self.assertIn(str(source.parent.parent), stdout)
            self.assertEqual(Path(command[1]).read_text(), payload)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_predictions_remap_subset_ids_but_reject_other_prompts_or_split(self):
        identity = {'weights': 'a', 'cohort': 'valid', 'decoder': 32}
        reuse.remember_predictions(identity, [{'question_id': 'screen:0', 'split': 'test',
            'question': 'q; candidates:a;b', 'target': ['a'], 'response': 'a', 'correct': True}])
        samples = [{'question_id': 'full:77', 'split': 'valid',
                    'question': 'q; candidates:a;b', 'answer': 'a'}]
        found = reuse.reusable_predictions(identity, samples)
        self.assertEqual(found[0]['question_id'], 'full:77')
        self.assertEqual(found[0]['split'], 'valid')
        for change in ({'weights': 'b'}, {'cohort': 'test'}, {'decoder': 64}):
            self.assertEqual(reuse.reusable_predictions({**identity, **change}, samples), [])
        self.assertEqual(reuse.reusable_predictions(identity,
            [{**samples[0], 'question': 'q; candidates:b;a'}]), [])

    def test_validation_parent_requires_exact_manifest_subset(self):
        parent, screen = self.root / 'valid.json', self.root / 'screen.json'
        full = {'questions': ['q1', 'q2'], 'answers': ['a', 'b'], 'graph': {}}
        reuse.write(parent, full)
        subset = {**full, 'questions': ['q2'], 'answers': ['b']}
        reuse.write(screen, subset)
        reuse.write(screen.with_suffix('.manifest.json'), {'source': str(parent),
            'source_sha256': reuse.sha256(parent), 'selected_indices': [1]})
        self.assertEqual(reuse.validation_parent(screen, subset), reuse.sha256(parent))
        with self.assertRaisesRegex(ValueError, 'exact subset'):
            reuse.validation_parent(screen, {**subset, 'answers': ['wrong']})

    def test_complete_prediction_reuse_never_loads_gpu_model(self):
        import evaluate_adapter as evaluator
        source, adapter = self.root / 'valid.json', self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'fixture weights')
        base = self.root / 'models/Qwen--Qwen2.5-7B-Instruct'
        base.mkdir(parents=True)
        (base / 'config.json').write_text('{}')
        record = {'id': '0', 'title': 'fixture', 'graph': {},
                  'questions': ['q1', 'q2'], 'answers': ['a', 'b']}
        reuse.write(source, record)
        args = SimpleNamespace(adapter=adapter, eval_file=source,
            output_dir=self.root / 'evaluation', model_cache_dir=self.root / 'models',
            model_name='qwen-7b', dtype='bfloat16', gen_max_length=32,
            resume=True, label='fixture', progress_every=100)
        identity = reuse.inference_identity(adapter, args, record, 'Qwen/Qwen2.5-7B-Instruct')
        reuse.remember_predictions(identity, [
            {'question_id': 'old:' + str(i), 'split': 'test', 'question': q,
             'target': [a], 'response': a, 'raw_response': a, 'correct': True,
             'in_list': True, 'candidate_valid': True}
            for i, (q, a) in enumerate(zip(record['questions'], record['answers']))])
        with patch.object(evaluator, 'parse_args', return_value=args), \
             patch.object(evaluator, 'load_adapter', side_effect=AssertionError('GPU must not load')):
            evaluator.main()
        self.assertEqual(reuse.read(args.output_dir / 'reuse_stats.json')['new_rows_required'], 0)
        self.assertEqual(reuse.read(args.output_dir / 'summary.json')['all']['em'], 1)
        (args.output_dir / 'reuse_identity.json').unlink()
        with patch.object(evaluator, 'parse_args', return_value=args):
            with self.assertRaisesRegex(ValueError, 'full decoder/source identity'):
                evaluator.main()

    def test_score_registry_excludes_wrong_teacher_and_changed_source(self):
        folder = self.root / 'scores'
        folder.mkdir()
        identity = {'split': 'train', 'filter_splits': ['train'], 'teacher': 'a'}
        reuse.write(folder / 'run_stats.json', {'checks_passed': True})
        reuse.write(folder / 'score_identity.json', identity)
        for name in ('candidate_scores.jsonl', 'qa_summary.jsonl'):
            (folder / name).write_text('{}\n')
        reuse.register_scores(identity, folder)
        self.assertEqual(reuse.score_sources(identity, self.root / 'other'), [folder])
        self.assertEqual(reuse.score_sources({**identity, 'teacher': 'b'}, self.root / 'other'), [])
        (folder / 'candidate_scores.jsonl').write_text('changed\n')
        self.assertEqual(reuse.score_sources(identity, self.root / 'other'), [])
        with self.assertRaisesRegex(ValueError, 'training scores'):
            reuse.register_scores({**identity, 'split': 'test'}, folder)

    def test_scoring_subset_reuses_full_archive_without_touching_gpu(self):
        import score_relation_candidates_offline as scorer
        folder, destination = self.root / 'source_scores', self.root / 'subset_scores'
        folder.mkdir()
        identity = {'split': 'train', 'filter_splits': ['train'], 'teacher': 'fixture'}
        vocabulary = ['concept:a', 'concept:b']
        for i in range(2):
            rows, summary = scorer.build_qa_score_rows(qa_id=f'task_qa:{i}', question='q',
                head_entity='h', tail_entity='t', gold_relation='concept:a',
                matched_train_relation='concept:a', relation_order=vocabulary,
                scores={'concept:a': -.1, 'concept:b': -3}, token_lengths={'concept:a': 1, 'concept:b': 1},
                model_checkpoint='fixture')
            scorer.atomic_append_qa_block(folder / 'candidate_scores.jsonl', folder / 'qa_summary.jsonl', rows, summary)
        reuse.write(folder / 'score_identity.json', identity)
        reuse.write(folder / 'run_stats.json', {'checks_passed': True})
        reuse.register_scores(identity, folder)
        args = SimpleNamespace(candidate_batch_size=4, temperature=1., num_shards=1, shard_id=0,
            raw_dir=self.root, relation_text_file=None, entity_text_file=None,
            filter_splits=['train'], relation_eligibility=None, task_file=self.root / 'task.json',
            compare_reference=None, limit=0, output_dir=destination, b1_adapter=self.root / 'adapter',
            dtype='bfloat16', benchmark=False, model_name='qwen-7b', resume=True, progress_every=0,
            split='train', model_cache_dir=self.root / 'models')
        for invocation in range(2):
            with patch.object(scorer, 'parse_args', return_value=args), \
                 patch.object(scorer, 'load_train_relation_order', return_value=vocabulary), \
                 patch.object(scorer, 'known_pair_relations', return_value={}), \
                 patch.object(scorer, 'load_relation_eligibility', return_value=None), \
                 patch.object(scorer, 'load_matchable', return_value=[{'qa_id': 'task_qa:1'}]), \
                 patch.object(scorer.reuse, 'scoring_identity', return_value=identity), \
                 patch.object(scorer, 'load_b1', side_effect=AssertionError('GPU model must not load')), \
                 patch.object(scorer.torch.cuda, 'is_available', side_effect=AssertionError('CUDA must not initialize')):
                scorer.main()
            stats = reuse.read(destination / 'run_stats.json')
            self.assertEqual(stats['qa_count'], 1)
            self.assertEqual(stats['scored_qa_this_invocation'], 0)
            self.assertEqual(stats['reused_cached_qa'], 1 if invocation == 0 else 0)
            self.assertEqual({row['qa_id'] for row in scorer.load_jsonl(destination / 'candidate_scores.jsonl')}, {'task_qa:1'})


class RetentionTests(unittest.TestCase):
    def test_rung_stop_saves_even_between_periodic_checkpoints(self):
        from train_listed_contrastive import StopAfterStepCallback
        state = SimpleNamespace(global_step=8)
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        returned = StopAfterStepCallback(8).on_step_end(SimpleNamespace(save_steps=16), state, control)
        self.assertTrue(returned.should_save)
        self.assertTrue(returned.should_training_stop)

    def test_space_guard_preserves_weights_when_space_is_insufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            weight = out / 'adapter_model.safetensors'
            weight.write_bytes(b'protected')
            with patch.object(reuse.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024)):
                with self.assertRaisesRegex(RuntimeError, 'Insufficient space'):
                    reuse.require_training_space(out)
            self.assertEqual(weight.read_bytes(), b'protected')

    def test_only_global_top2_and_latest_live_checkpoint_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            snapshots = {}
            for trial, step, score in [(0, 8, .80), (0, 16, .90), (1, 8, .85), (2, 8, .70)]:
                folder = out / f'trials/trial_{trial:04d}/step_{step:03d}'
                adapter = folder / 'adapter'
                adapter.mkdir(parents=True)
                snapshots[(trial, step)] = adapter / 'adapter_model.safetensors'
                snapshots[(trial, step)].write_bytes(b'weights')
                reuse.write(folder / 'metrics.json', {'trial': trial, 'step': step,
                    'screen_em': score, 'adapter': str(adapter)})
            for trial in (0, 1, 2):
                for step in (8, 16):
                    folder = out / f'trials/trial_{trial:04d}/train_run/trainer_listed/checkpoint-{step}'
                    folder.mkdir(parents=True)
                    (folder / 'optimizer.pt').write_bytes(b'optimizer')
                    (folder / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
                    for name in ('scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors'):
                        (folder / name).write_bytes(b'resume')
            # A finished evaluation plus complete optimizer/RNG state makes
            # the separately packaged live adapter a redundant copy.
            live = out / 'trials/trial_0001/train_run/listed/adapter'
            live.mkdir(parents=True)
            (live / 'adapter_model.safetensors').write_bytes(b'duplicate')
            checkpoint = out / 'trials/trial_0001/train_run/trainer_listed/checkpoint-16'
            for name in ('scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors'):
                (checkpoint / name).write_bytes(b'resume')
            # Latest step16 is evaluated, but lower scoring than protected8.
            reuse.write(out / 'trials/trial_0001/step_016/metrics.json', {
                'trial': 1, 'step': 16, 'screen_em': .6,
                'adapter': str(out / 'trials/trial_0001/step_016/adapter')})
            states = [SimpleNamespace(number=n, state=SimpleNamespace(name=s))
                      for n, s in [(0, 'COMPLETE'), (1, 'RUNNING'), (2, 'PRUNED')]]
            self.assertGreater(reuse.prune_search_storage(out, states), 0)
            self.assertTrue(snapshots[(0, 16)].exists())
            self.assertTrue(snapshots[(1, 8)].exists())
            self.assertFalse(snapshots[(0, 8)].exists())
            self.assertFalse(snapshots[(2, 8)].exists())
            self.assertFalse((out / 'trials/trial_0001/train_run/trainer_listed/checkpoint-8/optimizer.pt').exists())
            self.assertTrue((out / 'trials/trial_0001/train_run/trainer_listed/checkpoint-16/optimizer.pt').exists())
            self.assertFalse((live / 'adapter_model.safetensors').exists())
            self.assertTrue((out / 'trials/trial_0000/train_run/trainer_listed/checkpoint-16/trainer_state.json').exists())
            self.assertEqual(len(list(out.glob('trials/trial_*/step_*/metrics.json'))), 5)
            self.assertTrue((out / 'storage_cleanup.jsonl').exists())

    def test_incomplete_newer_checkpoint_cannot_delete_the_last_usable_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            root = out / 'trials/trial_0000/train_run/trainer_listed'
            good, partial = root / 'checkpoint-8', root / 'checkpoint-16'
            good.mkdir(parents=True)
            partial.mkdir()
            reuse.write(good / 'trainer_state.json', {'global_step': 8})
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors'):
                (good / name).write_bytes(b'needed')
            (partial / 'adapter_model.safetensors').write_bytes(b'incomplete')
            state = SimpleNamespace(number=0, state=SimpleNamespace(name='RUNNING'))
            self.assertTrue(reuse.complete_checkpoint(good))
            self.assertFalse(reuse.complete_checkpoint(partial))
            reuse.prune_search_storage(out, [state])
            self.assertTrue(reuse.complete_checkpoint(good))
            self.assertTrue((partial / 'adapter_model.safetensors').exists())

    def test_cleanup_refuses_symlink_to_external_paper_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / 'paper'
            paper.mkdir()
            (paper / 'adapter_model.safetensors').write_bytes(b'important')
            out = root / 'search'
            for i, score in enumerate((.9, .8, .7)):
                step = out / f'trials/trial_{i:04d}/step_008'
                step.mkdir(parents=True)
                adapter = step / 'adapter'
                if i == 2:
                    adapter.symlink_to(paper, target_is_directory=True)
                else:
                    adapter.mkdir()
                reuse.write(step / 'metrics.json', {'trial': i, 'step': 8, 'screen_em': score, 'adapter': str(adapter)})
            with self.assertRaisesRegex(ValueError, 'symlink'):
                reuse.prune_search_storage(out, [])
            self.assertEqual((paper / 'adapter_model.safetensors').read_bytes(), b'important')


if __name__ == '__main__':
    unittest.main()
