"""CPU-only, identity-bound reuse and storage management for Stage-2 HPO."""
from __future__ import annotations

import contextlib
import ast
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

DEFAULT_CACHE = Path('/home/kieran/grip-cu-runs/shared_cache')


def cache_root():
    return Path(os.environ.get('GRIP_REUSE_CACHE', str(DEFAULT_CACHE)))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def sha256(path):
    """Reuse a checksum only while the file's inode and change times agree."""
    path = Path(path).resolve()
    manifest = cache_root() / 'checksums.json'
    with locked(manifest.with_suffix('.lock')):
        records = read(manifest) if manifest.exists() else {}
        before = signature(path)
        cached = records.get(str(path))
        # WSL/shared filesystems can expose coarse change-time ticks. Never
        # trust a memo for a just-written file, even if mtime was restored.
        settled = time.time_ns() - max(before[3], before[4]) >= 2_000_000_000
        if cached and cached['stat'] == before and settled:
            return cached['sha256']
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(block)
        if signature(path) != before:
            raise RuntimeError(f'File changed while hashing: {path}')
        records[str(path)] = {'stat': before, 'sha256': checksum.hexdigest()}
        write(manifest, records)
        return checksum.hexdigest()


def fingerprint(adapter):
    return {name: sha256(Path(adapter) / name) for name in
            ('adapter_config.json', 'adapter_model.safetensors')}


def freeze_shell_launch(command, environment):
    """Bash reads after a child returns: never execute a mutable runner file."""
    environment = environment.copy()
    if (len(command) < 2 or Path(command[0]).name != 'bash'
            or Path(command[1]).name != 'run_error_memory_wsl.sh'):
        return command, environment
    source = Path(command[1]).resolve()
    payload = source.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    snapshot = cache_root() / 'runners' / checksum / source.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with locked(snapshot.with_suffix('.lock')):
        if snapshot.exists():
            if snapshot.read_bytes() != payload:
                raise ValueError('Immutable runner snapshot changed')
        else:
            temporary = snapshot.with_name(f'{snapshot.name}.{os.getpid()}.tmp')
            temporary.write_bytes(payload)
            temporary.chmod(0o444)
            temporary.replace(snapshot)
    environment['GRIP_HNG_ROOT'] = str(source.parent.parent)
    environment['GRIP_RUNNER_SOURCE_SHA256'] = checksum
    return [command[0], str(snapshot), *command[2:]], environment


def runtime_versions():
    return {name: version(name) for name in ('torch', 'transformers', 'peft', 'tokenizers')}


def scoring_rule_digest():
    tree = ast.parse(Path(__file__).with_name('score_relation_candidates_offline.py').read_text())
    return digest([ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('score_relations', 'score_item')])


def validation_parent(source, record):
    """A screen can share only its verified source cohort, never another split."""
    source = Path(source)
    manifest = source.with_suffix('.manifest.json')
    if not manifest.exists():
        return sha256(source)
    info = read(manifest)
    parent = Path(info['source'])
    if sha256(parent) != info['source_sha256']:
        raise ValueError('Validation screen parent changed')
    original = read(parent)
    indices = info['selected_indices']
    expected = {**original, 'questions': [original['questions'][i] for i in indices],
                'answers': [original['answers'][i] for i in indices]}
    if expected != record:
        raise ValueError('Validation screen is not the declared exact subset')
    return info['source_sha256']


def prediction_key(question, target):
    return digest([question, target])


def target_list(answer):
    return [answer] if isinstance(answer, str) else [str(value) for value in answer]


def decoder_question_text(question):
    return question[0] if isinstance(question, (list, tuple)) else str(question)


def inference_identity(adapter, args, record, model_id):
    hng = Path(__file__).resolve().parents[1]
    baseline = hng.parents[1] / '13_base_method/grip-exp'
    base = Path(args.model_cache_dir) / model_id.replace('/', '--')
    protocol = [hng / 'scripts/train_listed_contrastive.py',
                hng / 'src/hard_negative_grip/decode_io.py',
                hng / 'src/hard_negative_grip/metrics.py',
                hng / 'src/hard_negative_grip/official_lists.py',
                baseline / 'constants.py', baseline / 'models/utils.py',
                baseline / 'grip/tasks/eval_tasks/base.py',
                baseline / 'grip/tasks/eval_tasks/grip_eval.py']
    return {'schema': 'exact-prompt-greedy-v1', 'adapter': fingerprint(adapter),
            'runtime': runtime_versions(),
            'base': str(base.resolve()),
            'base_files': {p.name: (sha256(p) if p.suffix != '.safetensors' else signature(p))
                           for p in sorted(base.iterdir()) if p.is_file()},
            'protocol': {str(p.relative_to(hng.parents[1])): sha256(p) for p in protocol},
            'dtype': args.dtype, 'max_new_tokens': args.gen_max_length, 'do_sample': False,
            'cohort_sha256': validation_parent(args.eval_file, record),
            'record_context': digest({k: v for k, v in record.items() if k not in
                                      ('questions', 'answers', 'id', 'split')})}


def reusable_predictions(identity, samples):
    path = cache_root() / 'predictions' / f'{digest(identity)}.json'
    if not path.exists():
        return []
    data = read(path)
    if data.get('identity') != identity:
        raise ValueError('Prediction cache identity mismatch')
    rows = []
    for sample in samples:
        # Match train_listed_contrastive.evaluate_adapter: graph records store
        # [question_text, root_indices], but the actual prompt and saved row
        # contain question_text. Context/cohort/weights remain identity-bound.
        question_text = decoder_question_text(sample['question'])
        row = data['rows'].get(prediction_key(question_text, target_list(sample['answer'])))
        if row:
            rows.append({**row, 'question_id': sample['question_id'], 'split': sample['split']})
    return rows


def remember_predictions(identity, rows):
    path = cache_root() / 'predictions' / f'{digest(identity)}.json'
    with locked(path.with_suffix('.lock')):
        data = read(path) if path.exists() else {'identity': identity, 'rows': {}}
        if data['identity'] != identity:
            raise ValueError('Prediction cache identity mismatch')
        for row in rows:
            key = prediction_key(row['question'], row['target'])
            stable = {k: v for k, v in row.items() if k not in ('question_id', 'split')}
            if key in data['rows'] and data['rows'][key] != stable:
                raise ValueError('Identical prompt/weights produced conflicting cached predictions')
            data['rows'][key] = stable
        write(path, data)


def register_scores(identity, folder):
    """Register complete full-vocabulary train scores; retain files in place."""
    folder = Path(folder).resolve()
    stats = read(folder / 'run_stats.json')
    if (not stats.get('checks_passed') or identity.get('split') != 'train'
            or identity.get('filter_splits') != ['train']):
        raise ValueError('Only verified training scores may populate the cache')
    files = {name: signature(folder / name) for name in
             ('candidate_scores.jsonl', 'qa_summary.jsonl', 'score_identity.json', 'run_stats.json')}
    path = cache_root() / 'scores' / f'{digest(identity)}.json'
    with locked(path.with_suffix('.lock')):
        entries = read(path) if path.exists() else {}
        entries[str(folder)] = files
        write(path, entries)


def score_sources(identity, destination):
    path = cache_root() / 'scores' / f'{digest(identity)}.json'
    if not path.exists():
        return []
    sources = []
    for name, files in read(path).items():
        folder = Path(name)
        if (folder.resolve() == Path(destination).resolve() or
                (folder.exists() and Path(destination).exists() and folder.samefile(destination))):
            continue
        if all((folder / key).exists() and signature(folder / key) == value for key, value in files.items()):
            if read(folder / 'score_identity.json') == identity:
                sources.append(folder)
    return sources


def scoring_identity(args, relation_order, model_id='Qwen/Qwen2.5-7B-Instruct'):
    hng = Path(__file__).resolve().parents[1]
    base = Path(args.model_cache_dir) / model_id.replace('/', '--')
    files = [hng / 'src/hard_negative_grip/scoring.py',
             hng / 'src/hard_negative_grip/offline_scoring.py',
             hng / 'src/hard_negative_grip/task_file.py']
    return {'schema': 'train-full-vocab-v1', 'teacher': fingerprint(args.b1_adapter),
            'runtime': runtime_versions(), 'scoring_rule': scoring_rule_digest(),
            'task_sha256': sha256(args.task_file),
            'train_sha256': sha256(Path(args.raw_dir) / 'train.txt'),
            'mappings': {name: sha256(Path(value)) if value else None for name, value in
                         [('relations', args.relation_text_file), ('entities', args.entity_text_file)]},
            'relation_order': digest(relation_order), 'dtype': args.dtype,
            'temperature': args.temperature, 'candidate_batch_size': args.candidate_batch_size,
            'filter_splits': list(args.filter_splits), 'split': args.split,
            'model_name': args.model_name, 'model_cache': str(Path(args.model_cache_dir).resolve()),
            'base_files': {p.name: (sha256(p) if p.suffix != '.safetensors' else signature(p))
                           for p in sorted(base.iterdir()) if p.is_file()},
            'protocol': {p.name: sha256(p) for p in files}}


def score_blocks(path):
    """Stream one complete QA at a time instead of loading a 500MB bank."""
    current, block = None, []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            qid = row['qa_id']
            if current is not None and qid != current:
                yield current, block
                block = []
            current = qid
            block.append(row)
    if block:
        yield current, block


def complete_checkpoint(path):
    """An interrupted newer folder must not displace a usable resume state."""
    try:
        path = Path(path)
        step = int(path.name.split('-')[-1])
        if read(path / 'trainer_state.json').get('global_step') != step:
            return False
        required = [path / 'optimizer.pt', path / 'scheduler.pt']
        return (all(p.is_file() and p.stat().st_size > 0 for p in required)
                and any(p.stat().st_size > 0 for p in path.glob('rng_state*.pth'))
                and any(p.stat().st_size > 0 for p in path.rglob('adapter_model.safetensors')))
    except (OSError, ValueError, KeyError):
        return False


def prune_search_storage(out, trials):
    """Retain global top two snapshots plus one resume state per live trial.

    Metrics are never removed. Since evaluated metrics do not change, a
    snapshot outside the current top two cannot enter the final top two later.
    Call only between GPU subprocesses, after writing the new rung metric.
    """
    out = Path(out)
    metrics = [read(path) for path in (out / 'trials').glob('trial_*/step_*/metrics.json')]
    best = sorted(metrics, key=lambda row: (-row['screen_em'], row['trial'], row['step']))[:2]
    protected = {Path(row['adapter']).resolve() for row in best}
    selection = out / 'selection.json'
    if selection.exists():
        protected.add(Path(read(selection)['winner']['adapter']).resolve())
    states = {trial.number: trial.state.name for trial in trials}
    freed, deleted = 0, []
    for folder in (out / 'trials').glob('trial_*'):
        number = int(folder.name.split('_')[-1])
        # Unknown or interrupted state must never be treated as disposable.
        terminal = states.get(number) in ('COMPLETE', 'PRUNED')
        candidates = [path / 'adapter' for path in folder.glob('step_*')
                      if (path / 'metrics.json').exists() and (path / 'adapter').resolve() not in protected]
        checkpoints = list((folder / 'train_run/trainer_listed').glob('checkpoint-*'))
        complete = [path for path in checkpoints if complete_checkpoint(path)]
        latest = max(complete, key=lambda path: int(path.name.split('-')[-1]), default=None)
        candidates.extend(path for path in checkpoints if terminal or (path in complete and path != latest))
        resumable = (latest is not None and states.get(number) == 'RUNNING'
                     and (folder / f'step_{int(latest.name.split("-")[-1]):03d}/metrics.json').exists())
        if terminal or resumable:
            candidates.append(folder / 'train_run/listed/adapter')
        for directory in candidates:
            if directory.is_symlink():
                raise ValueError(f'Refusing to prune symlinked training artifacts: {directory}')
            if directory.resolve() in protected:
                continue
            if not directory.resolve().is_relative_to(out.resolve()):
                raise ValueError('Storage cleanup escaped search root')
            for path in directory.rglob('*'):
                if path.is_symlink():
                    continue
                if path.is_file() and (path.suffix in ('.safetensors', '.pt', '.pth') or
                                      path.name in ('pytorch_model.bin', 'optimizer.bin')):
                    size = path.stat().st_size
                    path.unlink()
                    freed += size
                    deleted.append({'path': str(path), 'bytes': size})
    if deleted:
        with (out / 'storage_cleanup.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': time.time(), 'freed_bytes': freed, 'deleted': deleted,
                                     'protected_adapters': sorted(map(str, protected))}) + '\n')
    managed_bytes = sum(path.stat().st_size for path in (out / 'trials').rglob('*')
                        if path.is_file() and not path.is_symlink() and
                        path.suffix in ('.safetensors', '.pt', '.pth'))
    write(out / 'storage_status.json', {'updated_at': time.time(),
          'free_bytes': shutil.disk_usage(out).free, 'managed_binary_bytes': managed_bytes,
          'minimum_training_free_gib': 30,
          'freed_this_pass': freed, 'protected_adapters': sorted(map(str, protected)),
          'policy': 'top2 validation snapshots; one latest optimizer checkpoint per unfinished trial; preserve evidence'})
    return freed


def require_training_space(out, minimum_gib=30):
    free = shutil.disk_usage(out).free
    if free < minimum_gib * 1024 ** 3:
        raise RuntimeError(f'Insufficient space for a safe checkpoint: {free / 1024**3:.1f}GiB free; '
                           f'{minimum_gib}GiB required; retained active/selected weights were not deleted')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register-scores'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    identity_file = args.directory / 'score_identity.json'
    if identity_file.exists():
        register_scores(read(identity_file), args.directory)
        print(f'Registered verified score archive: {args.directory}')
    else:
        print('Legacy score archive lacks an identity; leaving it in place without cross-run registration')
