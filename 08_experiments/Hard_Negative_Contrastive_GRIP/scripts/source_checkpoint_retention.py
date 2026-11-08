"""Explicit retention amendments; never waive loss of a final model/state."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

AMENDMENT = Path('recovery/checkpoint_retention_20261108/amendment.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retention_limit(design):
    require(design['save_steps'] > 0 and design['optimizer_steps'] > 0, 'invalid checkpoint interval/budget')
    return (design['optimizer_steps']+design['save_steps']-1)//design['save_steps']+1


def load_amendment(run):
    path = run/AMENDMENT
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    require(sha(path) == path.with_suffix('.sha256').read_text().strip(), 'retention amendment changed')
    require(record['schema'] == 'source-retention-amendment-v1'
            and sha(run/'contract.json') == record['original_contract_sha256'], 'amendment belongs to another contract')
    for key in ('controller', 'helper'):
        require(sha(Path(record[key+'_path'])) == record[key+'_sha256'], 'amended controller/helper changed')
    return record


def execution_contract(run, original):
    amendment = load_amendment(run)
    if amendment is None:
        return original
    c = deepcopy(original)
    for name, argv in c['training_argv'].items():
        encoded = json.dumps(original['training_argv'][name], ensure_ascii=False).encode()
        require(hashlib.sha256(encoded).hexdigest() == amendment['original_argv_sha256'][name], 'original training command changed')
        limit = amendment['retention_limits'][name]
        require(limit == retention_limit(c['design']) or name in amendment['legacy_retention_loss'] and limit == 0,
                'unregistered retention limit')
        argv[argv.index('--save_total_limit')+1] = str(limit)
    path = run/AMENDMENT.parent/'execution_contract.json'
    c['runtime_contract_path'] = str(path)
    c['retention_amendment_sha256'] = sha(run/AMENDMENT)
    path.write_text(json.dumps(c, ensure_ascii=False, indent=2)+'\n')
    return c


def cli_retention(run, c, name, actual):
    from hard_negative_grip.run_protection import checkpoint_training_kwargs
    effective = checkpoint_training_kwargs(actual['save_steps'], actual['save_total_limit'])
    needed = retention_limit(c['design'])-1
    if effective.get('save_strategy') == 'steps' and effective['save_steps'] == c['design']['save_steps'] and effective['save_total_limit'] >= needed:
        return
    a = load_amendment(run)
    require(a is not None and name in a['legacy_retention_loss']
            and (run/name/'training/listed/adapter/run_metadata.json').exists(),
            'effective checkpoint retention cannot keep all planned states')
    audit_retention(run, c, name)


_pinned = {}


def pinned_file(path, expected):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, expected)
    if key not in _pinned:
        require(sha(path) == expected, 'acknowledged final model/state changed')
        _pinned[key] = True


def audit_retention(run, c, name):
    d = c['design']
    required = ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json')
    planned = list(range(d['save_steps'], d['optimizer_steps']+1, d['save_steps']))
    missing = [step for step in planned if not all((run/name/f'training/trainer_listed/checkpoint-{step}'/f).is_file() for f in required)]
    if not missing:
        return dict(status='complete', retained_steps=planned, missing_steps=[])
    a = load_amendment(run)
    require(a is not None and name in a['legacy_retention_loss'], 'missing retained checkpoint state')
    loss = a['legacy_retention_loss'][name]
    require(d['dataset'] == 'nell23k' and name == 'random9' and d['optimizer_steps'] == 32
            and d['scheduler_max_steps'] == 240 and missing == loss['missing_steps'] == [8,16,24]
            and loss['irrecoverable'] is True, 'unacknowledged checkpoint loss; cannot continue')
    # The final step is never covered by the exception.
    final = run/name/f"training/trainer_listed/checkpoint-{d['optimizer_steps']}"
    require(all((final/f).is_file() for f in required), 'final resume state missing')
    for filename, digest in loss['final_adapter_sha256'].items():
        pinned_file(run/name/'training/listed/adapter'/filename, digest)
    pinned_file(final/'trainer_state.json', loss['checkpoint32_state_sha256'])
    return dict(status='completed_with_irrecoverable_intermediate_checkpoint_loss',
                retained_steps=[32], missing_steps=missing, no_retraining=True,
                loss_acknowledged_before_new_test_result=True)
