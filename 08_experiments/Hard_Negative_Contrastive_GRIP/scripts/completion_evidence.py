"""CPU-only audit and disk-efficient retention of a locked selected result."""
import collections
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat

import hpo_reuse as reuse

REPO = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_selected(out, record, baseline_dir, archive, baseline_hash, test_hash):
    out, baseline_dir, archive = map(Path, (out, baseline_dir, archive))
    result = reuse.read(out / 'final_result.json')
    selection = reuse.read(out / 'selection.json')
    winner = selection['winner']
    adapter = Path(winner['adapter'])
    require(result['winner']['adapter_sha256'] == winner['adapter_sha256'], 'Result/selection weights disagree')
    require(selection.get('search_stopped_before_test') is True, 'Selection was not closed before test')
    used = selection.get('test_used_in_this_search_selection', selection.get('test_used_for_checkpoint_selection'))
    require(used is False, 'Test was used for checkpoint selection')
    require(result['seed'] == 2026, 'Unexpected seed')
    require(reuse.fingerprint(adapter) == winner['adapter_sha256'], 'Selected weights changed')
    require(reuse.fingerprint(Path(record['b1_adapter'])) == baseline_hash, 'Baseline weights changed')
    require(reuse.sha256(Path(record['test'])) == test_hash, 'Test file changed')
    inputs = [out / 'final_result.json', out / 'selection.json',
              out / 'selected_test/predictions.jsonl', out / 'selected_test/summary.json',
              out / 'selected_test/cache_identity.json', baseline_dir / 'predictions.jsonl',
              baseline_dir / 'summary.json', baseline_dir / 'cache_identity.json']
    optional = [adapter.parent.parent / 'train_run/run_config.json']
    if record.get('sampler'):
        optional += [Path(record['sampler']) / name for name in ('metadata.json', 'wiring.json')]
    if record.get('relation_eligibility'):
        optional.append(Path(record['relation_eligibility']))
    inputs += [p for p in optional if p.is_file()]
    signatures = {str(p): reuse.sha256(p) for p in inputs}
    stamp = {'inputs': signatures, 'selected_weights': winner['adapter_sha256'],
             'baseline_weights': baseline_hash, 'test_sha256': test_hash,
             'dataset': record['dataset'], 'paper_reference': record['paper_reference'],
             'auditor_sha256': reuse.sha256(Path(__file__))}
    marker = archive / 'ARCHIVE.json'
    if marker.exists():
        saved = reuse.read(marker)
        require(saved['audit_identity'] == stamp, 'Archived evidence identity changed')
        require(reuse.fingerprint(archive / 'adapter') == winner['adapter_sha256'], 'Retained weights changed')
        for name, digest in saved['retained_evidence_sha256'].items():
            require(reuse.sha256(archive / name) == digest, 'Retained evidence changed')
        return saved
    count = len(reuse.read(Path(record['test']))['questions'])
    spec = importlib.util.spec_from_file_location('completion_grip_normalizer',
        REPO / '13_base_method/grip-exp/evaluation/utils.py')
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    predictions = []
    for folder, hashes in ((baseline_dir, baseline_hash), (out / 'selected_test', winner['adapter_sha256'])):
        identity = reuse.read(folder / 'cache_identity.json')
        require(identity['adapter_sha256'] == hashes and identity['eval_sha256'] == test_hash,
                'Prediction weights/data identity mismatch')
        require(identity['gen_max_length'] == 32 and identity['dtype'] == 'bfloat16', 'Decoder changed')
        with (folder / 'predictions.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        require(len(rows) == len({r['question_id'] for r in rows}) == count, 'Incomplete or duplicate predictions')
        for row in rows:
            correct = any(normalizer.normalize_answer(row['response']) == normalizer.normalize_answer(t)
                          for t in row['target'])
            require(correct == row['correct'], 'Incorrect saved correctness flag')
        em = sum(row['correct'] for row in rows) / count
        require(reuse.read(folder / 'summary.json')['all'] == {'count': count, 'em': em}, 'Summary mismatch')
        predictions.append(rows)
    before, after = predictions
    require(all(a['question_id'] == b['question_id'] and a['question'] == b['question'] and a['target'] == b['target']
                for a, b in zip(before, after)), 'Paired prompts/targets/IDs changed')
    correct, baseline_correct = sum(r['correct'] for r in after), sum(r['correct'] for r in before)
    require(result['test_count'] == count and result['selected_test_em'] == correct / count
            and result['b1_test_em'] == baseline_correct / count, 'Result metrics mismatch')
    selected_cache = out / 'selected_test/cache_identity.json'
    require(reuse.read(selected_cache)['adapter_sha256'] == winner['adapter_sha256'], 'Selected test cache changed')
    retained = archive / 'adapter'
    retained.mkdir(parents=True, exist_ok=True)
    for source in adapter.iterdir():
        if not source.is_file():
            continue
        target = retained / source.name
        if target.exists():
            require(reuse.sha256(target) == reuse.sha256(source), 'Retained adapter collision')
        elif source.name == 'adapter_model.safetensors':
            os.link(source, target)
        else:
            shutil.copy2(source, target)
        target.chmod(stat.S_IMODE(target.stat().st_mode) & ~0o222)
    require((retained / 'adapter_model.safetensors').samefile(adapter / 'adapter_model.safetensors'),
            'Expected hard-linked weight retention')
    evidence = archive / 'evidence'
    evidence.mkdir(exist_ok=True)
    retained_evidence = {}
    for index, source in enumerate(inputs):
        target = evidence / f'{index:02d}_{source.name}'
        if not target.exists():
            shutil.copy2(source, target)
        require(reuse.sha256(target) == signatures[str(source)], 'Retained prediction evidence changed')
        retained_evidence[str(target.relative_to(archive))] = signatures[str(source)]
    reuse.write(evidence / 'dataset_contract.json', record)
    retained_evidence['evidence/dataset_contract.json'] = reuse.sha256(evidence / 'dataset_contract.json')
    if (out / 'contract.json').exists():
        source = out / 'contract.json'
        target = evidence / 'refinement_contract.json'
        if not target.exists():
            shutil.copy2(source, target)
        require(reuse.sha256(target) == reuse.sha256(source), 'Refinement contract archive changed')
        retained_evidence['evidence/refinement_contract.json'] = reuse.sha256(target)
    paper = record['paper_reference']
    report = {'verified_at': datetime.now().astimezone().isoformat(), 'dataset': record['dataset'],
        'seed': 2026, 'test_count': count, 'selected_correct': correct, 'baseline_correct': baseline_correct,
        'selected_test_em': correct / count, 'matched_b1_test_em': baseline_correct / count,
        'paper_reference_em': paper, 'delta_vs_paper_pp': (correct / count - paper) * 100,
        'target_achieved': correct / count > max(paper, baseline_correct / count),
        'paired_counts': dict(collections.Counter('both_correct' if a['correct'] and b['correct'] else
            'rescue' if b['correct'] else 'regression' if a['correct'] else 'both_wrong'
            for a, b in zip(before, after))),
        'matched_prompts_targets_ids_verified': True, 'exact_match_flags_recomputed': True,
        'retained_adapter': str(retained), 'additional_weight_blocks_allocated': 0,
        'scope': result.get('scope', 'Single-seed local protocol; paper comparability remains open'),
        'audit_identity': stamp, 'retained_evidence_sha256': retained_evidence}
    reuse.write(marker, report)
    return report
