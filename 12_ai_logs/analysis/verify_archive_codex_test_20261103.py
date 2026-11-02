"""Audit paired CoDEx test predictions and retain the frozen winner, CPU only."""
import collections
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys

REPO = Path(__file__).resolve().parents[2]
HNG = REPO / '08_experiments/Hard_Negative_Contrastive_GRIP'
sys.path.insert(0, str(HNG / 'scripts'))
import hpo_reuse as reuse


def main():
    root = HNG / 'results/hpo/stage2_three_dataset_20261030/codexm'
    out = root / 'search'
    result, selection = reuse.read(out / 'final_result.json'), reuse.read(out / 'selection.json')
    spec = importlib.util.spec_from_file_location('grip_normalization_for_audit',
        REPO / '13_base_method/grip-exp/evaluation/utils.py')
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    arms = {}
    for arm in ('b1_test', 'selected_test'):
        with (out / arm / 'predictions.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len({row['question_id'] for row in rows}) == 10311
        for row in rows:
            correct = any(normalizer.normalize_answer(row['response']) == normalizer.normalize_answer(target)
                          for target in row['target'])
            assert correct == row['correct']
        arms[arm] = rows
    before, after = arms['b1_test'], arms['selected_test']
    assert all(a['question_id'] == b['question_id'] and a['question'] == b['question'] and a['target'] == b['target']
               for a, b in zip(before, after))
    assert sum(row['correct'] for row in before) == 9727
    assert sum(row['correct'] for row in after) == 9958
    assert result['selected_test_em'] == 9958 / 10311
    assert result['b1_test_em'] == 9727 / 10311
    assert selection['search_stopped_before_test'] and not selection['test_used_in_this_search_selection']
    source = Path(result['winner']['adapter']).resolve()
    record = reuse.read(root / 'dataset_contract.json')
    assert reuse.fingerprint(source) == selection['frozen_adapter_sha256']
    assert reuse.fingerprint(Path(record['b1_adapter'])) == selection['b1_sha256']
    assert reuse.sha256(Path(record['test'])) == selection['test_file_sha256']
    assert reuse.read(out / 'selected_test/cache_identity.json')['adapter_sha256'] == selection['frozen_adapter_sha256']
    assert datetime.fromisoformat(selection['selected_at']).timestamp() < (out / 'selected_test/predictions.jsonl').stat().st_mtime
    archive = Path('/home/kieran/grip-cu-runs/paper_retained/codexm_seed2026_trial1_step32_20261103')
    adapter = archive / 'adapter'
    adapter.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file():
            continue
        target = adapter / path.name
        if target.exists():
            assert reuse.sha256(target) == reuse.sha256(path)
        elif path.name == 'adapter_model.safetensors':
            os.link(path, target)
        else:
            shutil.copy2(path, target)
        target.chmod(stat.S_IMODE(target.stat().st_mode) & ~0o222)
    assert (adapter / 'adapter_model.safetensors').samefile(source / 'adapter_model.safetensors')
    assert reuse.fingerprint(adapter) == selection['frozen_adapter_sha256']
    sources = [out / 'final_result.json', out / 'selection.json', root / 'dataset_contract.json',
        out / 'selected_test/predictions.jsonl', out / 'selected_test/summary.json',
        out / 'b1_test/predictions.jsonl', out / 'b1_test/summary.json',
        out / 'trials/trial_0001/train_run/run_config.json',
        root / 'assets/k9/metadata.json', root / 'assets/k9/wiring.json',
        REPO / '13_base_method/grip-exp/outputs/paper_runs/codexm_taskfile_3090/metadata.json']
    evidence = archive / 'evidence'
    evidence.mkdir(exist_ok=True)
    checksums = {}
    for index, path in enumerate(sources):
        target = evidence / f'{index:02d}_{path.name}'
        if not target.exists():
            shutil.copy2(path, target)
        assert reuse.sha256(target) == reuse.sha256(path)
        checksums[str(path)] = reuse.sha256(path)
    config = reuse.read(out / 'trials/trial_0001/train_run/run_config.json')
    baseline_config = reuse.read(sources[-1])['config']
    report = {'verified_at': datetime.now().astimezone().isoformat(), 'claim_type': 'parsed',
        'dataset': 'codexm', 'seed': 2026, 'test_count': 10311,
        'selected_correct': 9958, 'baseline_correct': 9727,
        'selected_test_em': result['selected_test_em'], 'matched_b1_test_em': result['b1_test_em'],
        'delta_vs_matched_baseline_pp': 231 / 10311 * 100,
        'paper_reference_em': result['paper_reference_em'], 'delta_vs_paper_pp': result['delta_vs_paper_pp'],
        'paper_reference_source': {'url': 'https://arxiv.org/html/2511.07457v2', 'table': 3,
            'row': 'GRIP-qwen-7b', 'reference_percent': 94.42,
            'provenance': 'Previously checked source recorded in campaign; no fresh source fetch in this audit'},
        'matched_prompts_targets_ids_verified': True, 'exact_match_flags_recomputed': True,
        'paired_counts': dict(collections.Counter('both_correct' if a['correct'] and b['correct'] else
            'rescue' if b['correct'] else 'regression' if a['correct'] else 'both_wrong'
            for a, b in zip(before, after))),
        'selected_at_before_test': selection['selected_at'], 'search_stopped_before_test': True,
        'retained_adapter': str(adapter), 'retained_weight_sha256': selection['frozen_adapter_sha256'],
        'retention_method': 'Read-only hard link outside HPO cleanup; metadata and evidence copied',
        'additional_weight_blocks_allocated': 0, 'source_adapter': str(source),
        'stage2_hyperparameters': result['winner']['params'], 'selected_optimizer_step': 32,
        'schedule_epochs': config['involve_qa_epochs'], 'method_adam_epsilon': config['adam_epsilon'],
        'control_adam_epsilon': baseline_config['adam_epsilon'],
        'control_schedule_epochs': baseline_config['stage2_epochs'],
        'control_microbatch': baseline_config['per_device_train_batch_size'],
        'method_microbatch': config['per_device_train_batch_size'],
        'scope': 'Single-seed measured local result; numerical exceedance of paper reference; paper protocol audit open',
        'limitations': ['No multi-seed or statistical claim',
            'Current Stage2 schedule horizon5 epochs differs from paper CoDEx10 epochs; selected checkpoint is32 steps',
            'Prompt/task generation and candidate protocol comparability to paper remains to be audited',
            'Stage2 tuning changes multiple hyperparameters; not an isolated contrastive-loss effect',
            'Matched Random9 and other ablation controls remain incomplete'],
        'evidence_sha256': checksums}
    reuse.write(archive / 'ARCHIVE.json', report)
    reuse.write(REPO / '11_paper/artifacts/codexm_single_seed_test_20261103.json', report)
    print(json.dumps({k: report[k] for k in ('selected_correct', 'baseline_correct', 'test_count',
        'selected_test_em', 'delta_vs_paper_pp', 'paired_counts', 'retained_adapter')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
