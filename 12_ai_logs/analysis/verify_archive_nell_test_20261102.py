"""Verify NELL full-test evidence and retain the winning adapter without duplicating weight blocks."""
import collections
from datetime import datetime
import hashlib
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
    root = HNG / 'results/hpo/stage2_three_dataset_20261030/nell23k'
    out = root / 'search'
    result = reuse.read(out / 'final_result.json')
    selection = reuse.read(out / 'selection.json')
    utils = REPO / '13_base_method/grip-exp/evaluation/utils.py'
    spec = importlib.util.spec_from_file_location('grip_normalization_for_audit', utils)
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    predictions = {}
    for arm in ('b1_test', 'selected_test'):
        rows = [json.loads(line) for line in (out / arm / 'predictions.jsonl').open()]
        assert len(rows) == 4943
        assert len({row['question_id'] for row in rows}) == 4943
        for row in rows:
            correct = any(normalizer.normalize_answer(row['response']) == normalizer.normalize_answer(target)
                          for target in row['target'])
            assert correct == row['correct']
        predictions[arm] = rows
    before, after = predictions['b1_test'], predictions['selected_test']
    assert all(a['question_id'] == b['question_id'] and a['question'] == b['question'] and a['target'] == b['target']
               for a, b in zip(before, after))
    assert sum(row['correct'] for row in before) == 4181
    assert sum(row['correct'] for row in after) == 4340
    assert result['selected_test_em'] == 4340 / 4943
    assert result['b1_test_em'] == 4181 / 4943
    assert selection['search_stopped_before_test'] and not selection['test_used_in_this_search_selection']
    source = Path(result['winner']['adapter']).resolve()
    assert reuse.fingerprint(source) == selection['frozen_adapter_sha256']
    assert reuse.read(out / 'selected_test/cache_identity.json')['adapter_sha256'] == selection['frozen_adapter_sha256']
    archive = Path('/home/kieran/grip-cu-runs/paper_retained/nell23k_seed2026_trial1_step32_20261102')
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
               root / 'assets/k9/metadata.json', root / 'assets/k9/wiring.json']
    evidence = archive / 'evidence'
    evidence.mkdir(exist_ok=True)
    checksums = {}
    for index, path in enumerate(sources):
        target = evidence / f'{index:02d}_{path.name}'
        if not target.exists():
            shutil.copy2(path, target)
        assert reuse.sha256(target) == reuse.sha256(path)
        checksums[str(path)] = reuse.sha256(path)
    baseline_config = reuse.read(Path('/mnt/e/GRIP-CU/artifacts/stage2_memory_retention/nell_r1/control_20261018_C1/experiment_config.json'))
    method_config = reuse.read(out / 'trials/trial_0001/train_run/run_config.json')
    report = {
        'verified_at': datetime.now().astimezone().isoformat(), 'claim_type': 'parsed',
        'dataset': 'nell23k', 'seed': 2026, 'test_count': 4943,
        'selected_correct': 4340, 'baseline_correct': 4181,
        'selected_test_em': result['selected_test_em'], 'matched_b1_test_em': result['b1_test_em'],
        'delta_vs_matched_baseline_pp': (4340 - 4181) / 4943 * 100,
        'paper_reference_em': .8774, 'delta_vs_paper_pp': result['delta_vs_paper_pp'],
        'paper_reference_source': {'url': 'https://arxiv.org/html/2511.07457v2', 'table': 3,
                                   'row': 'GRIP-qwen-7b', 'verified_published_nell_em_percent': 87.74},
        'matched_prompts_targets_ids_verified': True, 'exact_match_flags_recomputed': True,
        'paired_counts': dict(collections.Counter('both_correct' if a['correct'] and b['correct'] else
            'rescue' if b['correct'] else 'regression' if a['correct'] else 'both_wrong'
            for a, b in zip(before, after))),
        'selected_at_before_test': selection['selected_at'], 'search_stopped_before_test': True,
        'retained_adapter': str(adapter), 'retained_weight_sha256': selection['frozen_adapter_sha256'],
        'retention_method': 'Read-only hard link to frozen tensor file; independent archive directory outside HPO cleanup; metadata/evidence copied',
        'additional_weight_blocks_allocated': 0, 'source_adapter': str(source),
        'stage2_hyperparameters': result['winner']['params'], 'selected_optimizer_step': 32,
        'control_adam_epsilon': baseline_config['adam_epsilon'],
        'method_adam_epsilon': method_config['adam_epsilon'],
        'scope': 'Single-seed measured local result; numerical exceedance of cited paper value, not a statistical or fully audited replication claim',
        'limitations': ['Prior NELL tests were inspected during earlier exploration',
                        'Task generation used validation few-shot references; validation is developmental',
                        'Stage2 includes tuned settings and a different Adam epsilon; pure contrastive-loss effect is not isolated',
                        'Published prompt/task-generation comparability remains to be fully audited'],
        'evidence_sha256': checksums}
    reuse.write(archive / 'ARCHIVE.json', report)
    reuse.write(REPO / '11_paper/artifacts/nell23k_single_seed_test_20261102.json', report)
    print(json.dumps({key: report[key] for key in ['selected_correct', 'baseline_correct', 'test_count',
        'selected_test_em', 'paper_reference_em', 'delta_vs_paper_pp', 'retained_adapter',
        'additional_weight_blocks_allocated']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
