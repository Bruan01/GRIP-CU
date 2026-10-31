"""Reuse completed NELL scores and finish CPU bank/wiring beside CoDEx training."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
HNG = REPO / '08_experiments/Hard_Negative_Contrastive_GRIP'
sys.path[:0] = [str(HNG / 'scripts'), str(HNG / 'src')]
import hpo_reuse as reuse
import supervise_nell_fb_hpo as legacy
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.offline_mining import complete_qa_ids_from_scores


def main():
    root = HNG / 'results/hpo/stage2_three_dataset_20261030'
    record = reuse.read(root / 'nell23k/dataset_contract.json')
    work, score = root / 'nell23k', root / 'nell23k/assets/b1_train_scores'
    stats, metadata = reuse.read(score / 'run_stats.json'), reuse.read(score / 'metadata.json')
    gate = reuse.read(Path(record['relation_eligibility']))
    assert stats['checks_passed'] and stats['qa_count'] == record['expected_listed'] == 2974
    args = SimpleNamespace(b1_adapter=Path(record['b1_adapter']), task_file=Path(record['input']),
        raw_dir=Path(record['raw']), relation_text_file=None, entity_text_file=None,
        dtype=metadata['dtype'], temperature=metadata['temperature'],
        candidate_batch_size=metadata['candidate_batch_size'], filter_splits=['train'], split='train',
        model_name='qwen-7b', model_cache_dir=Path('/mnt/e/GRIP-CU/model_cache'))
    order = load_train_relation_order(args.raw_dir)
    assert reuse.scoring_identity(args, order) == reuse.read(score / 'score_identity.json')
    complete = complete_qa_ids_from_scores(score / 'candidate_scores.jsonl', vocab_size=len(order))
    assert complete == set(gate['eligible_question_ids'])
    bank = Path(record['sampler'])
    environment = legacy.env()
    environment.update(CUDA_VISIBLE_DEVICES='', DATASET='nell23k', NEGATIVE_SOURCE='structure_counterfactual',
        SKIP_TMUX='1', INPUT_FILE=record['input'], EVAL_FILE=record['valid'], RAW_DIR=record['raw'],
        S1_ADAPTER=record['s1_adapter'], B1_ADAPTER=record['b1_adapter'],
        MODEL_CACHE_DIR=str(args.model_cache_dir), ARTIFACT_ROOT=str(work / 'assets'),
        SCORES_DIR=str(score), SAMPLER_DIR=str(bank), MANIFEST=str(bank / 'counterfactual_bank.jsonl'),
        COUNTERFACTUAL_METADATA=str(bank / 'metadata.json'), EXPECTED_LISTED='2974',
        RELATION_ELIGIBILITY=record['relation_eligibility'], LISTED_NEGATIVE_K='9', SEED='2026')
    for stage, artifact in [('freeze', bank / 'metadata.json'), ('wire', bank / 'wiring.json')]:
        if artifact.exists():
            continue
        environment.update(STEP=stage, RUN_ID=f'nell23k_recovery_{stage}_20261031', RUN_DIR=str(work / stage))
        legacy.hpo.run_logged(['bash', str(legacy.hpo.RUNNER)], environment.copy(),
                              root / 'logs' / f'nell23k_recovery_{stage}.log', HNG)
    assert reuse.read(bank / 'metadata.json')['complete_for_requested_k']
    assert reuse.read(bank / 'wiring.json')['train_only'] is True
    failure = work / 'failure.json'
    old = reuse.read(failure)
    reuse.write(failure, {**old, 'resolved': True, 'resolved_at': legacy.now(),
                         'resolution': 'Completed GPU scores reused; CPU freeze/wire passed; score inference not repeated'})
    reuse.write(work / 'recovery_20261031.json', {
        'completed_at': legacy.now(), 'scores_reused': 2974, 'candidate_rows_reused': stats['total_candidates'],
        'new_gpu_scoring_qa': 0, 'cpu_only': True, 'bank': str(bank), 'prior_failure': old,
        'root_cause': 'Mutable Bash runner edited during the original long-running child; GPU scoring itself completed',
        'prevention': 'Launch content-addressed immutable Bash snapshots; preserve original HNG through environment'})
    print('NELL CPU prerequisites recovered from all2974 existing scores; no GPU scoring repeated', flush=True)


if __name__ == '__main__':
    main()
