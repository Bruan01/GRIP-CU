#!/usr/bin/env python3
"""Frozen, single-seed K9 source controls for NELL and WN; prepare is CPU only.

Reuse each dataset's selected Full model. Train exactly Uniform and Full-quota
matched controls at its own original optimizer/scheduler budget, then retain
all direct-generation and original-list ranking outcomes, including negatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HNG = Path(__file__).resolve().parents[1]
ROOT = HNG.parents[1]
sys.path[:0] = [str(HNG/'scripts'), str(HNG/'src'), str(ROOT/'13_base_method/grip-exp')]
import run_fb_random9_control as base
from hard_negative_grip.counterfactuals import load_train_relation_profiles
from hard_negative_grip.frozen_random import score_groups, uniform_bank_row, validate_frozen_random_manifest
from hard_negative_grip.frequency_matched import matched_bank_row, validate_frequency_manifest
from hard_negative_grip.matched_control import (
    audit_predictions, paired_result, parameter_diff, read_json, sha256,
    training_command, write_csv, write_json,
)
from hard_negative_grip.official_lists import load_train_relation_order
from hard_negative_grip.rerank_audit import SCORE_RULE, compare, read_rows, summary, validate_choices
from hard_negative_grip.task_file import (
    build_qa_assets_from_task_texts, known_pair_relations, load_counterfactual_manifest,
    load_graph_record, load_relation_eligibility, original_question_ids_from_payload,
    validate_structure_counterfactual_manifest,
)

ARMS = ('random9', 'frequency_matched')
METHODS = ('full', *ARMS)
SOURCES = dict(random9='frozen_random', frequency_matched='frequency_matched')
LABELS = dict(full='Full SC-CL-9', random9='Uniform Random-9',
              frequency_matched='Full-quota-matched Random-9')
require = base.require


def cache_identity(cache):
    directory = Path(cache)/'Qwen--Qwen2.5-7B-Instruct'
    require(directory.is_dir(), 'offline model cache missing')
    small = {p.name:sha256(p) for p in sorted(directory.iterdir()) if p.is_file()
             and p.suffix in ('.json','.txt','.model') and p.stat().st_size < 100*1024**2}
    weights = {p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
               for p in sorted(directory.glob('*.safetensors'))}
    require(small and weights, 'offline model cache incomplete')
    return dict(path=str(directory.resolve()), config_tokenizer_sha256=small,
                weight_file_stats=weights, weight_identity_scope='file size/mtime; full base shard hashes not newly computed')


def relation_parser(code_root):
    # Loading the standalone module preserves the baseline parser exactly and
    # avoids importing grip.__init__, which imports the GPU training stack.
    name = 'frozen_relation_parser'
    spec = importlib.util.spec_from_file_location(name, code_root/base.REL_BASE/'grip/relation_cecr.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.parse_relation_candidates


def checked_design(d):
    expected = dict(nell23k=(32, 240, 12014, 2974, 4943, 10, 32),
                    wn18rr=(125, 125, 12334, 2709, 3134, 11, 1000))
    require(d['dataset'] in expected and d['seed'] == 2026 and d['k'] == 9,
            'unregistered dataset/seed/K')
    fields = ('optimizer_steps', 'scheduler_max_steps', 'qa_count', 'listed_qa_count',
              'test_count', 'candidate_count', 'generation_max_length')
    require(tuple(d[f] for f in fields) == expected[d['dataset']], 'unregistered experiment budget')
    require(d['max_new_training_runs'] == 2 and d['arms'] == list(ARMS), 'unbounded arm list')
    return d


def group_known_relations(group, known, dataset):
    pairs = {(r['head_entity'], r['tail_entity']) for r in group}
    require(len(pairs) == 1, 'inconsistent entity pair in score group')
    pair = next(iter(pairs))
    gold = [r for r in group if r['is_gold'] is True]
    require(len(gold) == 1, 'missing/duplicate parent gold')
    relation = gold[0]['matched_train_relation']
    if pair in known:
        require(relation in known[pair], 'QA gold unsupported by its train pair')
        return known[pair], 'verified_train_pair'
    # The historical WN Full includes exactly 55 QA rows without parsed pairs.
    # Preserve the original mask for an explicitly limited historical-pool
    # comparison; never invent a pair, change QA eligibility, or call it verified.
    require(dataset == 'wn18rr' and pair == (None, None), 'unverified train entity pair')
    excluded = {r['candidate_relation'] for r in group if r['is_valid_negative'] is False}
    require(excluded == {relation}, 'unexpected unverified historical WN mask')
    return excluded, 'unverified_pair_original_parent_mask'


def mark_pair_scope(row, pair_status):
    verified = pair_status == 'verified_train_pair'
    row['entity_pair_verification'] = pair_status
    row['train_pair_verified'] = verified
    # The legacy True flag means absence from the provided exclusion mask.
    # Explicitly distinguish that operational check from independent proof.
    for detail in row['selected_negatives']:
        detail['train_known_absent_independently_verified'] = verified
        detail['train_known_absent_proof_scope'] = ('verified_train_entity_pair' if verified
            else 'original_parent_exclusion_mask_only_entity_pair_unverified')
    return row


def freeze_link(run, source, relative, artifacts, expected=None):
    source = base.resolve_source(source)
    digest = sha256(source)
    require(expected is None or digest == expected, 'source identity changed: '+relative)
    target = run/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), 'refusing to overwrite frozen reference')
    target.symlink_to(source)
    artifacts[relative] = dict(sha256=digest, bytes=source.stat().st_size,
                               source=str(source), borrowed=True)


def make_training_argv(reference, arm, design):
    cmd = training_command(str(base.PYTHON), arm/'code_snapshot'/base.REL_HNG/'scripts/train_listed_contrastive.py',
                           reference, inputs=arm/'inputs', bank=arm.parent/'bank'/(arm.name+'.jsonl'),
                           output=arm/'training', model_cache=Path(design['model_cache']))
    cmd[cmd.index('--listed_negative_source')+1] = SOURCES[arm.name]
    for flag, name in (('--relation_text_file', 'raw/rel2text.json'),
                       ('--relation_eligibility', 'relation_eligibility.json')):
        if not (arm/'inputs'/name).exists():
            i = cmd.index(flag)
            del cmd[i:i+2]
    cmd[cmd.index('--save_steps')+1] = str(design['save_steps'])
    cmd[cmd.index('--save_total_limit')+1] = '0'  # retain every planned checkpoint
    return cmd


def verify_cli(reference, actual, arm, arm_dir):
    rows = parameter_diff(reference, actual)
    require(actual['listed_negative_source'] == SOURCES[arm]
            and actual['negative_sampling_mode'] == 'random', 'wrong control loader')
    require(Path(actual['random_manifest']).resolve() ==
            (arm_dir.parent/'bank'/(arm+'.jsonl')).resolve(), 'wrong frozen control bank')
    require(actual['listed_negative_seed'] == 2026 and actual['memory_size'] == 0,
            'candidate seed or memory changed')
    return rows


def match_qa_metadata(before, after):
    # Candidate strings and their declared source are the two interventions.
    # Eligibility, prompt/prefix, positive label and per-QA loss weight stay fixed.
    require(len(before) == len(after), 'QA count changed')
    for left, right in zip(before, after):
        require({k:v for k,v in left.items() if k not in ('listed_relations','listed_negative_source')} ==
                {k:v for k,v in right.items() if k not in ('listed_relations','listed_negative_source')},
                'noncandidate metadata/loss changed')


def prepare(run, config):
    if (run/'contract.json').exists():
        c = base.verify(run)
        require(c['design_sha256'] == sha256(config), 'design changed after freeze')
        return c
    require(not (run/'inputs').exists(), 'partial preparation retained; use a fresh directory')
    d = checked_design(read_json(config))
    base.append_event(run, dict(stage='prepare', status='running'))
    artifacts = {}
    base.freeze(run, config, 'inputs/design.json', artifacts)
    ref_path = base.resolve_source(d['reference_run_config'])
    require(sha256(ref_path) == d['reference_config_sha256'], 'Full config changed')
    ref = read_json(ref_path)
    require(ref['dataset'] == d['dataset'] and ref['seed'] == 2026
            and ref['listed_negative_k'] == 9 and ref['stop_after_step'] == d['optimizer_steps'],
            'Full training contract differs')
    base.freeze(run, ref_path, 'reference/run_config.json', artifacts)
    for source_key, target in (('reference_trainer_state', 'reference/trainer_state.json'),
                               ('reference_selection', 'reference/selection.json'),
                               ('reference_predictions', 'reference/direct_full.jsonl')):
        base.freeze(run, d[source_key], target, artifacts, d[source_key+'_sha256'])
    state = read_json(run/'reference/trainer_state.json')
    require(state['global_step'] == d['optimizer_steps']
            and state['max_steps'] == d['scheduler_max_steps'], 'Full optimizer/scheduler budget differs')
    for key, target in (('input_file', 'inputs/task.json'), ('eval_file', 'inputs/validation.json')):
        base.freeze(run, ref[key], target, artifacts)
    base.freeze(run, d['test_file'], 'inputs/test.json', artifacts, d['test_sha256'])
    raw = base.resolve_source(ref['raw_dir'])
    base.freeze(run, raw/'train.txt', 'inputs/raw/train.txt', artifacts)
    if ref.get('relation_text_file'):
        base.freeze(run, ref['relation_text_file'], 'inputs/raw/rel2text.json', artifacts)
    if ref.get('relation_eligibility'):
        base.freeze(run, ref['relation_eligibility'], 'inputs/relation_eligibility.json', artifacts)
    entity = ref.get('paths', {}).get('entity_text_file')
    if entity:
        base.freeze(run, entity, 'inputs/entity2text.txt', artifacts)
    for name, directory in (('stage1_adapter', ref['s1_adapter']), ('full_adapter', d['full_adapter'])):
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            expected = d['full_adapter_sha256'][filename] if name == 'full_adapter' else None
            freeze_link(run, base.resolve_source(directory)/filename, 'inputs/'+name+'/'+filename, artifacts, expected)
    for source in (HNG/'src').rglob('*.py'):
        base.freeze(run, source, Path('code_snapshot')/source.relative_to(ROOT), artifacts)
    for source in (HNG/'scripts').rglob('*.py'):
        base.freeze(run, source, Path('code_snapshot')/source.relative_to(ROOT), artifacts)
    baseline = ROOT/base.REL_BASE
    sources = set(baseline.glob('*.py'))
    for name in ('arguments', 'data', 'evaluation', 'grip', 'models'):
        sources.update((baseline/name).rglob('*.py'))
    for source in sorted(sources):
        base.freeze(run, source, Path('code_snapshot')/source.relative_to(ROOT), artifacts)
    base.freeze(run, ref['counterfactual_manifest'], 'reference/full_bank.jsonl', artifacts,
                d['full_bank_sha256'])
    base.freeze(run, ref['counterfactual_metadata'], 'reference/full_bank_metadata.json', artifacts)
    metadata = read_json(run/'reference/full_bank_metadata.json')
    require(metadata['filter_splits'] == ['train'] and metadata['rows_failed'] == 0,
            'Full bank filtering/coverage mismatch')
    freeze_link(run, metadata['score_path'], 'reference/b1_scores.jsonl', artifacts,
                d['source_scores_sha256'])
    relation_file = run/'inputs/raw/rel2text.json'
    relation_file = relation_file if relation_file.exists() else None
    entity_file = run/'inputs/entity2text.txt'
    vocabulary = load_train_relation_order(run/'inputs/raw', relation_text_file=relation_file)
    known = known_pair_relations(run/'inputs/raw', splits=('train',), relation_text_file=relation_file,
                                 entity_text_file=entity_file if entity_file.exists() else None)
    profiles = load_train_relation_profiles(run/'inputs/raw', relation_text_file=relation_file,
                                            strip_first_underscore=d['dataset']=='wn18rr')
    frequencies = {name: profiles[name].edge_count for name in vocabulary}
    full = load_counterfactual_manifest(run/'reference/full_bank.jsonl')
    validate_structure_counterfactual_manifest(full, k=9)
    eligible = load_relation_eligibility(run/'inputs/relation_eligibility.json', run/'inputs/task.json', run/'inputs/raw') if (run/'inputs/relation_eligibility.json').exists() else None
    require(len(full) == d['listed_qa_count'] and (eligible is None or set(full) == eligible), 'Full eligibility differs')
    banks = {name: {} for name in ARMS}
    diagnostics = []
    (run/'bank').mkdir()
    streams = {name: (run/'bank'/(name+'.jsonl')).open('w') for name in ARMS}
    try:
        with (run/'bank/pool_audit.jsonl').open('w') as pool_out, (run/'bank/frequency_audit.jsonl').open('w') as freq_out:
            for group in score_groups(run/'reference/b1_scores.jsonl'):
                qid = group[0]['qa_id']
                require(qid in full and qid not in banks['random9'], 'unexpected score group')
                known_relations, pair_status = group_known_relations(group, known, d['dataset'])
                args = dict(relation_order=vocabulary, known_relations=known_relations, seed=2026, k=9)
                random, pool = uniform_bank_row(group, **args)
                freq, audit = matched_bank_row(group, frequencies=frequencies, parent_row=full[qid], **args)
                mark_pair_scope(random, pair_status)
                mark_pair_scope(freq, pair_status)
                require(set(full[qid]['negative_relations']).issubset(pool['legal_relations']), 'Full pool changed')
                pool['entity_pair_verification'] = audit['entity_pair_verification'] = pair_status
                for name, row in (('random9', random), ('frequency_matched', freq)):
                    banks[name][qid] = row
                    streams[name].write(json.dumps(row, ensure_ascii=False)+'\n')
                pool_out.write(json.dumps(pool, ensure_ascii=False)+'\n')
                freq_out.write(json.dumps(audit, ensure_ascii=False)+'\n')
                diagnostics.append(dict(question_id=qid, pool_size=pool['pool_size'], entity_pair_verification=pair_status,
                    full_random_overlap=len(set(random['negative_relations']) & set(full[qid]['negative_relations'])),
                    full_frequency_overlap=len(set(freq['negative_relations']) & set(full[qid]['negative_relations'])),
                    forced_frequency_slots=audit['forced_slot_count'], exact_bin_match=audit['exact_bin_match'], fallback_count=audit['fallback_count']))
    finally:
        for stream in streams.values():
            stream.close()
    require(all(set(b) == set(full) for b in banks.values()), 'incomplete control bank')
    validate_frozen_random_manifest(banks['random9'], k=9)
    validate_frequency_manifest(banks['frequency_matched'], k=9)
    payload = read_json(run/'inputs/task.json')
    texts = payload['qa_samples']
    ids = original_question_ids_from_payload(payload, len(texts))
    common = dict(seed=2026, listed_negative_k=9, relation_order=vocabulary,
                  question_ids=ids, eligible_question_ids=eligible)
    _, before = build_qa_assets_from_task_texts(texts, listed_negative_source='structure_counterfactual',
                                             counterfactual_manifest=full, **common)
    for name in ARMS:
        _, after = build_qa_assets_from_task_texts(texts, listed_negative_source=SOURCES[name],
                                                random_manifest=banks[name], **common)
        require(len(texts) == d['qa_count'] and sum(bool(r['listed_relations']) for r in after) == d['listed_qa_count'], 'QA coverage changed')
        match_qa_metadata(before, after)
    write_csv(run/'bank/diagnostics.csv', diagnostics)
    unverified = [r['question_id'] for r in diagnostics if r['entity_pair_verification'] != 'verified_train_pair']
    require(len(unverified) == (55 if d['dataset']=='wn18rr' else 0), 'unexpected unverified parent-pair count')
    write_json(run/'bank/unverified_parent_pairs.json', dict(question_ids=unverified,
        policy='retain historical Full QA and original mask; no inferred/fuzzy pair or eligibility repair; exploratory comparison only' if unverified else 'all scored QA pairs verified against train'))
    prep = dict(status='pass', training_inputs_train_only=True, all_train_pairs_verified=not unverified,
                unverified_parent_pair_rows=len(unverified), comparison_scope='historical_parent_pool_conditional' if unverified else 'verified_train_pair_source_control',
                qa_count=len(texts), listed_qa_count=len(full),
                exact_eligibility_and_noncandidate_metadata=True,
                pool_size_counts=dict(Counter(r['pool_size'] for r in diagnostics)),
                mean_full_random_overlap=sum(r['full_random_overlap'] for r in diagnostics)/len(diagnostics),
                mean_full_frequency_overlap=sum(r['full_frequency_overlap'] for r in diagnostics)/len(diagnostics),
                forced_frequency_slots=sum(r['forced_frequency_slots'] for r in diagnostics),
                exact_bin_match_rows=sum(r['exact_bin_match'] for r in diagnostics),
                fallback_count=sum(r['fallback_count'] for r in diagnostics), model_loaded=False)
    write_json(run/'bank/preparation_audit.json', prep)
    for path in (run/'bank').iterdir():
        artifacts[str(path.relative_to(run))] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    samples = base.eval_samples(load_graph_record(run/'inputs/test.json'))
    normalize = base.normalization(run/'code_snapshot')
    direct = audit_predictions(base.jsonl(run/'reference/direct_full.jsonl'), samples, normalize)
    require(len(direct) == d['test_count'] and sum(r['correct'] for r in direct.values()) == d['reference_correct'], 'Full direct cohort/result changed')
    counts = sorted(frequencies.values())
    lo, hi = counts[(len(counts)-1)//3], counts[2*(len(counts)-1)//3]
    cohort = []
    parse_relation_candidates = relation_parser(run/'code_snapshot')
    for i, sample in enumerate(samples):
        row = direct[sample['question_id']]
        candidates = parse_relation_candidates(row['question'])
        gold, collisions = validate_choices(candidates, row['target'], normalize, expected_candidates=d['candidate_count'])
        count = frequencies.get(candidates[gold])
        require(count is not None, 'test gold outside train vocabulary')
        cohort.append(dict(dataset_index=i, question_id=sample['question_id'], question=row['question'],
                           target=row['target'], candidates=candidates, gold_index=gold,
                           frequency_group='low' if count <= lo else 'middle' if count <= hi else 'high',
                           train_edge_count=count, direct_correct={'full': row['correct']}))
    (run/'inputs/cohort.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in cohort))
    artifacts['inputs/cohort.jsonl'] = dict(sha256=sha256(run/'inputs/cohort.jsonl'), bytes=(run/'inputs/cohort.jsonl').stat().st_size)
    code = {name:r['sha256'] for name,r in artifacts.items() if name.startswith('code_snapshot/')}
    c = dict(schema='cross-dataset-source-controls-v1', created_at=base.now(), experiment_id=d['experiment_id'],
             repository_root=str(ROOT), design=d, design_sha256=sha256(config), immutable_artifacts=artifacts,
             code_digest=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
             cohort_sha256=sha256(run/'inputs/cohort.jsonl'), model_cache=d['model_cache'],
             base_model_identity=cache_identity(d['model_cache']), frequency_tercile_cutoffs=[lo,hi],
             selection='fixed original Full-selected configuration and optimizer step; no new validation/HPO')
    for name in ARMS:
        arm = run/name
        arm.mkdir()
        for folder in ('inputs', 'reference', 'code_snapshot'):
            (arm/folder).symlink_to(run/folder, target_is_directory=True)
        write_json(arm/'contract.json', dict(repository_root=str(ROOT), design=d))
        c.setdefault('training_argv', {})[name] = make_training_argv(ref, arm, d)
    write_json(run/'contract.json', c)
    write_json(run/'environment_prepare.json', base.environment(run))
    write_json(run/'status.json', dict(stage='prepared', status='ready', at=base.now(), new_training_started=False))
    base.append_event(run, dict(stage='prepare', status='completed', audit=prep))
    return c


def cli_check(run, c):
    for name in ARMS:
        arm = run/name
        path = arm/'checks/cli.json'
        if not path.exists():
            (arm/'training').mkdir(exist_ok=True)
            program = (
                "import sys;from pathlib import Path;import train_listed_contrastive as t;"
                "from hard_negative_grip.matched_control import read_json,write_json;"
                "from run_cross_dataset_controls import verify_cli;"
                "c=read_json(Path(sys.argv[1]));name=sys.argv[2];argv=c['training_argv'][name];"
                "sys.argv=argv[1:];a=t.parse_args();assets=t.resolve_training_assets(a);"
                "t.write_run_config(Path(a.output_dir),a);arm=Path(a.output_dir).parent;"
                "verify_cli(read_json(arm/'reference/run_config.json'),read_json(arm/'training/run_config.json'),name,arm);"
                "write_json(arm/'checks/cli.json',{'n_qa':len(assets[2]),'listed_qa':sum(bool(r['listed_relations']) for r in assets[3]),'model_loaded':False})"
            )
            base.execute(arm, 'cli_preflight', [str(base.PYTHON), '-c', program, str(run/'contract.json'), name])
        check = read_json(path)
        require(check == dict(n_qa=c['design']['qa_count'], listed_qa=c['design']['listed_qa_count'], model_loaded=False), 'CPU preflight coverage mismatch')
        verify_cli(read_json(run/'reference/run_config.json'), read_json(arm/'training/run_config.json'), name, arm)
    write_json(run/'checks/cli.json', dict(status='pass', arms=list(ARMS), model_loaded=False, at=base.now()))
    write_json(run/'status.json', dict(stage='preflighted', status='ready', at=base.now(), new_training_started=False))


def prerequisite(c):
    d = c['design']
    parent = Path(d['prerequisite_queue'])
    require(sha256(parent/'plan.json') == d['prerequisite_plan_sha256']
            and read_json(parent/'status.json')['status'] == 'completed', 'FB K prerequisite incomplete/changed')
    if d.get('previous_dataset_run'):
        previous = Path(d['previous_dataset_run'])
        require(sha256(previous/'contract.json') == d['previous_dataset_contract_sha256']
                and read_json(previous/'status.json')['status'] == 'completed', 'previous dataset incomplete/changed')


def validate_training(run, c, name):
    arm = run/name
    d = c['design']
    verify_cli(read_json(run/'reference/run_config.json'), read_json(arm/'training/run_config.json'), name, arm)
    state = read_json(arm/f"training/trainer_listed/checkpoint-{d['optimizer_steps']}/trainer_state.json")
    meta = read_json(arm/'training/listed/adapter/run_metadata.json')
    require(state['global_step'] == meta['trainer_global_step'] == d['optimizer_steps']
            and state['max_steps'] == meta['trainer_max_steps'] == d['scheduler_max_steps'], 'optimizer/scheduler mismatch')
    for step in range(d['save_steps'], d['optimizer_steps']+1, d['save_steps']):
        checkpoint = arm/f'training/trainer_listed/checkpoint-{step}'
        for filename in ('adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'):
            require((checkpoint/filename).is_file(), 'missing retained checkpoint state: '+str(checkpoint/filename))
    return meta


def train(run, c, name):
    prerequisite(c)
    arm = run/name
    require((run/'checks/cli.json').exists(), 'CPU preflight missing')
    if (arm/'training/listed/adapter/run_metadata.json').exists():
        validate_training(run, c, name)
        return
    for checkpoint in (arm/'training/trainer_listed').glob('checkpoint-*'):
        for filename in ('adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'):
            require((checkpoint/filename).is_file(), 'incomplete checkpoint; preserve and inspect before recovery')
    base.execute(arm, 'train', c['training_argv'][name])
    validate_training(run, c, name)


def ranking_view(run, c, name):
    view = run/'ranking'/name
    adapter = run/'inputs/full_adapter' if name == 'full' else run/name/'training/listed/adapter'
    weights = {n:sha256(adapter/n) for n in ('adapter_config.json','adapter_model.safetensors')}
    if (view/'contract.json').exists():
        vc = base.verify(view)
        require(vc['models'][name]['adapter_sha256'] == weights and vc['code_digest'] == c['code_digest']
                and vc['cohort_sha256'] == c['cohort_sha256'], 'ranking identity changed')
        return view
    (view/'inputs').mkdir(parents=True)
    (view/'code_snapshot').symlink_to(run/'code_snapshot', target_is_directory=True)
    artifacts = {n:r for n,r in c['immutable_artifacts'].items() if n.startswith('code_snapshot/')}
    for filename in ('test.json','cohort.jsonl'):
        target = view/'inputs'/filename
        target.symlink_to(run/'inputs'/filename)
        artifacts['inputs/'+filename] = dict(sha256=sha256(target), bytes=target.stat().st_size)
    for filename in weights:
        freeze_link(view, adapter/filename, 'inputs/adapter/'+filename, artifacts, weights[filename])
    vc = dict(repository_root=c['repository_root'], immutable_artifacts=artifacts, models={name:dict(
                  adapter_dir=str(view/'inputs/adapter'), adapter_sha256=weights)},
              code_digest=c['code_digest'], cohort_sha256=c['cohort_sha256'], model_cache=c['model_cache'],
              base_model_identity=c['base_model_identity'], candidate_count=c['design']['candidate_count'],
              expected_questions=c['design']['test_count'], score_batch_size=2, max_length=4096)
    write_json(view/'contract.json', vc)
    return view


def evaluate_ranking(run, c, name):
    view = ranking_view(run, c, name)
    base.execute(view, 'rerank', [str(base.PYTHON), '-u', str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_matched_candidate_rerank.py'),
                                 '--run-dir', str(view), '--model', name])


def evaluate_direct(run, c, name):
    arm = run/name
    out = arm/'evaluation/test'
    base.execute(arm, 'eval_test', [str(base.PYTHON), str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_adapter.py'),
                                  '--adapter', str(arm/'training/listed/adapter'), '--eval_file', str(run/'inputs/test.json'),
                                  '--output_dir', str(out), '--model_name', 'qwen-7b', '--model_cache_dir', c['model_cache'],
                                  '--dtype', 'bfloat16', '--gen_max_length', str(c['design']['generation_max_length']),
                                  '--progress_every', '100', '--resume', '--label', c['experiment_id']+'_'+name])


def report(run, c):
    d = c['design']
    cohort = base.jsonl(run/'inputs/cohort.jsonl')
    ranking, direct, metrics, comparisons = {}, {}, [], {}
    samples = base.eval_samples(load_graph_record(run/'inputs/test.json'))
    normalize = base.normalization(run/'code_snapshot')
    paper = run/'paper'
    paper.mkdir(exist_ok=True)
    for name in METHODS:
        view = run/'ranking'/name
        vc = base.verify(view)
        identity = read_json(view/'models'/name/'identity.json')
        require(identity == dict(model=name, adapter_sha256=vc['models'][name]['adapter_sha256'],
                    cohort_sha256=c['cohort_sha256'], code_digest=c['code_digest'], score_rule=SCORE_RULE,
                    score_batch_size=2, max_length=4096, seed=2026, dtype='bfloat16', tie_rule='original_candidate_order',
                    base_model_identity=c['base_model_identity'], candidate_count=d['candidate_count']), 'scoring identity mismatch')
        ranking[name] = read_rows(view/'models'/name/'predictions.jsonl', cohort, expected_candidates=d['candidate_count'])
        s = summary(ranking[name], len(cohort))
        require(s['status'] == 'completed', 'incomplete ranking')
        raw_direct = run/'reference/direct_full.jsonl' if name == 'full' else run/name/'evaluation/test/predictions.jsonl'
        direct[name] = audit_predictions(base.jsonl(raw_direct), samples, normalize)
        metrics.append(dict(method=name, label=LABELS[name], n=len(cohort),
                            direct_em_percent=100*sum(r['correct'] for r in direct[name].values())/len(cohort),
                            h1_percent=100*s['h1'], mrr_percent=100*s['mrr'], h3_percent=100*s['h3'],
                            candidate_count=d['candidate_count']))
    for left, right in (('full','random9'), ('full','frequency_matched'), ('frequency_matched','random9')):
        key = left+'_minus_'+right
        rank_result, pairs = compare(ranking[left], ranking[right], cohort, expected_candidates=d['candidate_count'])
        direct_result, dpairs = paired_result(direct[left], direct[right])
        # The legacy helper's field names denote left/right, not Full/Random.
        comparisons[key] = dict(ranking=rank_result, direct_left_minus_right=dict(
            n=direct_result['n'], net_correct=direct_result['net_correct'],
            delta_em_pp=direct_result['delta_sccl_minus_random9_pp'], paired_counts=direct_result['paired_counts']))
        for row, dr in zip(pairs, dpairs):
            require(row['question_id'] == dr['question_id'], 'pair order mismatch')
            row.update(left_direct_correct=dr['sccl_correct'], right_direct_correct=dr['random9_correct'])
        write_csv(paper/('paired_'+key+'.csv'), pairs)
    write_csv(paper/'metrics.csv', metrics)
    write_csv(paper/'paired_all_methods.csv', [dict(question_id=item['question_id'], frequency_group=item['frequency_group'],
        **{name+'_gold_rank':ranking[name][i]['gold_rank'] for name in METHODS},
        **{name+'_direct_correct':int(direct[name][item['question_id']]['correct']) for name in METHODS}) for i,item in enumerate(cohort)])
    costs = []
    for name in ARMS:
        meta = validate_training(run, c, name)
        costs.append(dict(method=name, optimizer_steps=d['optimizer_steps'], scheduler_max_steps=d['scheduler_max_steps'],
                          process_reported_training_seconds=meta.get('seconds'), candidate_forwards=meta.get('candidate_forwards'),
                          timing_scope='process metadata only; retain all resource/event/attempt logs for total cost'))
    write_csv(paper/'training_cost.csv', costs)
    result = dict(status='completed', at=base.now(), experiment_id=c['experiment_id'], dataset=d['dataset'],
                  seed=2026, metrics=metrics, comparisons=comparisons, new_training_runs=2,
                  reused_full=True, original_candidate_count=d['candidate_count'],
                  paper_role='claim-carrying' if d['dataset']=='nell23k' else 'supporting', section_id='experiments',
                  item_id='SC_CTRL_SOURCES_'+d['dataset'].upper(), claim_links=['C2'],
                  limitations=d['limitations'], reporting_policy='retain every arm and signed difference; no score-based inclusion or test-based retuning',
                  next_action='retain complete evidence; write scope-appropriate interpretation; no new runs')
    write_json(paper/'result.json', result)
    end = chr(92)*2
    (paper/'table.tex').write_text('\n'.join(['\\begin{tabular}{lrrrr}', '\\toprule',
        'Method & Direct EM & H@1 & MRR & H@3 '+end, '\\midrule']+
        [f"{m['label']} & {m['direct_em_percent']:.2f} & {m['h1_percent']:.2f} & {m['mrr_percent']:.2f} & {m['h3_percent']:.2f} "+end for m in metrics]+['\\bottomrule','\\end{tabular}'])+'\n')
    (paper/'RESULTS.md').write_text('# '+d['dataset']+' source controls\n\nAll three methods and every signed paired difference are retained. '
        'The quota control inherits Full per-question frequency bins, not an independent frequency-only method. '
        'One Full-selected configuration/seed; direct decoding and original-list ranking are distinct protocols. '
        'WN has eleven original evaluation options; NELL has ten. No full-graph filtered MRR claim.\n')
    entries = []
    folders = ['inputs','reference','bank','code_snapshot','paper','checks']
    folders += [name+'/'+folder for name in ARMS for folder in ('training','evaluation','logs','checks')]
    folders += ['ranking/'+name+'/'+folder for name in METHODS for folder in ('inputs','models','logs')]
    for folder in folders:
        for p in sorted((run/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name != 'run.lock':
                entries.append(dict(path=str(p.relative_to(run)), bytes=p.stat().st_size, sha256=sha256(p)))
    write_json(run/'artifact_manifest.json', dict(at=base.now(), artifacts=entries,
        policy='retain all weights/checkpoints/optimizer/RNG/raw generation/raw candidate scores; mutable control logs separately retained'))
    dest = Path(c['repository_root'])/('11_paper/artifacts/'+d['dataset']+'_source_controls_seed2026')
    dest.mkdir(parents=True, exist_ok=True)
    for p in paper.iterdir():
        shutil.copy2(p, dest/p.name)
    (dest/'.gitattributes').write_text('* -text\n')
    write_json(dest/'evidence_index.json', dict(run_dir=str(run), contract_sha256=sha256(run/'contract.json'),
               archive_sha256=sha256(run/'artifact_manifest.json'), borrowed_sources_must_be_retained=True))
    write_json(run/'completion_audit.json', dict(status='pass', at=base.now(),
               contract_sha256=sha256(run/'contract.json'), archive_sha256=sha256(run/'artifact_manifest.json'),
               model_question_rows=3*len(cohort), candidate_scores=3*len(cohort)*d['candidate_count'],
               direct_flags=3*len(cohort), all_pairs_and_identities_verified=True))
    write_json(run/'status.json', dict(stage='completed', status='completed', at=base.now(), result=result))
    return result


def run_all(run, c):
    prerequisite(c)
    cli_check(run, c)
    write_json(run/'status.json', dict(stage='ranking_full', status='running', at=base.now(), new_training_started=False))
    evaluate_ranking(run, c, 'full')
    for name in ARMS:
        write_json(run/'status.json', dict(stage=name, status='running', at=base.now(), new_training_started=True))
        train(run, c, name)
        evaluate_direct(run, c, name)
        evaluate_ranking(run, c, name)
    return report(run, c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','check','run','report'], nargs='?', default='prepare')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-dir', type=Path)
    a = p.parse_args()
    require(Path(sys.executable).resolve() == base.PYTHON.resolve(), 'use guardenv Python')
    run = (a.run_dir or Path(read_json(a.config)['default_run_dir'])).resolve()
    with base.locked(run):
        try:
            c = prepare(run, a.config) if a.action in ('prepare','run') else base.verify(run)
            if a.action == 'check':
                cli_check(run, c)
            elif a.action == 'run':
                print(json.dumps(run_all(run, c), ensure_ascii=False), flush=True)
            elif a.action == 'report':
                print(json.dumps(report(run, c), ensure_ascii=False), flush=True)
            else:
                print(json.dumps(dict(run_dir=str(run), status='prepared')), flush=True)
        except BaseException as error:
            write_json(run/'status.json', dict(stage=a.action, status='failed', at=base.now(), error=repr(error)))
            base.append_event(run, dict(stage=a.action, status='failed', error=repr(error)))
            raise


if __name__ == '__main__':
    main()
