#!/usr/bin/env python3
"""One independent frequency-proportional FB control; prepare/check are CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HNG = Path(__file__).resolve().parents[1]
ROOT = HNG.parents[1]
sys.path[:0] = [str(HNG/'scripts'), str(HNG/'src')]
import run_fb_random9_control as base
from hard_negative_grip.frequency_only import frequency_only_bank_row, validate_frequency_only_manifest
from hard_negative_grip.matched_control import audit_predictions, parameter_diff, read_json, sha256, training_command, write_csv, write_json
from hard_negative_grip.official_lists import load_train_relation_order, load_relation_text_mapping, load_triples
from hard_negative_grip.task_file import (
    assistant_gold, build_qa_assets_from_task_texts, known_pair_relations,
    load_counterfactual_manifest, load_graph_record, load_relation_eligibility,
    match_train_relation, original_question_ids_from_payload, question_entity_pair,
    train_relation_alias_index,
)
from hard_negative_grip.rerank_audit import SCORE_RULE, read_rows, summary
from run_fb_frequency_matched_control import comparison
from run_fb_rerank_campaign import cache_identity

VARIANT = 'frequency_only'
LABEL = 'Independent frequency-proportional Random-9'
DESTINATION = ROOT/'11_paper/artifacts/fb_frequency_only_control_seed2026'


def make_training_argv(reference, run, model_cache):
    cmd = training_command(str(base.PYTHON), run/'code_snapshot'/base.REL_HNG/'scripts/train_listed_contrastive.py',
                           reference, inputs=run/'inputs', bank=run/'bank/frequency_only.jsonl',
                           output=run/'training', model_cache=Path(model_cache))
    cmd[cmd.index('--listed_negative_source')+1] = VARIANT
    return cmd


def storage_gate(run, design, preparing=False):
    physical = Path(design['training_storage'])
    if preparing:
        physical.mkdir(parents=True, exist_ok=False)
        (run/'training').symlink_to(physical, target_is_directory=True)
    base.require((run/'training').resolve() == physical.resolve(), 'training storage redirect changed')
    free = shutil.disk_usage(physical).free
    # Four complete checkpoints (~10.5 GiB), final adapter (~2.2 GiB), plus 3 GiB reserve.
    if preparing:
        base.require(free >= 16*1024**3, 'physical E drive requires >=16 GiB free; do not delete old runs')
    else:
        base.require(free >= 3*1024**3, 'physical storage reserve exhausted')
    base.require(shutil.disk_usage('/mnt/c').free >= 1024**3, 'C drive log/VHDX reserve exhausted')
    write_json(run/'storage_latest.json', dict(training_physical_path=str(physical.resolve()),
               free_training_volume_bytes=free, free_c_bytes=shutil.disk_usage('/mnt/c').free,
               planned_checkpoints=[8,16,24,32], no_cleanup=True, at=base.now()))


def prepare(run, config):
    if (run/'contract.json').exists():
        c = base.verify(run)
        base.require(c['design_sha256'] == sha256(config), 'design changed after freeze')
        return c
    base.require(not (run/'inputs').exists(), 'partial preparation retained; use fresh directory')
    d = read_json(config)
    base.require((d['variant'],d['seed'],d['k'],d['optimizer_steps'],d['scheduler_max_steps']) ==
                 (VARIANT,2026,9,32,120) and d['frequency_exponent'] == 1, 'unregistered design')
    source = Path(d['source_control_run'])
    base.require(sha256(source/'contract.json') == d['source_contract_sha256'] and
                 sha256(source/'artifact_manifest.json') == d['source_archive_sha256'], 'parent identity changed')
    base.require(read_json(source/'status.json')['status'] == 'completed', 'parent incomplete')
    sc = base.verify(source)
    base.append_event(run, dict(stage='prepare', status='running'))
    artifacts = {}
    base.freeze(run, config, 'inputs/design.json', artifacts)
    overrides = ('scripts/train_listed_contrastive.py', 'src/hard_negative_grip/task_file.py',
                 'src/hard_negative_grip/frequency_only.py', 'scripts/run_fb_frequency_only_control.py',
                 'src/hard_negative_grip/frequency_matched.py', 'scripts/run_fb_frequency_matched_control.py',
                 'scripts/evaluate_matched_candidate_rerank.py', 'src/hard_negative_grip/rerank_audit.py',
                 'scripts/run_fb_rerank_campaign.py')
    excluded = {str(Path('code_snapshot')/base.REL_HNG/p) for p in overrides}
    for relative, record in sc['immutable_artifacts'].items():
        if relative == 'inputs/design.json' or relative in excluded:
            continue
        if not relative.startswith(('inputs/','reference/','code_snapshot/')):
            continue
        target = run/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith('code_snapshot/'):
            shutil.copy2(source/relative, target)
        else:
            target.symlink_to((source/relative).resolve())
        artifacts[relative] = dict(sha256=record['sha256'],bytes=record['bytes'],
                                 source=str(source/relative),borrowed=not relative.startswith('code_snapshot/'))
    for relative in overrides:
        base.freeze(run, HNG/relative, Path('code_snapshot')/base.REL_HNG/relative, artifacts)
    # Freeze only train.txt counts, without constructing or accepting structural features.
    vocabulary = load_train_relation_order(run/'inputs/raw', relation_text_file=run/'inputs/raw/rel2text.json')
    mapping = load_relation_text_mapping(run/'inputs/raw/rel2text.json')
    from collections import Counter
    frequencies = dict(Counter(mapping.get(r,r) for _,r,_ in load_triples(run/'inputs/raw/train.txt')))
    base.require(set(frequencies) == set(vocabulary), 'train-count vocabulary mismatch')
    known = known_pair_relations(run/'inputs/raw', splits=('train',),
                                relation_text_file=run/'inputs/raw/rel2text.json',
                                entity_text_file=run/'inputs/entity2text_train.txt')
    pair_index = {(h.casefold(),t.casefold()):(h,t) for h,t in known}
    payload = read_json(run/'inputs/task.json')
    texts = payload['qa_samples']
    ids = original_question_ids_from_payload(payload,len(texts)) or [f'task_qa:{i}' for i in range(len(texts))]
    base.require(len(texts) == d['qa_count'], 'QA identity changed')
    eligible = load_relation_eligibility(run/'inputs/relation_eligibility.json',run/'inputs/task.json',run/'inputs/raw')
    alias = train_relation_alias_index(vocabulary)
    bank, pools = {}, []
    for qid, text in zip(ids,texts):
        if qid not in eligible:
            continue
        pair = question_entity_pair(text,entity_pairs=pair_index)
        base.require(pair in known,'unverified train entity pair')
        gold = assistant_gold(text)
        row, audit = frequency_only_bank_row(question_id=qid,positive_relation=gold,
                  matched_train_relation=match_train_relation(gold,alias),relation_order=vocabulary,
                  known_relations=known[pair],frequencies=frequencies,seed=2026,k=9)
        bank[qid] = row
        pools.append(audit)
    base.require(set(bank) == eligible and len(bank) == 3109, 'incomplete eligible bank')
    validate_frequency_only_manifest(bank,k=9)
    (run/'bank').mkdir()
    for name, values in [('frequency_only.jsonl',bank.values()),('pool_audit.jsonl',pools)]:
        (run/'bank'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    write_csv(run/'bank/relation_frequencies.csv',[dict(relation=n,train_edge_count=frequencies[n]) for n in sorted(frequencies)])
    # Parent artifacts are consulted only AFTER sampling, to verify pool/cohort parity.
    old_pools = {r['question_id']:r for r in base.jsonl(source/'bank/pool_audit.jsonl')}
    base.require(set(old_pools) == set(bank), 'parent cohort mismatch')
    for p in pools:
        base.require(p['pool_sha256'] == old_pools[p['question_id']]['pool_sha256'] and
                     p['excluded_train_relations'] == old_pools[p['question_id']]['excluded_train_relations'],
                     'independently reconstructed legal pool mismatch')
    parent = load_counterfactual_manifest(run/'reference/counterfactual_bank.jsonl')
    common = dict(seed=2026,relation_order=vocabulary,question_ids=ids,eligible_question_ids=eligible)
    before_text,before = build_qa_assets_from_task_texts(texts,listed_negative_source='structure_counterfactual',counterfactual_manifest=parent,**common)
    after_text,after = build_qa_assets_from_task_texts(texts,listed_negative_source=VARIANT,random_manifest=bank,**common)
    base.require(before_text == after_text,'QA text changed')
    for left,right in zip(before,after):
        base.require({k:v for k,v in left.items() if k not in ('listed_relations','listed_negative_source')} ==
                     {k:v for k,v in right.items() if k not in ('listed_relations','listed_negative_source')},'noncandidate metadata changed')
    selected = [dict(question_id=qid,slot=i,**s) for qid,r in bank.items() for i,s in enumerate(r['selected_negatives'])]
    write_csv(run/'bank/selected_candidates.csv',selected)
    audit = dict(status='pass',n_qa=len(texts),listed_qa=len(bank),k=9,seed=2026,
                 train_only=True,all_train_pairs_verified=True,same_legal_pool=True,
                 same_non_candidate_metadata=True,frequency_exponent=1,
                 sampling_inputs=['train.txt','rel2text.json','training QA','train entity text','existing eligibility gate','seed'],
                 score_files_opened_for_sampling=0,uses_full_candidates_for_sampling=False,
                 uses_gold_frequency_similarity=False,selection_independent_of_full_bank=True,
                 new_training_started=False,performance_results_available=False)
    write_json(run/'bank/preparation_audit.json',audit)
    for p in (run/'bank').iterdir():
        artifacts[str(p.relative_to(run))] = dict(sha256=sha256(p),bytes=p.stat().st_size)
    rerank_source = Path(d['source_rerank_run'])
    base.require(sha256(rerank_source/'contract.json') == d['source_rerank_contract_sha256'], 'reference ranking identity changed')
    base.require(read_json(rerank_source/'status.json')['status'] == 'completed', 'reference rankings incomplete')
    rc = read_json(rerank_source/'contract.json')
    for relative in ('inputs/cohort.jsonl',):
        base.freeze(run,rerank_source/relative,'reference/rerank_cohort.jsonl',artifacts,rc['cohort_sha256'])
    for method in ('full','frequency_matched','random9'):
        for name in ('predictions.jsonl','identity.json','summary.json'):
            rel=f'reference/ranking/{method}/{name}'
            original=rerank_source/'models'/method/name
            expected=next(r['sha256'] for r in read_json(rerank_source/'artifact_manifest.json')['artifacts']
                          if r['path']==f'models/{method}/{name}')
            target=run/rel;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(original.resolve())
            base.require(sha256(target)==expected,'archived reference ranking changed')
            artifacts[rel]=dict(sha256=expected,bytes=target.stat().st_size,borrowed=True,source=str(original))
    storage_gate(run,d,preparing=True)
    cmd=make_training_argv(read_json(run/'reference/run_config.json'),run,sc['model_cache'])
    c=dict(schema='fb-independent-frequency-v1',experiment_id=d['experiment_id'],design=d,design_sha256=sha256(config),
           created_at=base.now(),repository_root=str(ROOT),immutable_artifacts=artifacts,training_argv=cmd,
           model_cache=sc['model_cache'],expected_optimizer_steps=32,expected_scheduler_max_steps=120,
           selection='fixed_step32_before_one_test_no_new_validation',borrowed_inputs_must_be_retained=True)
    write_json(run/'contract.json',c)
    write_json(run/'execution_plan.json',dict(training_argv=cmd,stages=['cli_check','train32','direct_test','ranking_test','report'],stop_after_report=True))
    write_json(run/'environment_prepare.json',base.environment(run))
    write_json(run/'status.json',dict(stage='prepared',status='ready',at=base.now()))
    DESTINATION.mkdir(parents=True,exist_ok=True)
    write_json(DESTINATION/'RUNNING.json',dict(status='prepared',run_dir=str(run),experiment_id=c['experiment_id'],
               contract_sha256=sha256(run/'contract.json'),performance_results_available=False))
    base.append_event(run,dict(stage='prepare',status='completed',audit=audit))
    return c


def validate_mode(run):
    actual=read_json(run/'training/run_config.json')
    base.require(actual['listed_negative_source']==VARIANT and actual['negative_sampling_mode']=='random','wrong loader')
    base.require(Path(actual['random_manifest']).resolve()==(run/'bank/frequency_only.jsonl').resolve(),'wrong bank')
    parameter_diff(read_json(run/'reference/run_config.json'),actual)


def validate_training(run,c):
    validate_mode(run)
    for step in (8,16,24,32):
        p=run/f'training/trainer_listed/checkpoint-{step}'
        for name in ('trainer_state.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth'):
            base.require((p/name).is_file(),'missing retained checkpoint state: '+str(p/name))
    state=read_json(run/'training/trainer_listed/checkpoint-32/trainer_state.json')
    meta=read_json(run/'training/listed/adapter/run_metadata.json')
    base.require(state['global_step']==meta['trainer_global_step']==32 and state['max_steps']==meta['trainer_max_steps']==120,'budget mismatch')
    return state


def train(run,c):
    base.cli_check(run,c);validate_mode(run)
    if (run/'training/listed/adapter/run_metadata.json').exists():
        validate_training(run,c);return
    base.require(not list((run/'training/trainer_listed').glob('checkpoint-*')), 'partial training retained; no automatic retry')
    storage_gate(run,c['design'],preparing=False)
    gpu=base.capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    base.require(gpu.get('returncode')==0 and not gpu['stdout'].strip(),'another GPU compute process is active')
    base.execute(run,'train',c['training_argv']);validate_training(run,c)


def evaluate(run,c):
    validate_training(run,c)
    weights={n:sha256(run/'training/listed/adapter'/n) for n in ('adapter_config.json','adapter_model.safetensors')}
    selection=dict(selection_rule=c['selection'],step=32,adapter_sha256=weights,new_full_validation=False,reopen_after_test=False)
    if (run/'selection.json').exists():
        base.require(read_json(run/'selection.json')==selection,'selected model changed')
    else:
        write_json(run/'selection.json',selection)
    base.execute(run,'eval_test',[str(base.PYTHON),str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_adapter.py'),
                 '--adapter',str(run/'training/listed/adapter'),'--eval_file',str(run/'inputs/test.json'),
                 '--output_dir',str(run/'evaluation/test'),'--model_name','qwen-7b','--model_cache_dir',c['model_cache'],
                 '--dtype','bfloat16','--gen_max_length','32','--progress_every','100','--resume','--label','fb_frequency_only_fixed32_seed2026'])
    rank=run/'rerank';rank.mkdir(exist_ok=True)
    if not (rank/'contract.json').exists():
        (rank/'code_snapshot').symlink_to(run/'code_snapshot',target_is_directory=True)
        (rank/'inputs').mkdir();(rank/'inputs/test.json').symlink_to((run/'inputs/test.json').resolve())
        (rank/'inputs/cohort.jsonl').symlink_to(run/'reference/rerank_cohort.jsonl')
        code={k:v['sha256'] for k,v in c['immutable_artifacts'].items() if k.startswith('code_snapshot/')}
        model_spec=dict(label=LABEL,adapter_dir=str(run/'training/listed/adapter'),adapter_sha256=weights)
        rc=dict(c,models={VARIANT:model_spec},cohort_sha256=sha256(rank/'inputs/cohort.jsonl'),candidate_count=10,
                code_digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
                score_batch_size=2,max_length=4096,score_rule=SCORE_RULE,base_model_identity=cache_identity(c['model_cache']),
                immutable_artifacts={f'../{k}':v for k,v in c['immutable_artifacts'].items()})
        write_json(rank/'contract.json',rc)
    base.execute(run,'rerank',[str(base.PYTHON),str(run/'code_snapshot'/base.REL_HNG/'scripts/evaluate_matched_candidate_rerank.py'),
                              '--run-dir',str(rank),'--model',VARIANT])


def report(run,c):
    validate_training(run,c);base.verify(run)
    cohort=base.jsonl(run/'reference/rerank_cohort.jsonl')
    samples=base.eval_samples(load_graph_record(run/'inputs/test.json'))
    direct=audit_predictions(base.jsonl(run/'evaluation/test/predictions.jsonl'),samples,base.normalization(run/'code_snapshot'))
    base.require(len(direct)==len(cohort)==10000,'incomplete cohort')
    weights=read_json(run/'selection.json')['adapter_sha256']
    base.require(read_json(run/'evaluation/test/reuse_identity.json')['adapter']==weights,'direct model identity mismatch')
    rank=run/'rerank/models'/VARIANT
    base.require(read_json(rank/'identity.json')['adapter_sha256']==weights,'ranking model identity mismatch')
    rows=read_rows(rank/'predictions.jsonl',cohort)
    measured=summary(rows,10000);base.require(measured['status']=='completed','incomplete ranking')
    metrics=[];pairs={};rank_refs={}
    for method in ('full','frequency_matched','random9'):
        refs=read_rows(run/f'reference/ranking/{method}/predictions.jsonl',cohort)
        base.require(len(refs)==10000,'incomplete reference ranking')
        rank_refs[method]=refs;s=summary(refs,10000)
        metrics.append(dict(method=method,n=10000,direct_em_percent=sum(r['direct_correct'][method] for r in cohort)/100,
                            h1_percent=100*s['h1'],mrr_percent=100*s['mrr'],h3_percent=100*s['h3']))
        reference_direct={r['question_id']:dict(question_id=r['question_id'],question=r['question'],target=r['target'],split='test',
                          correct=r['direct_correct'][method]) for r in cohort}
        pd,paired=comparison(reference_direct,direct,method,VARIANT)
        from hard_negative_grip.rerank_audit import compare
        pr,rank_pairs=compare(refs,rows,cohort)
        pairs[method+'_minus_'+VARIANT]=dict(direct=pd,ranking=pr)
        write_csv(run/f'paper/paired_{method}_minus_frequency_only_direct.csv',paired)
        write_csv(run/f'paper/paired_{method}_minus_frequency_only_ranking.csv',rank_pairs)
    metrics.append(dict(method=VARIANT,n=10000,direct_em_percent=sum(r['correct'] for r in direct.values())/100,
                        h1_percent=100*measured['h1'],mrr_percent=100*measured['mrr'],h3_percent=100*measured['h3']))
    result=dict(status='completed',at=base.now(),experiment_id=c['experiment_id'],seed=2026,n=10000,metrics=metrics,comparisons=pairs,
                training_runs=1,selection='fixed_step32',paper_role='claim-carrying',section_id='experiments',
                item_id='SC_CTRL_FREQONLY_FB',claim_links=['C2'],preparation_audit=read_json(run/'bank/preparation_audit.json'),
                limitations=['one seed at historical Full-selected configuration; no equal HPO',
                            'only one independent frequency rule (exponent=1) tested; a negative result does not establish structural causality',
                            'shared 464 QA repairs and historical test exposure retained',
                            'fixed original ten-option ranking, not full-graph filtered MRR',
                            'no equivalence or generalization claim from small observed differences'],
                next_action='interpret all signed results; do not tune sampler or reopen test')
    write_json(run/'paper/result.json',result);write_csv(run/'paper/metrics.csv',metrics)
    write_csv(run/'paper/paired_all_methods.csv',[dict(question_id=r['question_id'],**{m+'_gold_rank':rank_refs[m][i]['gold_rank'] for m in rank_refs},
                 frequency_only_gold_rank=rows[i]['gold_rank'],**{m+'_direct_correct':int(v) for m,v in r['direct_correct'].items()},
                 frequency_only_direct_correct=int(direct[r['question_id']]['correct'])) for i,r in enumerate(cohort)])
    for name in ('preparation_audit.json','relation_frequencies.csv','selected_candidates.csv'):
        shutil.copy2(run/'bank'/name,run/'paper'/name)
    base.inventory(run)
    # The ranking worker has a separate contract; retain and hash its scores too.
    manifest=read_json(run/'artifact_manifest.json')
    for p in sorted((run/'rerank').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'run.lock':
            manifest['artifacts'].append(dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha256(p)))
    write_json(run/'artifact_manifest.json',manifest)
    DESTINATION.mkdir(parents=True,exist_ok=True)
    for p in (run/'paper').iterdir():
        if p.is_file():shutil.copy2(p,DESTINATION/p.name)
    write_json(DESTINATION/'evidence_index.json',dict(run_dir=str(run),contract_sha256=sha256(run/'contract.json'),
               archive_sha256=sha256(run/'artifact_manifest.json'),borrowed_sources_must_be_retained=True))
    write_json(DESTINATION/'RUNNING.json',dict(status='completed',run_dir=str(run),experiment_id=c['experiment_id'],at=base.now()))
    write_json(run/'status.json',dict(stage='completed',status='completed',at=base.now(),result=result))
    base.append_event(run,dict(stage='report',status='completed',result=result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','check','train','evaluate','report','run'],default='prepare',nargs='?')
    p.add_argument('--config',type=Path,default=HNG/'configs/fb_frequency_only.json');p.add_argument('--run-dir',type=Path)
    a=p.parse_args();d=read_json(a.config);run=(a.run_dir or Path(d['default_run_dir'])).resolve()
    base.require(Path(sys.executable).resolve()==base.PYTHON.resolve(),'use guardenv Python')
    with base.locked(run):
        try:
            c=prepare(run,a.config) if a.action in ('prepare','run') else base.verify(run)
            print(json.dumps(dict(action=a.action,run_dir=str(run),experiment_id=c['experiment_id'])),flush=True)
            if a.action in ('check','run'):base.cli_check(run,c);validate_mode(run)
            if a.action in ('train','run'):train(run,c)
            if a.action in ('evaluate','run'):evaluate(run,c)
            if a.action in ('report','run'):print(json.dumps(report(run,c),ensure_ascii=False),flush=True)
        except BaseException as error:
            write_json(run/'status.json',dict(stage=a.action,status='failed',at=base.now(),error=repr(error)))
            base.append_event(run,dict(stage=a.action,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
