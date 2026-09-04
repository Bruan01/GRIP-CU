#!/usr/bin/env python3
"""Validate controlled experiment data/config without loading a model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from priority_distill.candidates import build_candidate_pools
from priority_distill.config import load_config
from priority_distill.io_utils import sha256_file, write_json
from priority_distill.records import build_evaluation_prompt, load_splits
from priority_distill.supervision import build_method_supervision

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/direct_answer_only.json')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/setup_audit.json')
    a=p.parse_args(); config_path=a.config.expanduser().resolve(); config=load_config(config_path)
    splits, paths, split_audit=load_splits(REPO, config['data']); seed=int(config['candidates']['seed'])
    pools=build_candidate_pools(splits['train'], int(config['candidates']['distractor_count']), seed)
    oracle=build_method_supervision(splits['train'], pools, 'oracle_priority_equal_token', seed)
    prompts=[build_evaluation_prompt(row['text']) for row in splits['test']]
    invariants={
      'candidate_pools_cover_train': len(pools)==len(splits['train']),
      'four_candidates_per_pool': all(len(pool)==4 for pool in pools.values()),
      'one_gold_per_pool': all(sum(candidate['is_gold'] for candidate in pool)==1 for pool in pools.values()),
      'distractors_train_only': all(candidate['source_split']=='train' for pool in pools.values() for candidate in pool),
      'oracle_has_one_gold_path': all(row['candidate_count']==1 and row['gold_position']==0 for row in oracle),
      'evaluation_has_no_candidate_evidence': all('Candidate evidence:' not in prompt for prompt in prompts),
      'controlled_protocol_is_valid': config['protocol'] in {'direct_answer_only','oracle_two_stage'},
    }
    if not all(invariants.values()): raise AssertionError(invariants)
    payload={'status':'READY_FOR_WSL_GPU','protocol':config['protocol'],'config':str(config_path.relative_to(REPO)),
             'config_sha256':sha256_file(config_path),'data':{**split_audit,'paths':{s:str(x.relative_to(REPO)) for s,x in paths.items()},'sha256':{s:sha256_file(x) for s,x in paths.items()}},
             'oracle_rows':len(oracle),'invariants':invariants}
    write_json(a.output.expanduser().resolve(), payload)
    print(json.dumps({'status':payload['status'],'protocol':payload['protocol'],'train':len(splits['train']),'validation':len(splits['validation']),'test':len(splits['test']),'output':str(a.output)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
