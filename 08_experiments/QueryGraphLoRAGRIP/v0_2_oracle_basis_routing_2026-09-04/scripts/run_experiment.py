#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];sys.path.insert(0,str(ROOT))
from query_graph_lora.config import load_config
from query_graph_lora.experiment import run_one
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/phase_a_oracle.json');p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--model');a=p.parse_args();c=load_config(a.config);s=run_one(REPO,ROOT,a.config.resolve(),c,a.method,a.seed,a.run_root/a.method/f'seed_{a.seed}',a.model);print(f"RUN_COMPLETE method={a.method} seed={a.seed} em={s['metrics']['validation']['canonical_em']:.6f}")
