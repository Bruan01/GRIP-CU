#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from query_graph_lora.config import load_config
from query_graph_lora.suite import write_suite_outputs
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/phase_a_oracle.json');p.add_argument('--run-root',type=Path,required=True);p.add_argument('--minimal',action='store_true');p.add_argument('--model');a=p.parse_args();c=load_config(a.config);methods=c['minimal_methods'] if a.minimal else c['methods'];a.run_root.mkdir(parents=True,exist_ok=False)
for method in methods:
 for seed in c['training']['seeds']:
  cmd=[sys.executable,str(ROOT/'scripts/run_experiment.py'),'--config',str(a.config),'--method',method,'--seed',str(seed),'--run-root',str(a.run_root)]+(['--model',a.model] if a.model else []);subprocess.run(cmd,check=True)
result=write_suite_outputs(a.run_root,c);print(f"SUITE_COMPLETE decision={result['gate']['decision']} root={a.run_root}")
