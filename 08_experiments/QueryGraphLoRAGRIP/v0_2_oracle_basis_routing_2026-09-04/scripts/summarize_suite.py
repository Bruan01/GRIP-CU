#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from query_graph_lora.config import load_config
from query_graph_lora.suite import write_suite_outputs
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/phase_a_oracle.json');p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();print(write_suite_outputs(a.run_root,load_config(a.config))['gate']['decision'])
