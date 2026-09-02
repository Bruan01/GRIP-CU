#!/usr/bin/env python3
"""Run one controlled protocol."""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from priority_distill.config import load_config
from priority_distill.controlled import run_controlled

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name-or-path")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    out = a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if not a.overwrite: raise FileExistsError(f"output exists; pass --overwrite: {out}")
        shutil.rmtree(out)
    summary = run_controlled(repo_root=REPO, config_path=a.config.expanduser().resolve(), config=load_config(a.config.expanduser().resolve()), output_dir=out, model_override=a.model_name_or_path)
    print(f"complete protocol={summary['protocol']} actual_optimizer_steps={summary['training']['actual_optimizer_steps']} output={out}")
if __name__ == "__main__": main()
