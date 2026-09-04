#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import torch
from query_graph_lora.modules import inject_lora
class Tiny(torch.nn.Module):
 def __init__(self):super().__init__();self.down_proj=torch.nn.Linear(8,8,bias=False)
 def forward(self,x):return self.down_proj(x)
for method in ['static_rank8','uniform_basis','oracle_hop_route']:
 m=Tiny();c,r=inject_lora(m,method,['down_proj'],8,4,2,16,0);c.set_batch([0,1]);assert m(torch.randn(2,3,8)).shape==(2,3,8);assert r.trainable_parameters==128
print('WSL_RUNTIME_SELF_TEST_PASSED')
