import json
from pathlib import Path
from .routing import METHODS
def load_config(path):
 with Path(path).open(encoding='utf-8') as f:c=json.load(f)
 validate_config(c);return c
def validate_config(c):
 if c.get('format_version')!=1:raise ValueError('format_version must be 1')
 if c.get('evaluation_splits')!=['validation']:raise ValueError('Phase-A must be validation-only')
 if set(c['methods'])-set(METHODS):raise ValueError('unknown method')
 if len(c['methods'])!=len(set(c['methods'])):raise ValueError('duplicate method')
 l=c['lora']
 if l['groups']!=4 or l['groups']*l['group_rank']!=l['total_rank']:raise ValueError('rank layout mismatch')
 required={'static_rank8','uniform_basis','shuffled_oracle_route','oracle_hop_route','oracle_relation_path_route'}
 if not required.issubset(c['methods']):raise ValueError('minimal matrix missing')
 if set(c.get('minimal_methods',[]))!=required:raise ValueError('minimal_methods must match registered Phase-A matrix')
 if c['training']['seeds']!=[43,44] or c['gate']['minimum_seeds']!=2:raise ValueError('seed gate mismatch')
