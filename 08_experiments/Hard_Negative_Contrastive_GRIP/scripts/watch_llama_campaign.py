#!/usr/bin/env python3
"""Read-only process/disk watchdog. Emits durable alerts; never restarts workers."""
import json,os,time,shutil,subprocess
from pathlib import Path
r=Path('/home/kieran/grip-cu-runs/llama_four_20261111')
def save(name,v):
 p=r/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
while True:
 try:
  s=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {}
  launch=json.loads((r/'launch.json').read_text()) if (r/'launch.json').exists() else {}
  alive=False
  if launch.get('pid'):
   try:os.kill(launch['pid'],0);alive=True
   except ProcessLookupError:pass
  disk={d:shutil.disk_usage(d).free for d in ['/mnt/c','/mnt/e']}
  gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,timeout=10)
  status=dict(at=time.time(),campaign=s,controller_alive=alive,free_bytes=disk,gpu=gpu.stdout.strip())
  alerts=[]
  if s.get('status')=='failed':alerts.append('campaign failed; queue stopped, no retry')
  if launch and not alive and s.get('status')!='completed':alerts.append('controller exited without completed status')
  if disk['/mnt/e']<4*1024**3:alerts.append('E volume below 4GiB launch reserve')
  if disk['/mnt/c']<1024**3:alerts.append('C volume below 1GiB launch reserve')
  if (r/'status.json').exists() and time.time()-(r/'status.json').stat().st_mtime>180:alerts.append('controller heartbeat older than 180 seconds')
  status['alerts']=alerts;save('watchdog.json',status)
  if alerts:
   with (r/'alerts.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
  if s.get('status') in ['completed','failed'] or (launch and not alive):break
 except Exception as e:save('watchdog_error.json',dict(at=time.time(),error=repr(e)))
 time.sleep(30)
