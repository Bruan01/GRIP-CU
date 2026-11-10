from pathlib import Path
import json,hashlib,requests,time,shutil
r=Path('/home/kieran/grip-cu-runs/llama_four_20261111');inv=json.loads(Path('/mnt/c/users/administrator/desktop/实验/grip-cu/08_experiments/Hard_Negative_Contrastive_GRIP/docs/llama_four_dataset_20261111/model_file_inventory.json').read_text())
out=Path('/mnt/e/GRIP-CU/model_cache/meta-llama--Llama-3.1-8B-Instruct');out.mkdir(parents=True,exist_ok=True)
def write(v):
 p=r/'download_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
files=[x for x in inv['files'] if x['bytes'] and '/' not in x['path'] and x['path']!='.gitattributes']
try:
 for x in files:
  p=out/x['path'];temp=p.with_suffix(p.suffix+'.partial')
  if p.exists():
   assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];continue
  for attempt in range(3):
   offset=temp.stat().st_size if temp.exists() else 0
   assert shutil.disk_usage(out).free>x['bytes']-offset+3*1024**3,'disk reserve insufficient'
   write(dict(status='downloading',file=x['path'],bytes=offset,total=x['bytes'],attempt=attempt))
   try:
    u='https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/resolve/master/'+x['path']
    with requests.get(u,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as resp:
     resp.raise_for_status()
     if offset and resp.status_code!=206:offset=0
     with temp.open('ab' if offset else 'wb') as f:
      mark=time.monotonic()
      for b in resp.iter_content(4*1024**2):
       if b:f.write(b);offset+=len(b)
       if time.monotonic()-mark>10:write(dict(status='downloading',file=x['path'],bytes=offset,total=x['bytes']));mark=time.monotonic()
    assert temp.stat().st_size==x['bytes'],'download length mismatch'
    assert sha(temp)==x['sha256'],'download SHA mismatch'
    temp.replace(p);print('verified',x['path'],flush=True);break
   except Exception:
    if attempt==2:raise
    time.sleep(3)
 write(dict(status='completed',model_dir=str(out),verified_files=len(files),at=time.time()))
except BaseException as e:write(dict(status='failed',error=repr(e)));raise
