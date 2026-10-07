"""Range-verified downloader for large artifacts behind buffering proxies."""
from src.data import ROOT,sha256_file
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import json
import time
import argparse
from pathlib import Path

def range_download(url,path,workers=6,chunk=8*1024*1024):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    sep='&' if '?' in url else '?'
    def request(start,end):
        req=urllib.request.Request(url+sep+'segment='+str(start)+'&end='+str(end),headers={'Range':f'bytes={start}-{end}'})
        return urllib.request.urlopen(req,timeout=120)
    with request(0,0) as r:
        cr=r.headers.get('Content-Range','')
        if r.status!=206 or not cr.startswith('bytes 0-0/'):raise IOError('Server did not honor Range: '+cr)
        total=int(cr.split('/')[-1]); r.read()
    parts=ROOT/'tmp/chunks'/path.name; parts.mkdir(parents=True,exist_ok=True)
    starts=list(range(0,total,chunk))
    def fetch(start):
        end=min(start+chunk,total)-1; part=parts/str(start)
        if part.exists() and part.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                with request(start,end) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise IOError('Wrong range response')
                    data=r.read()
                if len(data)!=end-start+1:raise IOError('Short chunk')
                part.write_bytes(data)
                print(path.name,start,end,total,flush=True);return
            except Exception:
                if attempt==4:raise
                time.sleep(2*(attempt+1))
    with ThreadPoolExecutor(workers) as pool:list(pool.map(fetch,starts))
    temp=path.with_suffix(path.suffix+'.assembled')
    with open(temp,'wb') as f:
        for start in starts:
            with open(parts/str(start),'rb') as p:
                while block:=p.read(1024*1024):f.write(block)
    if temp.stat().st_size!=total:raise IOError('Assembly size mismatch')
    temp.replace(path)
    meta={'url':url,'bytes':total,'sha256':sha256_file(path),'transfer':'verified byte ranges'}
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(meta,indent=2))
    print('Complete',meta,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('path');p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    range_download(args.url,ROOT/args.path,workers=args.workers)
