"""Verified public data downloads and annotation adapters."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Case:
    image_id: str
    dataset: str
    split: str
    image_path: str
    target_path: str
    valid_path: str
    file_hashes: dict | None = None

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def download(url, path):
    """Complete files have a manifest; incomplete transfers never count as data."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    receipt=path.with_suffix(path.suffix+'.json')
    if path.exists() and receipt.exists():
        meta=json.loads(receipt.read_text())
        if meta['sha256']==sha256_file(path):return meta
    temp=path.with_suffix(path.suffix+'.part')
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'CV-Course-Research/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response, open(temp,'wb') as f:
                length=response.headers.get('Content-Length'); total=0
                while chunk:=response.read(1024*1024): f.write(chunk); total+=len(chunk)
                if length is not None and total!=int(length):raise IOError('Incomplete transfer')
                meta={'url':url,'resolved_url':response.geturl(),'bytes':total,'sha256':sha256_file(temp)}
            temp.replace(path); receipt.write_text(json.dumps(meta,indent=2)); return meta
        except Exception:
            if attempt==3:raise
            time.sleep(2*(attempt+1))

def safe_extract(archive,destination):
    destination=Path(destination).resolve(); destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as tar:
        members=tar.getmembers()
        for m in members:
            p=(destination/m.name).resolve()
            if not p.is_relative_to(destination) or m.issym() or m.islnk():
                raise ValueError('Unsafe archive member: '+m.name)
        tar.extractall(destination,members=members,filter='data')

def pet_trimap(mask):
    mask=np.asarray(mask)
    if not set(np.unique(mask)).issubset({1,2,3}):raise ValueError('Unexpected Pet trimap labels')
    return mask==1,mask!=3

def image_files(folder):
    return [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]

def split_ids(ids,n_validation,seed):
    ids=sorted(ids)
    if len(set(ids))!=len(ids):raise ValueError('Duplicate image identities')
    order=np.random.default_rng(seed).permutation(len(ids))
    return ([ids[i] for i in order[:n_validation]],[ids[i] for i in order[n_validation:]])

def verify_case_files(case):
    if not case.file_hashes:raise ValueError('Missing data-file fingerprints; prepare data first')
    for key,path in [('image',case.image_path),('target',case.target_path),('valid',case.valid_path)]:
        if sha256_file(ROOT/path)!=case.file_hashes.get(key):raise ValueError('Data file changed: '+str(path))

def load_case(case):
    verify_case_files(case)
    image=np.asarray(Image.open(ROOT/case.image_path).convert('RGB'))
    target=np.asarray(Image.open(ROOT/case.target_path))>0
    valid=np.asarray(Image.open(ROOT/case.valid_path))>0
    if image.shape[:2]!=target.shape or target.shape!=valid.shape:raise ValueError('Annotation dimensions mismatch')
    if not (target&valid).any():raise ValueError('Empty annotated target')
    return image,target,valid

def save_case(image_id,dataset,split,image_path,fg,valid):
    out=ROOT/'data/prepared'/dataset; out.mkdir(parents=True,exist_ok=True)
    tp=out/(image_id+'_target.png'); vp=out/(image_id+'_valid.png')
    Image.fromarray(fg.astype('uint8')*255).save(tp)
    Image.fromarray(valid.astype('uint8')*255).save(vp)
    return Case(image_id,dataset,split,str(Path(image_path).relative_to(ROOT)),str(tp.relative_to(ROOT)),str(vp.relative_to(ROOT)),
                {'image':sha256_file(image_path),'target':sha256_file(tp),'valid':sha256_file(vp)})

def prepare_data(config):
    main=ROOT/'data/raw/iseg'; images={p.stem:p for p in image_files(main/'images')}
    if len(images)!=151:raise ValueError('Expected all 151 main images before creating splits')
    ids=sorted(images); val,test=split_ids(ids,config['validation_images'],config['seed'])
    cases=[]; diagnostics=[]
    for name in ids:
        m=np.asarray(Image.open(main/'images-gt'/(name+'.png')))
        if not set(np.unique(m)).issubset({0,128,255}):raise ValueError('Unexpected iseg annotation: '+name)
        cases.append(save_case(name,'iseg','validation' if name in val else 'test',images[name],m==255,m!=128))
    pet=ROOT/'data/raw/pets'
    if not (pet/'images').is_dir():extract_pet_mirror(config)
    groups={}
    for line in (pet/'annotations/test.txt').read_text().splitlines():
        name,breed,*_=line.split();groups.setdefault(int(breed),[]).append(name)
    rng=np.random.default_rng(config['seed'])
    for breed,names in sorted(groups.items()):
        names=sorted(names)
        for index in rng.choice(len(names),config['extra_per_breed'],replace=False):
            name=names[index]; fg,valid=pet_trimap(np.asarray(Image.open(pet/'annotations/trimaps'/(name+'.png'))))
            cases.append(save_case(name,'pets','extra',pet/'images'/(name+'.jpg'),fg,valid))
    fingerprints={}
    for case in cases:
        image,target,valid=load_case(case)
        key=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        if key in fingerprints:raise ValueError('Duplicate decoded image: '+case.image_id+' and '+fingerprints[key])
        fingerprints[key]=case.image_id
        diagnostics.append({'image_id':case.image_id,'dataset':case.dataset,'split':case.split,
                            'shape':list(image.shape),'image_sha256':sha256_file(ROOT/case.image_path),
                            'decoded_sha256':key,'foreground_pixels':int(target.sum()),'ignored_pixels':int((~valid).sum())})
    manifest={'seed':config['seed'],'cases':[asdict(c) for c in cases],'diagnostics':diagnostics,'exclusions':[],
              'label_mapping':{'iseg':{'foreground':255,'background':0,'ignored':128},'pets':{'foreground':1,'background':2,'ignored':3}}}
    (ROOT/'data/splits.json').write_text(json.dumps(manifest,indent=2))
    return cases

def extract_pet_mirror(config):
    """Extract selected original JPEG bytes; keep official identity and labels."""
    import pyarrow.parquet as pq
    pet=ROOT/'data/raw/pets';groups={};classes={}
    for line in (pet/'annotations/test.txt').read_text().splitlines():
        name,breed,*_=line.split();groups.setdefault(int(breed),[]).append(name);classes[name]=int(breed)
    rng=np.random.default_rng(config['seed']);wanted=set()
    for breed,names in sorted(groups.items()):
        names=sorted(names)
        wanted.update(names[i] for i in rng.choice(len(names),config['extra_per_breed'],replace=False))
    out=pet/'images';out.mkdir(parents=True,exist_ok=True);saved=[]
    for batch in pq.ParquetFile(pet/'test.parquet').iter_batches(batch_size=128):
        for row in batch.to_pylist():
            name=row['image_id']
            if name not in wanted:continue
            if row['label']+1!=classes[name]:raise ValueError('Mirror class does not match official test list: '+name)
            path=out/(name+'.jpg');path.write_bytes(row['image']['bytes'])
            saved.append({'image_id':name,'sha256':sha256_file(path),'official_class':classes[name]})
    if set(r['image_id'] for r in saved)!=wanted:raise ValueError('Missing selected mirrored images')
    (pet/'selected_mirror_images.json').write_text(json.dumps({'mirror_sha256':sha256_file(pet/'test.parquet'),'images':saved},indent=2))
