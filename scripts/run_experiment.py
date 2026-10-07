"""Run validation first, then frozen-configuration held-out experiments."""
import argparse
import hashlib
import json
import time
import numpy as np
from src.data import ROOT,Case,load_case,sha256_file,verify_case_files
from src.inference import SamRunner,record_complete
from src.prompts import make_prompt,perturb_points,flip_point,K_INDICES
from src.scoring import true_iou,threshold_stability,reliability_scores

def cell_key(radius,k):return f'r{radius:g}_k{k}'
def run_split(split,config_path=ROOT/'configs/study.json',resume=True,limit=None):
    cfg=json.loads(config_path.read_text()); cfg_hash=sha256_file(config_path)
    if split!='validation':
        frozen=json.loads((ROOT/'configs/frozen.json').read_text())
        if frozen['validation_sha256']!=sha256_file(ROOT/'results/validation.jsonl'):raise ValueError('Validation changed after selection')
        if frozen['study_sha256']!=cfg_hash:raise ValueError('Study config changed after selection')
    model_hash=sha256_file(ROOT/cfg['checkpoint'])
    manifest=json.loads((ROOT/'data/splits.json').read_text())
    manifest_hash=sha256_file(ROOT/'data/splits.json')
    if split!='validation':
        if frozen.get('data_manifest_hash')!=manifest_hash:raise ValueError('Dataset changed after validation selection')
        if frozen.get('model_hash')!=model_hash:raise ValueError('Model changed after validation selection')
    cases=[Case(**c) for c in manifest['cases'] if c['split']==split]
    if limit is not None:cases=cases[:limit]
    folder=ROOT/'results/cases'/split;folder.mkdir(parents=True,exist_ok=True)
    runner=SamRunner(ROOT/cfg['checkpoint'],cfg['device'])
    print('Device',runner.device,'images',len(cases),flush=True)
    for n,case in enumerate(cases):
        verify_case_files(case)
        paths=[folder/(case.image_id+'_'+regime+'.json') for regime in cfg['regimes']]
        if resume and all(p.exists() and record_complete(json.loads(p.read_text()),cfg_hash,model_hash,manifest_hash) for p in paths):continue
        image,target,valid=load_case(case);h,w=target.shape
        encoded=runner.encode(image);items=[]
        for regime in cfg['regimes']:
            stable_seed=cfg['seed']+int(hashlib.sha256((case.image_id+regime).encode()).hexdigest()[:8],16)
            point,meta=make_prompt(target,regime,stable_seed)
            original=runner.predict(point)
            variations=[];sweep={};outside={};decode={}
            for radius in cfg['radii']:
                pts=perturb_points(point,target.shape,radius)
                preds=runner.predict_many(pts);masks=np.stack([p.mask for p in preds]);variations.append(masks)
                inside=np.array([target[int(round(y)),int(round(x))] for x,y in pts])
                decode[str(radius)]=sum(p.decode_seconds for p in preds)
                for k in cfg['budgets']:
                    key=cell_key(radius,k);ix=K_INDICES[k]
                    sweep[key]=reliability_scores(original.predicted_iou,original.mask,masks[ix],original.mask)
                    outside[key]=float((~inside[ix]).mean())
            items.append((regime,point,meta,original,np.stack(variations),sweep,outside,decode))
        flip_encoding=runner.encode(image[:,::-1])
        for regime,point,meta,original,variations,sweep,outside,decode in items:
            flipped=runner.predict(flip_point(point,w)); mapped=flipped.mask[:,::-1]
            for scores in sweep.values():scores['flip']=reliability_scores(original.predicted_iou,original.mask,variations[0,:2],mapped)['flip']
            path=folder/(case.image_id+'_'+regime+'.npz')
            np.savez_compressed(path,original=original.mask,flip=mapped,perturbations=variations)
            row={'image_id':case.image_id,'dataset':case.dataset,'split':split,'regime':regime,'point':point.tolist(),
                 'prompt_diagnostics':meta,'true_iou':true_iou(original.mask,target,valid),'confidence':original.predicted_iou,
                 'stability':threshold_stability(original.logits),'sweep':sweep,'outside':outside,'object_fraction':float(target.mean()),
                 'timing':{'encoding':encoded['encode_seconds'],'flip_encoding':flip_encoding['encode_seconds'],
                           'original_decode':original.decode_seconds,'flip_decode':flipped.decode_seconds,'perturbation_batches_8':decode},
                 'device':runner.device,'config_hash':cfg_hash,'model_hash':model_hash,'data_manifest_hash':sha256_file(ROOT/'data/splits.json'),
                 'data_files':{case.image_path:case.file_hashes['image'],case.target_path:case.file_hashes['target'],case.valid_path:case.file_hashes['valid']},
                 'mask_path':str(path.relative_to(ROOT)),'mask_hash':sha256_file(path)}
            receipt=path.with_suffix('.json');temp=receipt.with_suffix('.json.part')
            temp.write_text(json.dumps(row,indent=2));temp.replace(receipt)
        print(split,n+1,'/',len(cases),case.image_id,'predicted IoU',[round(x[3].predicted_iou,3) for x in items],flush=True)
    rows=[]
    for case in cases:
        for regime in cfg['regimes']:
            p=folder/(case.image_id+'_'+regime+'.json')
            row=json.loads(p.read_text())
            if not record_complete(row,cfg_hash,model_hash,manifest_hash):raise ValueError('Incomplete observation '+str(p))
            rows.append(row)
    output=ROOT/'results'/(split+'.jsonl')
    output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    metadata={'device':runner.device,'images':len(cases),'cases':len(rows),'frozen_parameters':all(not p.requires_grad for p in runner.model.parameters()),
              'model_hash':model_hash,'config_hash':cfg_hash,'encoder_calls_this_run':runner.encoder_calls,'decoder_calls_this_run':runner.decoder_calls,
              'torch_version':runner.torch.__version__,'peak_cuda_allocated_bytes':runner.torch.cuda.max_memory_allocated() if runner.device=='cuda' else None}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test','extra'],required=True);p.add_argument('--limit',type=int);args=p.parse_args()
    run_split(args.split,limit=args.limit)
