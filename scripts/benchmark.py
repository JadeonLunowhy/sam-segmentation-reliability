"""Actual budget timings, not linear extrapolation from a large batch."""
import json
import time
import numpy as np
from src.data import ROOT,Case,load_case
from src.inference import SamRunner
from src.prompts import make_prompt,perturb_points,K_INDICES,flip_point

if __name__=='__main__':
    cfg=json.loads((ROOT/'configs/study.json').read_text());manifest=json.loads((ROOT/'data/splits.json').read_text());runner=SamRunner(ROOT/cfg['checkpoint'],cfg['device'])
    frozen=json.loads((ROOT/'configs/frozen.json').read_text());radius=float(frozen['selected'][1:].split('_k')[0])
    cases=[Case(**c) for c in manifest['cases'] if c['split']=='validation'][:5]
    timings=[]
    for case in cases:
        image,target,valid=load_case(case);point,_=make_prompt(target,'interior',cfg['seed'])
        enc=runner.encode(image)['encode_seconds'];runner.predict(point) # warm decoder
        original=runner.predict(point).decode_seconds
        pts=perturb_points(point,target.shape,radius)
        budgets={}
        for k in [2,4,8]:
            budgets[str(k)]=sum(p.decode_seconds for p in runner.predict_many(pts[K_INDICES[k]]))
        flip_enc=runner.encode(image[:,::-1])['encode_seconds'];flip_dec=runner.predict(flip_point(point,image.shape[1])).decode_seconds
        timings.append({'image_id':case.image_id,'encoder_seconds':enc,'original_decode_seconds':original,
                        'additional_prompt_seconds':budgets,'additional_flip_seconds':flip_enc+flip_dec})
    (ROOT/'results/budget_timing.json').write_text(json.dumps({'device':runner.device,'radius':radius,'samples':timings,'note':'measured batches; five validation images; inference-only'},indent=2))
    print(json.dumps(timings,indent=2))
