"""Use your image and xy click. Ground-truth masks are not required."""
import argparse
import json
import numpy as np
from PIL import Image
from src.data import ROOT
from src.inference import SamRunner
from src.prompts import perturb_points,K_INDICES
from src.scoring import reliability_scores,threshold_stability

def demo(image_path,x,y,output):
    frozen=json.loads((ROOT/'configs/frozen.json').read_text());radius,k=frozen['selected'][1:].split('_k');radius=float(radius);k=int(k)
    cfg=json.loads((ROOT/'configs/study.json').read_text());runner=SamRunner(ROOT/cfg['checkpoint'],cfg['device'])
    image=np.asarray(Image.open(image_path).convert('RGB'));h,w=image.shape[:2]
    if not (0<=x<w and 0<=y<h):raise ValueError('Click outside image')
    runner.encode(image);original=runner.predict(np.array([x,y]));points=perturb_points([x,y],(h,w),radius)[K_INDICES[k]]
    variations=runner.predict_many(points);scores=reliability_scores(original.predicted_iou,original.mask,np.stack([p.mask for p in variations]),original.mask)
    scores.pop('flip');scores['stability']=threshold_stability(original.logits)
    path=ROOT/output;path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(original.mask.astype('uint8')*255).save(path)
    path.with_suffix('.json').write_text(json.dumps({'scores':scores,'cell':frozen['selected'],'point':[x,y],
                  'interpretation':'ranking scores, not calibrated correctness probabilities','device':runner.device},indent=2))
    print(scores)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--x',type=float,required=True);p.add_argument('--y',type=float,required=True);p.add_argument('--output',default='results/demo_mask.png');a=p.parse_args();demo(a.image,a.x,a.y,a.output)
