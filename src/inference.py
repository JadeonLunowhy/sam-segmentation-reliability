"""Official SAM adapter. Annotation-free model and scoring interfaces."""
from dataclasses import dataclass
from pathlib import Path
import time
import sys
import numpy as np
from src.data import ROOT,sha256_file

@dataclass
class Prediction:
    mask: np.ndarray
    logits: np.ndarray
    predicted_iou: float
    decode_seconds: float

def select_candidate(masks,scores):
    index=int(np.argmax(scores))
    return masks[index],float(scores[index]),index

def record_complete(row,config_hash,model_hash,data_manifest_hash=None):
    if row.get('config_hash')!=config_hash or row.get('model_hash')!=model_hash:return False
    if data_manifest_hash is not None and row.get('data_manifest_hash')!=data_manifest_hash:return False
    if data_manifest_hash is not None and not row.get('data_files'):return False
    try:
        if any(sha256_file(ROOT/p)!=h for p,h in row.get('data_files',{}).items()):return False
    except OSError:return False
    path=ROOT/row.get('mask_path','missing')
    if not path.is_file() or not row.get('sweep'):return False
    if row.get('mask_hash')!=sha256_file(path):return False
    try:
        with np.load(path) as m:return all(k in m for k in ['original','flip','perturbations'])
    except Exception:return False

class SamRunner:
    def __init__(self,checkpoint,device='auto'):
        import torch
        sys.path.insert(0,str(ROOT/'third_party/segment-anything'))
        from segment_anything import sam_model_registry,SamPredictor
        self.torch=torch
        self.device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        torch.set_num_threads(6)
        self.model=sam_model_registry['vit_b'](checkpoint=str(checkpoint))
        self.model.to(self.device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.predictor=SamPredictor(self.model)
        self.encoder_calls=0;self.decoder_calls=0

    def sync(self):
        if self.device=='cuda':self.torch.cuda.synchronize()

    def encode(self,image):
        self.sync();start=time.perf_counter()
        with self.torch.inference_mode():self.predictor.set_image(np.ascontiguousarray(image))
        self.sync();self.encoder_calls+=1
        return {'encode_seconds':time.perf_counter()-start,'encoder_calls':1}

    def predict_many(self,points):
        torch=self.torch;points=np.asarray(points,float)
        coords=torch.as_tensor(self.predictor.transform.apply_coords(points,self.predictor.original_size),device=self.device,dtype=torch.float32)[:,None,:]
        labels=torch.ones(coords.shape[:2],device=self.device,dtype=torch.int64)
        self.sync();start=time.perf_counter()
        with torch.inference_mode():
            logits,scores,_=self.predictor.predict_torch(coords,labels,multimask_output=True,return_logits=True)
            idx=scores.argmax(dim=1); batch=torch.arange(len(points),device=self.device)
            selected=logits[batch,idx].cpu().numpy();confidence=scores[batch,idx].cpu().numpy()
        self.sync();elapsed=time.perf_counter()-start;self.decoder_calls+=len(points)
        return [Prediction(l>self.model.mask_threshold,l,float(s),elapsed/len(points)) for l,s in zip(selected,confidence)]

    def predict(self,point_xy):return self.predict_many(np.asarray(point_xy)[None,:])[0]
