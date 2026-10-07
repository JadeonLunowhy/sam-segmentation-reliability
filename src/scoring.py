"""Reliability estimation uses predictions only, never reference labels."""
import numpy as np

def mask_iou(a,b,valid=None):
    a=np.asarray(a,dtype=bool); b=np.asarray(b,dtype=bool)
    if a.shape!=b.shape:raise ValueError('Mask dimensions mismatch')
    if valid is not None:a=a&valid; b=b&valid
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 1.

def true_iou(prediction,target,valid):
    if not (target&valid).any():raise ValueError('Empty valid target')
    return mask_iou(prediction,target,valid)

def threshold_stability(logits,threshold=0,offset=1):
    high=np.count_nonzero(logits>threshold+offset)
    low=np.count_nonzero(logits>threshold-offset)
    # Upstream 0/0 is undefined. A no-foreground prediction is unreliable.
    return float(high/low) if low else 0.

def reliability_scores(predicted_iou,original,perturbations,flipped):
    agreements=np.array([mask_iou(original,p) for p in perturbations])
    mean=float(agreements.mean())
    return {'confidence':float(predicted_iou),'consistency':mean,
            'minimum':float(agreements.min()),'fusion':float((np.clip(predicted_iou,0,1)+mean)/2),
            'flip':mask_iou(original,flipped)}
