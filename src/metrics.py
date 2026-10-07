"""Tie-aware finite-sample risk and failure-detection metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

def selective_curve(iou,reliability):
    iou=np.asarray(iou,float); reliability=np.asarray(reliability,float)
    if not len(iou) or iou.shape!=reliability.shape:raise ValueError('Invalid observation arrays')
    if not np.isfinite(iou).all() or not np.isfinite(reliability).all():raise ValueError('Nonfinite scores')
    order=np.argsort(-reliability,kind='stable'); score=reliability[order]; loss=1-iou[order]
    # Within every tied group, expected cumulative loss uses group mean.
    starts=np.r_[0,np.flatnonzero(np.diff(score)!=0)+1]; ends=np.r_[starts[1:],len(score)]
    expected=np.empty_like(loss)
    for start,end in zip(starts,ends):expected[start:end]=loss[start:end].mean()
    coverage=np.arange(1,len(loss)+1)/len(loss)
    return coverage,np.cumsum(expected)/np.arange(1,len(loss)+1)

def aurc(iou,reliability):return float(selective_curve(iou,reliability)[1].mean())

def failure_metrics(iou,reliability,threshold):
    y=np.asarray(iou)<threshold; score=1-np.asarray(reliability)
    mixed=y.any() and not y.all()
    return {'auroc':float(roc_auc_score(y,score)) if mixed else None,
            'average_precision':float(average_precision_score(y,score)) if mixed else None,
            'failures':int(y.sum()),'cases':len(y),'prevalence':float(y.mean())}

def summarize(iou,scores):
    coverage,risk=selective_curve(iou,scores)
    fail=np.asarray(iou)<0.5
    _,failure_risk=selective_curve(1-fail.astype(float),scores)
    rejected={}
    for c in [0.9,0.75,0.5]:
        k=max(1,int(np.ceil(c*len(iou))))
        rejected[str(c)]=float(1-failure_risk[k-1]*k/fail.sum()) if fail.any() else None
    return {'aurc':float(risk.mean()),'failure_0.5':failure_metrics(iou,scores,0.5),
            'failure_0.75':failure_metrics(iou,scores,0.75),
            'rejected_failure_fraction_0.5':rejected,
            'retained_iou':{str(c):float(1-risk[max(0,int(np.ceil(c*len(iou)))-1)]) for c in [0.9,0.75,0.5]}}
