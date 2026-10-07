"""Paired bootstrap keeps all prompt observations from an image together."""
import numpy as np
from src.metrics import aurc,failure_metrics

def grouped_indices(ids,rng):
    ids=np.asarray(ids); groups=[np.flatnonzero(ids==x) for x in np.unique(ids)]
    selected=rng.integers(0,len(groups),len(groups))
    return np.concatenate([groups[i] for i in selected])

def bootstrap_difference(rows,first,second,metric,draws=2000,seed=20261003):
    ids=np.array([r['image_id'] for r in rows]); q=np.array([r['true_iou'] for r in rows])
    a=np.array([r[first] for r in rows]); b=np.array([r[second] for r in rows])
    def value(ix):
        if metric=='aurc':return aurc(q[ix],a[ix])-aurc(q[ix],b[ix])
        aa=failure_metrics(q[ix],a[ix],0.5)['auroc']; bb=failure_metrics(q[ix],b[ix],0.5)['auroc']
        return None if aa is None or bb is None else aa-bb
    estimate=value(np.arange(len(rows))); rng=np.random.default_rng(seed)
    samples=[value(grouped_indices(ids,rng)) for _ in range(draws)]
    defined=np.array([s for s in samples if s is not None])
    return {'estimate':estimate,'ci95':np.quantile(defined,[0.025,0.975]).tolist() if len(defined) else None,
            'valid_draws':len(defined),'requested_draws':draws,'unit':'image','first':first,'second':second,'metric':metric}
