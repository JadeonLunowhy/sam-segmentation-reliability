"""Ground truth simulates clicks here; production scoring never sees it."""
import numpy as np
from scipy.ndimage import distance_transform_edt
K_INDICES={2:[0,4],4:[0,2,4,6],8:list(range(8))}

def make_prompt(target,regime,seed):
    if not target.any():raise ValueError('Empty target')
    # Explicit zero padding makes distance-to-background valid at image edges.
    dist=distance_transform_edt(np.pad(target,1))[1:-1,1:-1]
    fallback=False
    if regime=='interior':y,x=np.unravel_index(np.argmax(dist),dist.shape)
    elif regime=='boundary':
        eligible=np.argwhere((dist>0)&(dist<=0.02*np.hypot(*target.shape)))
        if len(eligible):y,x=eligible[np.random.default_rng(seed).integers(len(eligible))]
        else:y,x=np.unravel_index(np.argmax(dist),dist.shape); fallback=True
    else:raise ValueError('Unknown click regime')
    return np.array([x,y],dtype=float),{'fallback':fallback,'distance_to_boundary':float(dist[y,x])}

def perturb_points(point,shape,radius_fraction):
    angles=np.arange(8)*np.pi/4
    radius=radius_fraction*np.hypot(*shape)
    pts=np.asarray(point)+radius*np.stack([np.cos(angles),np.sin(angles)],axis=1)
    return np.clip(pts,[0,0],[shape[1]-1,shape[0]-1])

def flip_point(point,width):return np.array([width-1-point[0],point[1]],dtype=float)
