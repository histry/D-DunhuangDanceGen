"""Explicit hard/editable domains; source-only conditions and support proxies."""
import numpy as np
import torch

def regions(fixed,order):
    """Partition every derivative template once: fixed, crossing, editable."""
    if order<1 or fixed.ndim!=2 or fixed.dtype!=torch.bool or fixed.shape[1]<=order:raise ValueError("Frame mask/order")
    chunks=torch.stack([fixed[:,i:fixed.shape[1]-order+i] for i in range(order+1)],-1)
    all_fixed=chunks.all(-1);any_fixed=chunks.any(-1)
    return dict(fixed=all_fixed,crossing=any_fixed&~all_fixed,editable=~any_fixed)

def numpy_regions(hard,order=3):
    return {n:m[0].numpy() for n,m in regions(torch.tensor(hard[:,:,4:].all(-1)),order).items()}

def conditions(y,cfg,variant=0,pattern=0,hard=True):
    if y.shape!=(1,225,151):raise ValueError("One global225 target")
    left,right=cfg["contexts"][variant];end=225-right
    k=torch.zeros_like(y,dtype=torch.bool);k[:,:left,4:]=True;k[:,end:,4:]=True
    fixed=k.clone() if hard else torch.zeros_like(k)
    centers=[left+(end-left)//4,left+3*(end-left)//4]
    for center in centers:
        for joint in cfg["observation_joints"][pattern]:k[:,center-1:center+2,7+6*joint:13+6*joint]=True
    return torch.where(k,y,torch.zeros_like(y)),k,fixed,centers

def intervention(value,known,hard,donor,control,cfg,variant):
    if control=="correct":return value,known,hard
    left,right=cfg["contexts"][variant]
    v=value.clone();k=known.clone();h=hard.clone()
    if control in ("left_wrong","right_wrong"):
        sl=slice(0,left) if control=="left_wrong" else slice(225-right,225);v[:,sl,4:]=donor[:,sl,4:]
    elif control=="both_absent":
        for sl in (slice(0,left),slice(225-right,225)):v[:,sl,4:]=0;k[:,sl,4:]=False;h[:,sl,4:]=False
    else:raise ValueError("Endpoint intervention")
    return v,k,h

def runs(mask):
    pad=np.r_[False,np.asarray(mask,bool),False];d=np.diff(pad.astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def support_proxy(full_fk,cfg):
    """Full-recording derivative, floor and run anchors; never model-dependent."""
    if full_fk.ndim!=3 or full_fk.shape[1:]!=(24,3) or len(full_fk)<2 or not np.isfinite(full_fk).all():raise ValueError("Finite full-recording FK required")
    foot=full_fk[:,cfg["foot_joints"]];floor=float(np.quantile(foot[:,:,1],cfg["floor_quantile"]))
    speed=np.empty(foot.shape[:2],np.float32);speed[1:]=np.linalg.norm(np.diff(foot,axis=0)*30,axis=-1);speed[0]=speed[1]
    low=(foot[:,:,1]<=floor+cfg["static_height_m"])&(speed<=cfg["static_speed_mps"])
    high=(foot[:,:,1]>=floor+cfg["swing_height_m"])&(speed>=cfg["swing_speed_mps"])
    static=np.zeros_like(low);swing=np.zeros_like(high);anchors=np.zeros_like(foot);ids=np.full(low.shape,-1,int);counter=0
    for j in range(foot.shape[1]):
        for a,b in runs(low[:,j]):
            if b-a<cfg["static_min_frames"]:continue
            static[a:b,j]=True;anchors[a:b,j]=foot[a,j];ids[a:b,j]=counter;counter+=1
        for a,b in runs(high[:,j]):
            if b-a>=cfg["swing_min_frames"]:swing[a:b,j]=True
    return dict(proxy_static=static,proxy_swing=swing,proxy_anchor=anchors,proxy_segment=ids,proxy_floor=np.array(floor,np.float32))
