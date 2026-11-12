"""Banded DTW plus the existing strict partial-path implementation."""
import numpy as np

def banded_dtw(cost,allowed,mv,dv):
    import torch
    batch,n,k=cost.shape
    if n!=k or not mv.all() or not dv.all():raise ValueError("Registered complete common grid required for full DTW")
    large=1e6;row=torch.arange(n+1,device=cost.device)
    prior=torch.full((batch,n+1),large,device=cost.device,dtype=cost.dtype);prior[:,0]=0;previous=prior
    for diagonal in range(1,n+k+1):
        col=diagonal-row;ri=(row-1).clamp(0,n-1);cj=(col-1).clamp(0,k-1)
        match=prior[:,(row-1).clamp_min(0)]
        top=previous[:,(row-1).clamp_min(0)];left=previous
        current=torch.minimum(match,torch.minimum(top,left))+cost[:,ri,cj]
        good=(row>0)&(col>0)&(col<=k)&allowed[:,ri,cj]
        current=current.masked_fill(~good,large);prior,previous=previous,current
    return previous[:,n]

def operators(m,d,mv,dv,cfg,time_score=None):
    import torch
    from dunhuang.alignment.partial_path import soft_path_cost
    if m.shape!=d.shape or m.ndim!=3 or m.shape[1:]!=(120,128):raise ValueError("Frozen common 30Hz token grid required")
    if not mv.all() or not dv.all():raise ValueError("Do not silently change the registered candidate population for missing tokens")
    sim=m@d.transpose(-1,-2);time=torch.arange(120,device=m.device)/30
    distance=(time[:,None]-time[None,:]).abs()
    allowed=(distance<=cfg["band_seconds"]+1e-6)[None].expand(len(m),-1,-1)
    clock=cfg["time_weight"]*distance/cfg["band_seconds"];cost=1-sim+clock
    null=torch.full(mv.shape,cfg["null_cost"],device=m.device,dtype=m.dtype)
    full=banded_dtw(cost,allowed,mv,dv)
    partial=soft_path_cost(cost,null,null,allowed,mv,dv,cfg["epsilon"])
    time_only=soft_path_cost(clock[None],null[:1],null[:1],allowed[:1],mv[:1],dv[:1],cfg["epsilon"]) if time_score is None else None
    normalizer=mv.sum(-1).to(m.dtype)
    return dict(S0=(m*d).sum(-1).mean(-1),S1=1-full/normalizer,
        S2=(null.sum(-1)*2-partial)/normalizer,S3=((null.sum(-1)*2-time_only)/normalizer if time_score is None else time_score.expand(len(m))))
