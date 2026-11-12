"""Strict one-to-one partial sequence alignment, NOT the full many-to-one Drop-DTW.
Three transitions: match, drop music, drop motion. Every hard match is noncrossing.
The soft recurrence sums globally valid paths; its marginals may spread over paths.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def exact_path(cost, drop_music, drop_motion, allowed=None):
    cost=np.asarray(cost,float); n,k=cost.shape
    a=np.broadcast_to(np.asarray(drop_music,float),(n,))
    b=np.broadcast_to(np.asarray(drop_motion,float),(k,))
    allowed=np.ones((n,k),bool) if allowed is None else np.asarray(allowed,bool)
    if allowed.shape!=cost.shape or not np.isfinite(cost[allowed]).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite permitted costs required")
    dp=np.full((n+1,k+1),np.inf); dp[0,0]=0; back=np.zeros((n+1,k+1),np.uint8)
    for i in range(1,n+1): dp[i,0]=dp[i-1,0]+a[i-1];back[i,0]=1
    for j in range(1,k+1): dp[0,j]=dp[0,j-1]+b[j-1];back[0,j]=2
    for i in range(1,n+1):
        for j in range(1,k+1):
            c=[dp[i-1,j-1]+cost[i-1,j-1] if allowed[i-1,j-1] else np.inf,
               dp[i-1,j]+a[i-1],dp[i,j-1]+b[j-1]]
            q=int(np.argmin(c));dp[i,j]=c[q];back[i,j]=q
    i,j=n,k;matches=[];dm=[];dd=[]
    while i or j:
        q=back[i,j]
        if q==0: matches.append((i-1,j-1));i-=1;j-=1
        elif q==1: dm.append(i-1);i-=1
        else: dd.append(j-1);j-=1
    return dict(cost=float(dp[n,k]),matches=matches[::-1],music_drops=dm[::-1],motion_drops=dd[::-1])

def soft_path_cost(cost, drop_music, drop_motion, allowed, mv, dv, epsilon):
    """Anti-diagonal batched differentiable DP. Invalid tokens skip deterministically.
    Finite sentinel avoids NaN gradients from all-infinite intermediate cells.
    """
    if epsilon<=0: raise ValueError("Positive path temperature required")
    batch,n,k=cost.shape
    if allowed.shape!=cost.shape or mv.shape!=(batch,n) or dv.shape!=(batch,k):
        raise ValueError("Path shapes differ")
    if mv.dtype!=torch.bool or dv.dtype!=torch.bool or not mv.any(-1).all() or not dv.any(-1).all():
        raise ValueError("Nonempty boolean observations required")
    large=1e5
    row=torch.arange(n+1,device=cost.device); prior=torch.full((batch,n+1),large,device=cost.device,dtype=cost.dtype)
    prior[:,0]=0; previous=prior
    for diagonal in range(1,n+k+1):
        col=diagonal-row; inside=(col>=0)&(col<=k)
        ri=(row-1).clamp(0,n-1);cj=(col-1).clamp(0,k-1)
        top=previous[:,(row-1).clamp_min(0)]+drop_music[:,ri]
        left=previous+drop_motion[:,cj]
        match=prior[:,(row-1).clamp_min(0)]+cost[:,ri,cj]
        good_match=(row>0)&(col>0)&allowed[:,ri,cj]&mv[:,ri]&dv[:,cj]
        candidates=torch.stack((match.masked_fill(~good_match,large),
            top.masked_fill((row==0)|(col<0),large),
            left.masked_fill((col<=0),large)),dim=-1)
        current=-epsilon*torch.logsumexp(-candidates/epsilon,dim=-1)
        # Masked padding contributes no cost, no extra entropy, and no match.
        current=torch.where((row>0)&~mv[:,ri],previous[:,(row-1).clamp_min(0)],current)
        current=torch.where(((row==0)|mv[:,ri])&(col>0)&~dv[:,cj],previous,current)
        current=current.masked_fill(~inside,large)
        prior,previous=previous,current
    return previous[:,n]

class StrictPartialPath(nn.Module):
    def __init__(self,width=128,band_seconds=.5,epsilon=.15,iterations=80):
        super().__init__()
        self.music=nn.Linear(width,width,bias=False);self.motion=nn.Linear(width,width,bias=False)
        nn.init.eye_(self.music.weight);nn.init.eye_(self.motion.weight)
        self.null_logit=nn.Parameter(torch.tensor(0.))
        self.band_seconds=band_seconds;self.epsilon=epsilon
        if band_seconds<=0: raise ValueError("Positive physical band required")
    def costs(self,music,motion,mv,dv,mt,dt):
        if mt.ndim==1: mt=mt.expand(len(music),-1)
        if dt.ndim==1: dt=dt.expand(len(music),-1)
        if mt.shape!=mv.shape or dt.shape!=dv.shape or not torch.isfinite(mt).all() or not torch.isfinite(dt).all():
            raise ValueError("Explicit finite physical event clocks required")
        # Simultaneous typed/body-part events may have equal clocks, but never reversed clocks.
        if torch.any(torch.diff(mt,dim=-1)<0) or torch.any(torch.diff(dt,dim=-1)<0): raise ValueError("Nondecreasing clocks required")
        if not torch.isfinite(music[mv]).all() or not torch.isfinite(motion[dv]).all(): raise ValueError("Nonfinite observed event feature")
        m=F.normalize(self.music(torch.where(mv[...,None],music,0)),dim=-1)
        d=F.normalize(self.motion(torch.where(dv[...,None],motion,0)),dim=-1)
        distance=(mt[:,:,None]-dt[:,None,:]).abs()
        allowed=(distance<=self.band_seconds+1e-6)&mv[:,:,None]&dv[:,None,:]
        cost=1-m@d.transpose(-1,-2)+.2*distance/self.band_seconds
        null=.05+.9*self.null_logit.sigmoid()
        return cost,allowed,null
    def forward(self,music,motion,mv,dv,mt,dt):
        cost,allowed,null=self.costs(music,motion,mv,dv,mt,dt)
        a=null*mv; b=null*dv
        total=soft_path_cost(cost,a,b,allowed,mv,dv,self.epsilon)
        normalizer=(mv.sum(-1)+dv.sum(-1)).clamp_min(1)
        # Log-partition relative to a single all-drop reference. Not a calibrated probability.
        return dict(score=((a.sum(-1)+b.sum(-1))-total)/normalizer,
            cost=cost,allowed=allowed,music_valid=mv,motion_valid=dv,null_cost=null)
    def decode(self,result,index=0):
        a=result["music_valid"][index].detach().cpu().numpy()
        b=result["motion_valid"][index].detach().cpu().numpy()
        mi=np.flatnonzero(a);di=np.flatnonzero(b)
        cost=result["cost"][index].detach().cpu().numpy()[np.ix_(mi,di)]
        allowed=result["allowed"][index].detach().cpu().numpy()[np.ix_(mi,di)]
        r=exact_path(cost,float(result["null_cost"].detach()),float(result["null_cost"].detach()),allowed)
        r["matches"]=[(int(mi[i]),int(di[j])) for i,j in r["matches"]]
        r["music_drops"]=[int(mi[i]) for i in r["music_drops"]]
        r["motion_drops"]=[int(di[i]) for i in r["motion_drops"]]
        r["crossings"]=0
        return r
