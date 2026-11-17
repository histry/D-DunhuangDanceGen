"""Numba acceleration of the frozen strict match/drop recurrence, same tie order."""
import numpy as np
from numba import njit

@njit(cache=True)
def recurrence(sim,local_cost,allowed,mv,dv,distance,drop,band,null_penalty,warp_penalty):
    n,k=sim.shape;cost=np.full((n+1,k+1),np.inf);count=np.zeros((n+1,k+1),np.int64);sums=np.zeros((n+1,k+1));warps=np.zeros((n+1,k+1));cost[0,0]=0.
    for i in range(n):cost[i+1,0]=cost[i,0]+drop*mv[i]
    for j in range(k):cost[0,j+1]=cost[0,j]+drop*dv[j]
    for i in range(1,n+1):
        for j in range(1,k+1):
            c=cost[i-1,j-1]+local_cost[i-1,j-1] if allowed[i-1,j-1] else np.inf;up=cost[i-1,j]+drop*mv[i-1];left=cost[i,j-1]+drop*dv[j-1];ii=i-1;jj=j-1;matched=True
            if up<c:c=up;ii=i-1;jj=j;matched=False
            if left<c:c=left;ii=i;jj=j-1;matched=False
            cost[i,j]=c;count[i,j]=count[ii,jj]+matched;sums[i,j]=sums[ii,jj]+(sim[i-1,j-1] if matched else 0.);warps[i,j]=warps[ii,jj]+(distance[i-1,j-1] if matched else 0.)
    cnt=count[n,k];coverage=2*cnt/max(mv.sum()+dv.sum(),1);warp=warps[n,k]/max(cnt,1)/band;value=sums[n,k]/max(cnt,1)-null_penalty*(1-coverage)-warp_penalty*warp if cnt else -1.
    return value,coverage,cost[n,k]

def partial(query,candidate,cfg,drop=None):
    stride=cfg['token_stride'];m=query['mt'][::stride];d=candidate['dt'][::stride];mv=query['mv'][::stride];dv=candidate['dv'][::stride];ticks=query['positions'][::stride];sim=m@d.T;distance=abs(ticks[:,None]-ticks[None,:]);allowed=(distance<=cfg['path_band_frames'])&mv[:,None]&dv[None,:]
    return recurrence(sim,(1-sim).astype(float),allowed,mv,dv,distance,float(cfg['path_drop_cost'] if drop is None else drop),cfg['path_band_frames'],cfg['path_null_penalty'],cfg['path_warp_penalty'])
