"""Pure masked scoring. No clocks/labels or hidden target activity enter content scores."""
import math
import numpy as np


class R41EventScorer:
    """Versioned research service: normalized50/274 -> same checkpoint global/local tokens."""
    def __init__(self,model,cfg,normalizer):self.model=model;self.cfg=cfg;self.normalizer=normalizer;self.can_route=False
    def encode_normalized(self,inputs):
        music,motion,mask,valid=inputs
        if music.shape[-1]!=50 or motion.shape[-1]!=274 or motion.shape!=mask.shape or music.shape[:2]!=motion.shape[:2]:raise ValueError('R41 50D/274D common-clock contract; not the legacy74D service')
        mg,dg,*_=self.model.encode(*inputs);mt,dt,mv,dv=self.model.content(*inputs)
        return mg,dg,mt,dt,mv,dv


def masked_diagonal(m,d,mv,dv,positions,offset=0,stride=3):
    positions=np.asarray(positions);other={int(p):i for i,p in enumerate(positions)};pairs=[(i,other[int(p+offset)]) for i,p in enumerate(positions) if i%stride==0 and int(p+offset) in other and mv[i] and dv[other[int(p+offset)]]]
    if not pairs:return -1.,0.
    a,b=np.array(pairs).T;score=float(np.mean(np.sum(m[a]*d[b],axis=-1)));coverage=len(a)/max(int(mv[::stride].sum()),1)
    return score,float(coverage)


def partial_score(m,d,mv,dv,positions,cfg):
    """Reuse the strict one-to-one path. No unrestricted DTW or semantic labels."""
    from dunhuang.alignment.partial_path import exact_path
    mi=np.flatnonzero(mv);di=np.flatnonzero(dv)
    if not len(mi) or not len(di):return -1.,0.,dict(matches=[],music_drops=mi.tolist(),motion_drops=di.tolist())
    sim=m[mi]@d[di].T;positions=np.asarray(positions);distance=abs(positions[mi,None]-positions[None,di]);allowed=distance<=cfg['path_band_frames']
    path=exact_path(1-sim,cfg['path_drop_cost'],cfg['path_drop_cost'],allowed);pairs=path['matches'];coverage=2*len(pairs)/(len(mi)+len(di))
    if pairs:
        a,b=np.array(pairs).T;warp=float(np.mean(distance[a,b])/cfg['path_band_frames']);value=float(sim[a,b].mean())-cfg['path_null_penalty']*(1-coverage)-cfg['path_warp_penalty']*warp
    else:value=-1.;warp=0.
    path['matches']=[[int(mi[i]),int(di[j])] for i,j in pairs];path['coverage']=coverage;path['null_fraction']=1-coverage;path['warp_cost']=warp;path['scope']='STRICT_ONE_TO_ONE_PARTIAL_PATH_NOT_FULL_DROP_DTW'
    return value,float(coverage),path


def scores(query,candidate,cfg,include_partial=True):
    m,d=query['mt'],candidate['dt'];mv,dv=query['mv'],candidate['dv'];positions=query['positions']
    if not np.array_equal(positions,candidate['positions']):raise ValueError('Registered common target clock')
    diagonal,cov=masked_diagonal(m,d,mv,dv,positions,stride=cfg['token_stride'])
    offsets=[masked_diagonal(m,d,mv,dv,positions,k,cfg['token_stride']) for k in cfg['offsets']];best=max(range(len(offsets)),key=lambda i:(offsets[i][0],-abs(cfg['offsets'][i]),-i));partial,pcov,path=partial_score(m[::cfg['token_stride']],d[::cfg['token_stride']],mv[::cfg['token_stride']],dv[::cfg['token_stride']],positions[::cfg['token_stride']],cfg) if include_partial else (None,None,None)
    return dict(Global=float(query['mg']@candidate['dg']),Diagonal=diagonal,Offset=float(offsets[best][0]),Partial=partial,diagonal_coverage=cov,offset_coverage=offsets[best][1],selected_offset=cfg['offsets'][best],partial_coverage=pcov),path


def retrieval_metrics(scores,positive):
    scores=np.asarray(scores,float);positive=np.asarray(positive,bool)
    if scores.ndim!=1 or scores.shape!=positive.shape or not positive.any() or positive.all() or not np.isfinite(scores).all():raise ValueError('Finite trusted positive/negative competition')
    order=np.argsort(-scores,kind='stable');above=0
    while above<len(order):
        tail=order[above:];group=tail[np.isclose(scores[tail],scores[order[above]],atol=1e-7,rtol=0)]
        count=int(positive[group].sum());n=len(group)
        if count:
            prob=[math.comb(n-k,count-1)/math.comb(n,count) for k in range(1,n-count+2)]
            return dict(recall_at1=float(sum(p for k,p in enumerate(prob,1) if above+k<=1)),recall_at5=float(sum(p for k,p in enumerate(prob,1) if above+k<=5)),mrr=float(sum(p/(above+k) for k,p in enumerate(prob,1))),uniform_candidate_chance=float(positive.mean()),candidates=len(scores),positives=int(positive.sum()))
        above+=n
    raise AssertionError('Positive rank absent')


def batch_partial(query,candidates,cfg):
    """Exact match/drop recurrence vectorized over candidates; same tie order as exact_path."""
    stride=cfg['token_stride'];m=query['mt'][::stride];d=np.stack([c['dt'][::stride] for c in candidates]);mv=query['mv'][::stride];dv=np.stack([c['dv'][::stride] for c in candidates]);ticks=query['positions'][::stride];n=len(m);batch=len(d)
    sim=m[None]@d.transpose(0,2,1);local_cost=(1-sim).astype(float);distance=abs(ticks[:,None]-ticks[None]);allowed=(distance<=cfg['path_band_frames'])[None]&mv[None,:,None]&dv[:,None,:]
    costs=np.full((batch,n+1,n+1),np.inf);counts=np.zeros_like(costs,dtype=np.int32);sums=np.zeros_like(costs);warps=np.zeros_like(costs);costs[:,0,0]=0;drop=cfg['path_drop_cost'];costs[:,1:,0]=np.cumsum(np.broadcast_to(drop*mv,(batch,n)),axis=1);costs[:,0,1:]=np.cumsum(drop*dv,axis=1);b=np.arange(batch)
    for i in range(1,n+1):
        for j in range(1,n+1):
            v=np.stack([np.where(allowed[:,i-1,j-1],costs[:,i-1,j-1]+local_cost[:,i-1,j-1],np.inf),costs[:,i-1,j]+drop*mv[i-1],costs[:,i,j-1]+drop*dv[:,j-1]],axis=1);q=v.argmin(1);ii=np.where(q==2,i,i-1);jj=np.where(q==1,j,j-1);matched=q==0;costs[:,i,j]=v[b,q];counts[:,i,j]=counts[b,ii,jj]+matched;sums[:,i,j]=sums[b,ii,jj]+matched*sim[:,i-1,j-1];warps[:,i,j]=warps[b,ii,jj]+matched*distance[i-1,j-1]
    count=counts[:,-1,-1];coverage=2*count/np.maximum(mv.sum()+dv.sum(1),1);warp=warps[:,-1,-1]/np.maximum(count,1)/cfg['path_band_frames'];score=np.where(count>0,sums[:,-1,-1]/np.maximum(count,1)-cfg['path_null_penalty']*(1-coverage)-cfg['path_warp_penalty']*warp,-1.)
    return score,coverage,costs[:,-1,-1]
