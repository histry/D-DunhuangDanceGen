"""Controlled within-modality event-copy tasks with exact identity, NULL and warp truth.
These do not create cross-modal semantic labels. All sequences have equal observed length.
"""
import numpy as np
import torch
from .partial_path import soft_path_cost

def construct(tokens,donor,seed,n=12):
    tokens=np.asarray(tokens);donor=np.asarray(donor)
    if tokens.ndim!=2 or len(tokens)<n or donor.shape!=tokens.shape:raise ValueError("Equal full observed token grids required")
    rng=np.random.default_rng(seed);ids=np.sort(rng.choice(len(tokens),n,replace=False))
    times=ids/15; x=tokens[ids].copy()
    removed=set(map(int,rng.choice(n,2,replace=False)));keep=[i for i in range(n) if i not in removed]
    pairs=[(float(times[i]),i,x[i].copy()) for i in keep]
    for j in range(2):
        t=float(rng.uniform(0,(len(tokens)-1)/15));v=donor[int(rng.integers(len(donor)))].copy()
        pairs.append((t,-1,v))
    pairs.sort(key=lambda r:r[0])
    # Legal piecewise monotone map on [0,4], slopes .9/1.2/.9 within [.75,1.33].
    knots=np.array([[0.,0.],[4/3,1.2],[8/3,2.8],[4.,4.]])
    warped=np.interp([p[0] for p in pairs],knots[:,0],knots[:,1])
    gt=np.full(n,-1,np.int64)
    for j,(_,i,_) in enumerate(pairs):
        if i>=0:gt[i]=j
    return dict(music=x,motion=np.stack([p[2] for p in pairs]),mt=times.astype(np.float32),
        dt=warped.astype(np.float32),target=gt,source_token_indices=ids,warp=knots,
        inserted_indices=np.array([j for j,p in enumerate(pairs) if p[1]<0],np.int64),
        truth_scope="SYNTHETIC_TRUTH_WITHIN_MODALITY_IDENTITY_NOT_CROSSMODAL")

def batch(tokens,ids,donors,seed):
    samples=[construct(tokens[i],tokens[j],seed+int(i)*1009+k) for k,(i,j) in enumerate(zip(ids,donors))]
    return {k:np.stack([s[k] for s in samples]) for k in ("music","motion","mt","dt","target")}

def supervised_path_loss(head,sample,device):
    x={k:torch.as_tensor(v,device=device) for k,v in sample.items()}
    valid=torch.ones(x["mt"].shape,device=device,dtype=torch.bool)
    result=head(x["music"],x["motion"],valid,valid,x["mt"],x["dt"])
    cost=result["cost"];null=result["null_cost"];allowed=result["allowed"]
    target=x["target"];positive=target>=0
    if not allowed.gather(-1,target.clamp_min(0)[...,None]).squeeze(-1)[positive].all():
        raise ValueError("Synthetic identity outside registered physical clock band")
    matched=cost.gather(-1,target.clamp_min(0)[...,None]).squeeze(-1)*positive
    true_cost=matched.sum(-1)+2*(~positive).sum(-1)*null
    a=null*valid;b=null*valid
    partition=soft_path_cost(cost,a,b,allowed,valid,valid,head.epsilon)
    partial=((true_cost-partition)/(valid.sum(-1)*2)).mean()
    row_real=torch.logsumexp((-cost/head.epsilon).masked_fill(~allowed,-1e4),-1)
    row_null=(-2*null/head.epsilon).expand_as(row_real)
    row_logits=torch.stack((row_real,row_null),-1)
    col_real=torch.logsumexp((-cost/head.epsilon).masked_fill(~allowed,-1e4),-2)
    col_null=(-2*null/head.epsilon).expand_as(col_real)
    col_logits=torch.stack((col_real,col_null),-1)
    inserted=torch.ones_like(positive)
    for i in range(len(target)):inserted[i,target[i][positive[i]]]=False
    null_loss=.5*(torch.nn.functional.cross_entropy(row_logits.flatten(0,1),(~positive).long().flatten())+
        torch.nn.functional.cross_entropy(col_logits.flatten(0,1),inserted.long().flatten()))
    return dict(partial=partial,null=null_loss),result

def metrics(head,samples,mode):
    from .event_study import score
    errors=[];correct=[];nt=[];npred=[];all_conf=[];all_good=[];crossings=[]
    with torch.no_grad():
        for s in samples:
            device=next(head.parameters()).device
            bank={k:torch.as_tensor(s[k][None],device=device) for k in ("music","motion","mt","dt")}
            bank["mv"]=torch.ones(bank["mt"].shape,device=device,dtype=torch.bool);bank["dv"]=torch.ones(bank["dt"].shape,device=device,dtype=torch.bool)
            if mode=="null_ot":
                r=head(bank["music"],bank["motion"],bank["mv"],bank["dv"],bank["mt"],bank["dt"])
                p=r["solver_plan"][0].cpu().numpy();prediction=p[:-1].argmax(-1);prediction[prediction==p.shape[1]-1]=-1
                pairs=[(i,int(j)) for i,j in enumerate(prediction) if j>=0]
                crossings.append(sum(j>q for a,(i,j) in enumerate(pairs) for _,q in pairs[a+1:]))
                confidence=p[:-1].max(-1)/p[:-1].sum(-1)
            else:
                r=head(bank["music"],bank["motion"],bank["mv"],bank["dv"],bank["mt"],bank["dt"])
                decoded=head.decode(r);prediction=np.full(len(s["target"]),-1,np.int64)
                for i,j in decoded["matches"]:prediction[i]=j
                crossings.append(decoded["crossings"])
                c=r["cost"][0].cpu().numpy();confidence=1/(1+np.exp(np.clip(c.min(-1)-2*float(r["null_cost"]),-60,60)))
                confidence=np.where(prediction<0,1-confidence,confidence)
            for i,(t,p) in enumerate(zip(s["target"],prediction)):
                nt.append(t<0);npred.append(p<0);good=int(t)==int(p);correct.append(good)
                all_conf.append(float(confidence[i]));all_good.append(good)
                if t>=0 and p>=0:errors.append(abs(float(s["dt"][int(t)]-s["dt"][int(p)])))
    true=np.asarray(nt);pred=np.asarray(npred);tp=int((true&pred).sum())
    precision=tp/int(pred.sum()) if pred.sum() else None;recall=tp/int(true.sum()) if true.sum() else None
    risk=[]
    for f in (.2,.4,.6,.8,1.):
        q=np.sort(all_conf)[max(0,len(all_conf)-int(np.ceil(f*len(all_conf))))]
        keep=np.asarray(all_conf)>=q;risk.append(dict(requested_coverage=f,actual_coverage=float(keep.mean()),risk=float(1-np.asarray(all_good)[keep].mean())))
    return dict(scope="SYNTHETIC_TRUTH within-modality copy/insert/delete/legal-warp; not human cross-modal event truth",
        sequences=len(samples),correspondence_accuracy=float(np.mean(correct)),warp_error_seconds=float(np.mean(errors)) if errors else None,
        warp_error_population="accepted ground-truth non-NULL events; NULL misses count in correspondence accuracy",
        NULL_precision=precision,NULL_recall=recall,NULL_F1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None,
        hard_path_crossings=int(sum(crossings)),risk_coverage=risk,confidence_calibrated=False)
