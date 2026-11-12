"""Frozen-clock interventions and content scoring on a clock-only fixed path.
An optimised-path score difference is not an additive content contribution.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .partial_path import exact_path,soft_path_cost
from .event_metrics import proxy_retrieval
from .clock_metrics import grouped

VARIANTS=("full","token","descriptor","clock","token_shuffle","full_shuffle")
def intervention(x,valid,variant,seed):
    y=torch.where(valid[...,None],x,0).clone()
    if variant=="token":y[...,128:]=0
    elif variant=="descriptor":y[...,:128]=0
    elif variant=="clock":y.zero_()
    elif variant in ("token_shuffle","full_shuffle"):
        # Per-source deterministic permutation, independent of candidate order.
        rng=np.random.default_rng(seed)
        for i in range(len(y)):
            ids=torch.nonzero(valid[i]).flatten();order=torch.as_tensor(rng.permutation(len(ids)),device=x.device)
            channels=128 if variant=="token_shuffle" else y.shape[-1]
            y[i,ids,:channels]=y[i,ids[order],:channels]
    elif variant!="full":raise ValueError("Unknown content intervention")
    return y

def pair_result(head,bank,queries,candidates,variant="full",normalization=None):
    device=next(head.parameters()).device
    def get(name,ids):return torch.as_tensor(bank[name][ids],device=device)
    m=get("music",queries);d=get("motion",candidates);mv=get("mv",queries);dv=get("dv",candidates)
    if normalization is not None:
        m=(m-torch.as_tensor(normalization["music"]["mean"],device=device,dtype=m.dtype))@torch.as_tensor(normalization["music"]["matrix"],device=device,dtype=m.dtype)
        d=(d-torch.as_tensor(normalization["motion"]["mean"],device=device,dtype=d.dtype))@torch.as_tensor(normalization["motion"]["matrix"],device=device,dtype=d.dtype)
    # Source identity enters only the deterministic permutation seed, not the scorer.
    for i,(q,c) in enumerate(zip(queries,candidates)):
        m[i:i+1]=intervention(m[i:i+1],mv[i:i+1],variant,7001+int(q))
        d[i:i+1]=intervention(d[i:i+1],dv[i:i+1],variant,9001+int(c))
    mt=get("mt",queries);dt=get("dt",candidates)
    out=head(m,d,mv,dv,mt,dt)
    pm=F.normalize(head.music(torch.where(mv[...,None],m,0)),dim=-1)
    pd=F.normalize(head.motion(torch.where(dv[...,None],d,0)),dim=-1)
    out["similarity"]=pm@pd.transpose(-1,-2)
    out["distance"]=(mt[:,:,None]-dt[:,None,:]).abs()
    return out

def template_paths(bank,queries,candidates,band,null):
    paths=np.zeros((len(queries),bank["mv"].shape[1],bank["dv"].shape[1]),bool)
    for b,(q,c) in enumerate(zip(queries,candidates)):
        mv=bank["mv"][q];dv=bank["dv"][c];mi=np.flatnonzero(mv);di=np.flatnonzero(dv)
        distance=abs(bank["mt"][q][mv][:,None]-bank["dt"][c][dv][None,:])
        p=exact_path(.2*distance/band,null,null,distance<=band+1e-6)
        for i,j in p["matches"]:paths[b,mi[i],di[j]]=True
    return paths

def fixed_score(result,paths,band):
    path=torch.as_tensor(paths,device=result["score"].device,dtype=result["cost"].dtype)
    if torch.any(path.bool()&~result["allowed"]):raise ValueError("Fixed template uses forbidden edge")
    edges=path.sum((-1,-2));total=result["music_valid"].sum(-1)+result["motion_valid"].sum(-1)
    content=(path*result["similarity"]).sum((-1,-2))/total
    time=-(path*.2*result["distance"]/band).sum((-1,-2))/total
    constant=edges*(2*result["null_cost"]-1)/total
    return dict(score=content+time+constant,content=content,time=time,constant=constant,
        mean_content=(path*result["similarity"]).sum((-1,-2))/edges.clamp_min(1),edges=edges)

def query_auc(scores,positive,valid):
    values=[]
    for s,p,v in zip(scores,positive,valid):
        a=s[p];b=s[v&~p]
        values.append(float(((a[:,None]>b[None,:]+1e-7)+.5*(abs(a[:,None]-b[None,:])<=1e-7)).mean()))
    return np.asarray(values)

def metrics(scores,positive,valid,rows,sufficient=True):
    if len(scores)==0:return dict(status="NOT_EVALUABLE_EMPTY_POPULATION",queries=0,R1=None,AUC=None)
    out=proxy_retrieval(scores,positive,valid);a=query_auc(scores,positive,valid)
    out.update(AUC=float(a.mean()),AUC_grouped=grouped(a,rows),population_sufficient=sufficient,
        semantic_labels_available=False,fresh_blind=False)
    if not sufficient:out["status"]="DESCRIPTIVE_ONLY_INSUFFICIENT_CONDITIONAL_POPULATION"
    return out

def fit_normalization(bank,rows,kind,shrinkage=.05):
    if kind not in ("zscore","whiten"):raise ValueError("Unknown train-only normalization")
    train=np.array([r["role"]=="train" for r in rows]);result={}
    for name,mask in (("music","mv"),("motion","dv")):
        x=bank[name][train][bank[mask][train]].astype(np.float64)
        mean=x.mean(0);centered=x-mean;cov=centered.T@centered/max(len(x)-1,1)
        if kind=="zscore":matrix=np.diag(1/np.maximum(np.sqrt(np.diag(cov)),1e-4))
        else:
            scale=max(np.trace(cov)/len(mean),1e-8)
            eigen,v=np.linalg.eigh((1-shrinkage)*cov+shrinkage*scale*np.eye(len(mean)))
            matrix=(v*(1/np.sqrt(np.maximum(eigen,scale*1e-4))))@v.T
        result[name]=dict(mean=mean,matrix=matrix,samples=len(x))
    return result
