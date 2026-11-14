"""Distinct free, in-between and generated-prefix tasks; one 225-frame clock."""
import numpy as np

def views(x):
    if len(x)!=225:raise ValueError("Exactly225 global frames required")
    return np.stack((x[:150],x[75:]))

def metadata(n=225,boundary=112):
    if n not in (150,225) or not 0<boundary<n:raise ValueError("Clock contract")
    t=np.arange(n);phase=np.where(t<boundary,t/boundary,(t-boundary)/(n-boundary))
    return np.stack((np.sin(2*np.pi*t/225),np.cos(2*np.pi*t/225),(t>=boundary).astype(float),phase),-1).astype(np.float32)

def conditions(target,task,*,hard=True,prefix=None,boundary=112,extra=None):
    x=np.asarray(target);n=len(x)
    if x.shape!=(n,151) or n not in (150,225):raise ValueError("Motion condition shape")
    value=np.zeros_like(x);known=np.zeros_like(x,bool);fixed=np.zeros_like(known)
    for center in (boundary//2,boundary+(n-boundary)//2):
        for j in range(12,20):known[max(0,center-1):min(n,center+2),7+6*j:13+6*j]=True
    if task=="inbetween":
        known[:8,4:]=True;known[-8:,4:]=True
        if hard:fixed[:8,4:]=True;fixed[-8:,4:]=True
    elif task=="progressive":
        if prefix is None or np.shape(prefix)!=(75,151) or not np.isfinite(prefix).all():raise ValueError("Committed generated prefix required")
        known[:75,4:]=True;fixed[:75,4:]=True
    elif task!="free":raise ValueError("Task is not interchangeable")
    if extra is not None:
        for t,j in extra:known[t:t+2,7+6*j:13+6*j]=True
    value=np.where(known,x,0).astype(np.float32)
    if task=="progressive":value[:75,4:]=prefix[:,4:]
    if not np.isfinite(value[known]).all():raise ValueError("Nonfinite actual observation")
    return value,known,fixed

def factor_conditions(target,position,hard,event,event_mask):
    if np.shape(target)!=(150,151) or position not in ("mid","ends"):raise ValueError("Factor contract")
    known=np.zeros_like(target,bool);fixed=known.copy()
    for a,b in ((52,60),(90,98)) if position=="mid" else ((0,8),(142,150)):
        known[a:b,4:]=True
        if hard:fixed[a:b,4:]=True
    if event:known|=event_mask
    return np.where(known,target,0).astype(np.float32),known,fixed

def feature_distance(x,y):
    a=np.asarray(x,float).reshape(-1,4800);b=np.asarray(y,float).reshape(-1,4800)
    denom=np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)
    return dict(max_abs=float(abs(a-b).max()),relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),
        cosine_distance=float(np.mean(1-(a*b).sum(-1)/denom)))


def pair_candidates(rows,minimum_start=0):
    by={(r["recording_uid"],r["start_frame"]):r for r in rows}
    return [(l,by[l["recording_uid"],l["start_frame"]+75]) for l in rows
        if l["start_frame"]>=minimum_start and (l["recording_uid"],l["start_frame"]+75) in by]
