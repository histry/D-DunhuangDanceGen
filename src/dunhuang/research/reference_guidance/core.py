"""Reference visibility, protected-motion readouts and Train-only linear probes."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices

def reference(value,known,keys,radius=2,phase_weight=3.):
    value=np.asarray(value);known=np.asarray(known);keys=np.asarray(keys)
    if value.shape!=(150,151) or known.shape!=value.shape or known.dtype!=bool or keys.shape!=(150,8) or keys.dtype!=bool:raise ValueError("Reference clock/support differs")
    visible=known[:,79:127].reshape(150,8,6).all(-1)
    if not visible.any() or not (keys&visible).any():raise ValueError("No actually visible source keys")
    src=value[:,79:127].reshape(150,8,6)
    if not np.isfinite(src[visible]).all():raise ValueError("Nonfinite provided reference")
    safe=np.where(visible[...,None],src,np.array([1,0,0,0,1,0],src.dtype))
    stage=np.zeros_like(visible)
    for t,j in zip(*np.where(keys&visible)):stage[max(0,t-radius):min(150,t+radius+1),j]=True
    stage&=visible
    weights=np.where(stage,phase_weight,1.)*visible
    return safe,visible,stage,weights

def phase_reference(value,known,keys,phase_cfg,radius=2,phase_weight=3.):
    from dunhuang.data.change_event_phases import motion_profiles,detect
    safe,visible,_,_=reference(value,known,keys,radius,phase_weight)
    stage=np.zeros_like(visible);whole=visible.all(-1)
    edges=np.diff(np.r_[False,whole,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a<2*radius+3:continue
        x=np.zeros((b-a,151),dtype=safe.dtype)
        x[:,7:]=np.tile([1,0,0,0,1,0],(b-a,24))
        x[:,79:127]=safe[a:b].reshape(b-a,48)
        events=detect(motion_profiles(x,list(range(12,20))),phase_cfg)
        for event in events:
            t=event["frame"]
            # Never differentiate across a visibility gap or use its artificial edge.
            if radius<t<b-a-radius-1:
                lo=max(a,a+t-radius);hi=min(b,a+t+radius+1)
                stage[lo:hi]=True
    stage&=visible
    weights=np.where(stage,phase_weight,1.)*visible
    return safe,visible,stage,weights

def protection(base,output,cfg):
    br=matrices(base[:,7:].reshape(150,24,6));rr=matrices(output[:,7:].reshape(150,24,6))
    delta=Rotation.from_matrix((br.swapaxes(-1,-2)@rr).reshape(-1,3,3)).magnitude().reshape(150,24)
    other=[j for j in range(24) if not 12<=j<20]
    def dynamics(r):
        omega=Rotation.from_matrix((r[:-1,12:20].swapaxes(-1,-2)@r[1:,12:20]).reshape(-1,3,3)).as_rotvec().reshape(149,8,3)*30
        return [float(np.linalg.norm(omega,axis=-1).mean()),float(np.quantile(np.linalg.norm(np.diff(omega,axis=0)*30,axis=-1),.95)),float(np.quantile(np.linalg.norm(np.diff(omega,n=2,axis=0)*900,axis=-1),.95))]
    b=dynamics(br);o=dynamics(rr);ratio=[o[i]/max(b[i],1e-9) for i in range(3)]
    measures=dict(non_target_mean_rad=float(delta[:,other].mean()),root_rms_m=float(np.sqrt(np.square(output[:,4:7]-base[:,4:7]).mean())),contact_max=float(abs(output[:,:4]-base[:,:4]).max()),activity_ratio=ratio[0],acceleration_proxy_ratio=ratio[1],jerk_proxy_ratio=ratio[2])
    ok=measures["non_target_mean_rad"]<=cfg["non_target_mean_rad"] and measures["root_rms_m"]<=cfg["root_rms_m"] and measures["contact_max"]<=cfg["contact_max"] and cfg["activity_ratio"][0]<=ratio[0]<=cfg["activity_ratio"][1] and ratio[1]<=cfg["acceleration_ratio"] and ratio[2]<=cfg["jerk_ratio"]
    return dict(**measures,protected=bool(ok),proxy_is_not_learned_realism=True)

def ridge_fit(x,y,groups,penalty):
    x=np.asarray(x,float);y=np.asarray(y,float);groups=np.asarray(groups)
    if x.ndim!=2 or len(x)!=len(y) or len(y)!=len(groups) or not np.isfinite(x).all() or set(np.unique(y))!={0.,1.}:raise ValueError("Finite binary Train population required")
    # Equal source-group weight, then balanced classes; both derived from Train.
    w=np.array([1/np.sum(groups==g) for g in groups])
    w=np.where(y==1,w/w[y==1].sum(),w/w[y==0].sum())*.5
    mean=(x*w[:,None]).sum(0);std=np.sqrt(((x-mean)**2*w[:,None]).sum(0)).clip(1e-6)
    z=np.c_[(x-mean)/std,np.ones(len(x))]
    reg=np.eye(z.shape[1])*penalty;reg[-1,-1]=0
    beta=np.linalg.solve(z.T@(z*w[:,None])+reg,z.T@(w*y))
    return dict(mean=mean,std=std,beta=beta)

def ridge_predict(fit,x):
    return np.clip(np.c_[(x-fit["mean"])/fit["std"],np.ones(len(x))]@fit["beta"],0,1)

def ap(y,score):
    y=np.asarray(y,bool);score=np.asarray(score)
    if not y.any() or y.all():return None
    # Group equal scores to avoid optimistic, order-dependent ties.
    order=np.argsort(-score,kind="stable");truth=y[order];ranked=score[order]
    ends=np.r_[np.flatnonzero(ranked[1:]!=ranked[:-1]),len(y)-1]
    tp=np.cumsum(truth)[ends];count=ends+1
    return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/count))

def group_scores(y,score,groups,threshold):
    result={}
    for g in sorted(set(groups)):
        take=groups==g;a=ap(y[take],score[take])
        if a is None:continue
        pred=score[take]>=threshold;truth=y[take].astype(bool)
        f1=2*np.sum(pred&truth)/max(int(pred.sum()+truth.sum()),1)
        result[str(g)]=[a,float(f1)]
    return result

def canonical(rows,modality,cap):
    # One raw-source frame across overlapping contexts; music also across recordings.
    assets={}
    for i,r in enumerate(rows):
        if r["role"] not in ("train","val"):raise ValueError("Test tensors forbidden")
        key=r["audio"]["sha256"] if modality=="music" else r["raw_motion"]["sha256"]
        offset=30*r["audio_offset_seconds"] if modality=="music" else 0.
        if abs(offset-round(offset))>1e-6:raise ValueError("Fractional clock needs explicit label interpolation")
        bank=assets.setdefault(key,{})
        for t in range(120):
            frame=r["start_frame"]+15+t+round(offset)
            candidate=(abs(t-59.5),r["window_uid"],i,t)
            if frame not in bank or candidate[:2]<bank[frame][:2]:bank[frame]=candidate
    selected=[]
    for key,bank in sorted(assets.items()):
        frames=sorted(bank)
        # Fixed uniform coverage; never use labels or prediction scores for selection.
        for pos in np.unique(np.linspace(0,len(frames)-1,min(cap,len(frames))).round().astype(int)):
            frame=frames[pos];_,_,i,t=bank[frame]
            selected.append((key,frame,i,t))
    return selected
