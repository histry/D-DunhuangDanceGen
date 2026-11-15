"""Observed derivative templates and optional salient kinematic events."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from dunhuang.contracts import matrices

def features(rot,valid=None):
    rot=np.asarray(rot,np.float32).reshape(-1,22,6);n=len(rot)
    valid=np.ones((n,22),bool) if valid is None else np.asarray(valid,bool)
    if valid.shape!=(n,22):raise ValueError("Joint mask shape")
    R=matrices(rot,row=True);vel=np.zeros((n,22,3),np.float32)
    observed=np.zeros((n,22),bool);observed[1:]=valid[1:]&valid[:-1]
    values=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).as_rotvec().reshape(n-1,22,3)*30
    vel[1:]=np.where(observed[1:,:,None],values,0)
    return np.concatenate((rot,vel),-1),observed

def events(rot):
    f,v=features(rot);speed=np.linalg.norm(f[:,12:22,6:],axis=-1).mean(-1)
    observed=v[:,12:22].all(-1)
    delta=np.zeros_like(speed);delta[1:]=(speed[1:]-speed[:-1])*30
    dv=np.r_[False,observed[1:]&observed[:-1]]
    second=np.zeros_like(speed);second[1:]=np.diff(delta)*30
    dd=np.r_[False,dv[1:]&dv[:-1]]
    result=[]
    for name,values,mask,absolute in (("speed_change_peak",delta,dv,True),("speed_peak",speed,observed,False),("speed_second_difference_peak",second,dd,True),("low_speed", -speed,observed,False)):
        x=np.abs(values) if absolute else values.copy()
        indices=np.flatnonzero(mask);event=None
        if len(indices)>=5:
            span=float(np.ptp(x[indices]));prominence=max({"speed_change_peak":.1,"speed_peak":.05,"speed_second_difference_peak":3.,"low_speed":.05}[name],.1*span)
            # Unknown outside observed interior; flat signals never receive forced peaks.
            y=np.where(mask,x,-np.inf);peaks,props=find_peaks(y,prominence=prominence,distance=5)
            peaks=[int(p) for p in peaks if 1<=p<len(x)-1 and mask[p-1:p+2].all() and x[p]-np.median(x[indices])>prominence]
            if span>prominence and peaks:event=max(peaks,key=lambda p:float(x[p]))
        result.append(dict(kind=name,frame=event,status="CANDIDATE" if event is not None else "UNKNOWN",semantic_truth=False))
    return result

def view(rot,seed=None,occluded=False,drop=False):
    rot=np.asarray(rot,np.float32);assert rot.shape==(96,132)
    valid=np.ones((96,22),bool);rng=np.random.default_rng(seed)
    if occluded:
        a=int(rng.integers(0,73));valid[a:a+24]=False
        valid[:,int(rng.choice([18,19,20,21]))]=False
    if drop:valid[:]=False
    f,_=features(rot,valid)
    t=np.arange(96)/(96-1)
    clock=np.stack((np.sin(2*np.pi*t),np.cos(2*np.pi*t)),-1).astype(np.float32)
    return dict(features=f,valid=valid,clock=clock,ref_phase=np.zeros((96,8),np.float32),query_phase=np.zeros((256,8),np.float32))
