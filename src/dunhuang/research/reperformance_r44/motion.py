"""One SO(3) source interpolant and one Root interpolant; no learned pose residual."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from dunhuang.contracts import matrices,sixd
from dunhuang.research.complete_r36.core import relative_fk

def decode(raw):return matrices(raw[:,7:].reshape(-1,22,6),row=True),np.asarray(raw[:,4:7])

def retime(raw,q):
    raw=np.asarray(raw);q=np.asarray(q,float);n=len(raw)
    if raw.shape!=(256,139) or q.shape!=(n,) or not np.isfinite(q).all() or q.min()<0 or q.max()>n-1 or np.any(np.diff(q)<=0):raise ValueError('Real monotone 256-frame source-clock support')
    if np.array_equal(q,np.arange(n)):return raw.copy()
    R,root=decode(raw);t=np.arange(n);out=raw.copy();sampled=np.stack([RotationSpline(t,Rotation.from_matrix(R[:,j]))(q).as_matrix() for j in range(22)],axis=1);out[:,7:]=sixd(sampled,row=True).reshape(n,132);out[:,4:7]=CubicSpline(t,root,axis=0)(q)
    integer=abs(q-np.rint(q))<1e-10;ids=np.rint(q[integer]).astype(int);out[integer,4:]=raw[ids,4:];assert np.array_equal(out[:,:4],raw[:,:4]);return out

def world(raw,J,parents):
    R,root=decode(raw);return relative_fk(R,J,parents)+root[:,None]+np.asarray(J)[None,:1]

def kinematics(raw,J,parents):
    R,_=decode(raw);p=relative_fk(R,J,parents);parts=[p[8:99],p[158:248]];rms=lambda x:float(np.sqrt(np.mean(x*x)));result={}
    for order in (1,2,3):result['core_D'+str(order)]=rms(np.concatenate([np.diff(x,n=order,axis=0)*30**order for x in parts]));result['gap_D'+str(order)]=rms(np.diff(p[101:155],n=order,axis=0)*30**order);result['boundary_D'+str(order)]=rms(np.concatenate([np.diff(p[b-4:b+5],n=order,axis=0)*30**order for b in (104,152)]))
    result['core_activity']=result['core_D1'];result['upper_range_m']=float(np.mean(np.ptp(np.concatenate(parts)[:,12:],axis=0)));return result
