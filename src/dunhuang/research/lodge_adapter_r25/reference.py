"""Source-only reference views with explicit validity, clocks and kinematic proxies."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.signal import find_peaks
from dunhuang.contracts import matrices

def features(rot):
    R=matrices(rot.reshape(-1,22,6),row=True);n=len(R)
    velocity=np.zeros((n,22,3),np.float32)
    velocity[1:]=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).as_rotvec().reshape(n-1,22,3)*30
    return np.concatenate((rot.reshape(n,22,6),velocity),-1).astype(np.float32)

def phase_features(n=96,positions=None,landmarks=None,core=(64,160),frames=256):
    positions=np.arange(n,dtype=float) if positions is None else np.asarray(positions,float)
    if positions.shape!=(n,) or (np.diff(positions)<=0).any():raise ValueError("Monotone observed reference clock required")
    a,b=core
    def build(pos):
        tau=(a+pos)/255;out=np.stack((tau,np.sin(2*np.pi*tau),np.cos(2*np.pi*tau),np.ones_like(tau)),1)
        bumps=np.zeros((len(pos),4))
        for channel,p in enumerate(landmarks if landmarks is not None else [0,n//3,2*n//3,n-1]):
            bumps[:,channel]=np.exp(-.5*((pos-p)/4.)**2)
        return np.concatenate((out,bumps),1).astype(np.float32)
    rp=build(positions);qp=np.zeros((frames,8),np.float32)
    qp[:,0]=np.arange(frames)/255;qp[:,1]=np.sin(2*np.pi*qp[:,0]);qp[:,2]=np.cos(2*np.pi*qp[:,0])
    qp[a:b]=build(np.arange(b-a))
    return rp,qp

def landmarks(rot):
    x=features(rot);speed=np.linalg.norm(x[:,12:22,6:],axis=-1).mean(-1)
    accel=np.abs(np.diff(speed,prepend=speed[0]));return [int(np.argmax(accel)),int(np.argmax(speed)),int(np.argmax(np.abs(np.diff(speed,n=2,prepend=speed[:2])))),int(np.argmin(speed[1:]))+1]

def view(rot,seed=None,augment=False,drop=False,phase_control="correct"):
    rot=np.asarray(rot,np.float32);n=len(rot);assert rot.shape==(96,132)
    pos=np.arange(n,dtype=float);rng=np.random.default_rng(seed)
    if augment:
        knot=np.array([0,.25,.5,.75,1]);shift=np.r_[0,rng.uniform(-.025,.025,3),0]
        pos=np.interp(np.linspace(0,1,n),knot,knot+shift)*(n-1)
        R=matrices(rot.reshape(n,22,6),row=True)
        sample=np.stack([Slerp(np.arange(n),Rotation.from_matrix(R[:,j]))(pos).as_matrix() for j in range(22)],1)
        rot=sample[...,:2,:].reshape(n,132).astype(np.float32)
    f=features(rot);valid=np.ones((n,22),bool)
    if augment:
        begin=int(rng.integers(0,n-24+1));valid[begin:begin+24]=False
        joint=int(rng.choice([18,19,20,21]));valid[:,joint]=False
    if drop:valid[:]=False
    f[1:,:,6:]=np.where((valid[1:]&valid[:-1])[:,:,None],f[1:,:,6:],0)
    f[0,:,6:]=0
    rp,qp=phase_features(n,pos,pos[np.array(landmarks(rot))])
    if phase_control=="removed":rp[:]=0;qp[:]=0
    elif phase_control=="wrong":rp[:,4:]=np.roll(rp[:,4:],n//3,axis=0)
    elif phase_control!="correct":raise ValueError("Unknown phase intervention")
    clock=np.stack((np.sin(2*np.pi*np.arange(n)/(n-1)),np.cos(2*np.pi*np.arange(n)/(n-1))),1).astype(np.float32)
    return dict(features=f,valid=valid,clock=clock,ref_phase=rp,query_phase=qp)
