"""Shared continuous root/24-rotation decoder. Full-target latent fits are upper bounds."""
import numpy as np
from scipy.interpolate import BSpline,CubicSpline
from scipy.linalg import lstsq
from dunhuang.contracts import matrices

def basis(times,coefficients=48,degree=3):
    t=np.asarray(times,float)
    if degree!=3 or coefficients<8 or np.any(t<0) or np.any(t>1) or not np.isfinite(t).all():
        raise ValueError("Finite unit times and cubic basis required")
    knots=np.r_[np.zeros(4),np.linspace(0,1,coefficients-degree+1)[1:-1],np.ones(4)]
    return BSpline(knots,np.eye(coefficients),degree,extrapolate=False)(t)

def lift(left,right,n,times):
    a=len(left)-1;b=n-len(right);duration=b-a
    L=CubicSpline(np.arange(len(left)),left,axis=0)
    R=CubicSpline(np.arange(b,n),right,axis=0)
    c0=L(a);c1=L(a,1)*duration;c2=L(a,2)*duration**2/2
    rhs=np.stack((R(b)-c0-c1-c2,R(b,1)*duration-c1-2*c2,R(b,2)*duration**2-2*c2))
    c3,c4,c5=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]],float),rhs)
    t=np.asarray(times,float);u=np.clip((t-a)/duration,0,1)
    f=sum(u[:,None]**i*c for i,c in enumerate((c0,c1,c2,c3,c4,c5)))
    before=t<=a;after=t>=b
    f[before]=L(t[before]);f[after]=R(t[after])
    return f,u,(t>a)&(t<b)

def encode(native,coefficients=48,context=None):
    x=np.asarray(native)
    if x.shape!=(225,151) or not np.isfinite(x).all():raise ValueError("Complete finite 225x151 required")
    matrices(x[:,7:].reshape(225,24,6))
    f=x[:,4:].astype(float);t=np.arange(225)
    if context is None:
        design=basis(t/224,coefficients);target=f
        boundary=np.zeros((0,147))
    else:
        left,right=context
        if min(left,right)<4 or left+right>=225:raise ValueError("Boundary sample contract")
        boundary=np.concatenate((f[:left],f[-right:]))
        base,u,inside=lift(f[:left],f[-right:],225,t)
        design=basis(u,coefficients)[:,3:-3];design[~inside]=0
        target=f-base
    z,residue,rank,s=lstsq(design,target,lapack_driver="gelsd")
    assert rank==design.shape[1]
    return dict(coefficients=z,boundary=boundary,context=np.array(context or [0,0],int),
        basis_count=np.array(coefficients),frames=np.array(225),rank=np.array(rank),
        condition_number=np.array(s[0]/s[-1]),full_target_latent_fit=np.array(True))

def decode(state,times=None):
    t=np.arange(225,dtype=float) if times is None else np.asarray(times,float)
    if not np.isfinite(t).all() or (t<0).any() or (t>224).any():raise ValueError("Frame-time support [0,224]")
    left,right=map(int,state["context"]);count=int(state["basis_count"])
    if left:
        boundary=state["boundary"];base,u,inside=lift(boundary[:left],boundary[left:],225,t)
        design=basis(u,count)[:,3:-3];design[~inside]=0
        f=base+design@state["coefficients"]
    else:f=basis(t/224,count)@state["coefficients"]
    matrices(f[:,3:].reshape(len(t),24,6)) # Fail closed on degenerate SO(3) chart.
    out=np.zeros((len(t),151),np.float32);out[:,4:]=f.astype(np.float32)
    return out
