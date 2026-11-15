"""Explicit reference geometry: remove the nonphysical 6D gauge, never alter targets."""
import numpy as np
from dunhuang.contracts import matrices
from dunhuang.research.reference_r26.common import packet as original_packet

def rotations(x,mode="canonical"):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=132 or not np.isfinite(x).all():raise ValueError("Finite Tx132 reference required")
    vectors=x.reshape(-1,22,2,3).astype(np.float64)
    a=vectors[:,:,0];b=vectors[:,:,1]
    norm=np.linalg.norm(a,axis=-1,keepdims=True)
    if (norm<1e-8).any():raise ValueError("Degenerate first rotation vector")
    unit=a/norm;other=b-(unit*b).sum(-1,keepdims=True)*unit
    if (np.linalg.norm(other,axis=-1)<1e-8).any():raise ValueError("Degenerate rotation plane")
    R=matrices(x.reshape(-1,22,6),row=True)
    if mode=="raw":return x.copy()
    if mode=="canonical":return R[...,:2,:].reshape(-1,132).astype(np.float32)
    if mode=="equivalent":
        v1=1.5*R[...,0,:];v2=.7*R[...,1,:]+.4*R[...,0,:]
        return np.stack((v1,v2),-2).reshape(-1,132).astype(np.float32)
    raise ValueError("Unknown reference representation")

def packet(rows,mode="canonical",drops=None):
    return original_packet([rotations(x,mode) for x in rows],drops=drops)
