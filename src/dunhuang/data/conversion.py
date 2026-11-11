"""AIST SMPL24 and FineDance body22 -> native151, with explicit observations."""
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices,sixd,MotionAdapter

def decode(path,dataset):
    if dataset=='aistpp':
        # Callers hash-authenticate this registered research pickle first.
        with open(path,'rb') as f:data=pickle.load(f)
        poses=np.asarray(data['smpl_poses'],np.float32).reshape(-1,24,3)
        scaling=np.asarray(data['smpl_scaling'],dtype=float)
        if scaling.size!=1 or not np.isfinite(scaling).all() or scaling.item()<=0:raise ValueError('Invalid SMPL scaling')
        root=np.asarray(data['smpl_trans'],np.float32)/scaling.item()
        rotation=Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(-1,24,3,3)
        # Official EDGE 60->30 exact decimation; no interpolated extra frame.
        root=root[::2];rotation=rotation[::2];observed=24
    elif dataset=='finedance':
        x=np.load(path,allow_pickle=False)
        if x.ndim!=2 or x.shape[1]!=315 or not np.isfinite(x).all():raise ValueError('Expected FineDance T x 315')
        root=x[:,:3].astype(np.float32)
        body=matrices(x[:,3:].reshape(-1,52,6)[:,:22],row=True)
        rotation=np.broadcast_to(np.eye(3),(len(x),24,3,3)).copy();rotation[:,:22]=body;observed=22
    else:raise ValueError('Unknown dataset')
    if root.shape!=(len(rotation),3) or not np.isfinite(root).all():raise ValueError('Root clock mismatch')
    native=np.zeros((len(root),151),np.float32);native[:,4:7]=root
    native[:,7:]=sixd(rotation).reshape(-1,144)
    mask=np.zeros(151,bool);mask[4:7+6*observed]=True
    return native,mask

def contacts(edge,official_skeleton):
    """Exactly upstream EDGE forward-displacement < .01m at 30fps; final=1."""
    import torch
    r=matrices(edge[:,7:].reshape(-1,24,6),row=True)
    aa=Rotation.from_matrix(r.reshape(-1,3,3)).as_rotvec().reshape(1,-1,24,3).astype(np.float32)
    with torch.no_grad():
        fk=official_skeleton.forward(torch.from_numpy(aa),torch.from_numpy(edge[:,4:7][None]))[0].numpy()
    feet=fk[:,(7,8,10,11)];speed=np.zeros((len(edge),4),np.float32)
    speed[:-1]=np.linalg.norm(np.diff(feet,axis=0),axis=-1)
    return (speed<.01).astype(np.float32)

def normalized_window(native,start,scale,bias,recenter=True):
    x=native[start:start+150].copy()
    if len(x)!=150:raise ValueError('Full real 150-frame target required')
    x[-1,:4]=1. # Official per-window terminal displacement convention.
    origin=x[0,4:7].copy() if recenter else np.zeros(3,np.float32);origin[1]=0
    edge=MotionAdapter.to_edge(x,origin)
    normalized=edge*scale+bias
    valid=np.isfinite(normalized)&(abs(normalized)<=1+32*np.finfo(np.float32).eps)
    return x,normalized,valid
