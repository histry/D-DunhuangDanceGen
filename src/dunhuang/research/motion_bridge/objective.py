"""Raw x0 objectives. No hard/fused ground-truth replacement before losses."""
import torch
from .model import rotations

def terms(pred,target,known,scale,bias,a=60,b=90):
    p=(pred-bias)/scale;g=(target-bias)/scale
    P=rotations(p[:,:,7:].reshape(-1,150,24,6));G=rotations(g[:,:,7:].reshape(-1,150,24,6))
    rootmask=known[:,:,4:7]
    pm=known[:,:,7:].reshape(-1,150,24,6).all(-1)
    def masked(x,m):
        return x[m].mean() if m.any() else x.sum()*0
    # Chordal SO(3) distance: ||R-G||_F^2/2 = 4 sin^2(theta/2).
    pose=(P-G).square().sum((-1,-2))*.5
    v=torch.diff(p[:,:,4:7],dim=1)*30;gv=torch.diff(g[:,:,4:7],dim=1)*30
    acc=torch.diff(v,dim=1)*30;ga=torch.diff(gv,dim=1)*30
    dr=P[:,:-1].transpose(-1,-2)@P[:,1:];dg=G[:,:-1].transpose(-1,-2)@G[:,1:]
    idx=sorted(set(range(a-3,a+2))|set(range(b-3,b+2)))
    return dict(reconstruction=(pred[:,a:b,4:]-target[:,a:b,4:]).square().mean(),
        anchor_root=masked(((p[:,:,4:7]-g[:,:,4:7])/.1).square(),rootmask),
        anchor_pose=masked(pose/.3**2,pm),
        seam_velocity=(v[:,idx]-gv[:,idx]).square().mean(),
        seam_acceleration=((acc[:,idx]-ga[:,idx])/10).square().mean(),
        seam_rotation=((dr[:,idx]-dg[:,idx])*30/3).square().mean())

def total(parts,weights):return sum(parts[k]*weights[k] for k in parts)

def capacity_objective(pred,target,known,scale,bias):
    # Only observed data enters direction construction. No unknown target leakage.
    parts=terms(pred,target,known,scale,bias)
    return parts["anchor_root"]+parts["anchor_pose"]
