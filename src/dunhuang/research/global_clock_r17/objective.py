"""225-frame reconstruction, real observed state, and paired world-FK dynamics."""
import torch
from dunhuang.research.motion_bridge.model import rotations
from .model import fk

def losses(pred,target,value,known,hard,scale,bias,origin,parents,offsets):
    raw=(pred-bias)/scale;gt=(target-bias)/scale
    reconstruction=(pred[:,:,4:]-target[:,:,4:]).square().mean()
    candidate=torch.where(hard,value,pred)
    p=(candidate-bias)/scale
    boundary=pred.sum()*0
    if known[:,:,4:7].any():boundary=boundary+(((raw[:,:,4:7]-gt[:,:,4:7])/.1).square()[known[:,:,4:7]]).mean()
    for n,unit in ((1,3.),(2,30.)):
        boundary=boundary+((torch.diff(p[:,:,4:7],n=n,dim=1)-torch.diff(gt[:,:,4:7],n=n,dim=1))*30**n/unit).square().mean()
    P=rotations(raw[:,:,7:].reshape(-1,225,24,6));G=rotations(gt[:,:,7:].reshape(-1,225,24,6))
    rk=known[:,:,7:].reshape(-1,225,24,6).all(-1)
    if rk.any():boundary=boundary+(P-G).square().sum((-1,-2))[rk].mean()/.3**2
    points=fk(candidate,scale,bias,origin,parents,offsets);truth=fk(target,scale,bias,origin,parents,offsets)
    feet=[7,8,10,11]
    position=((points[:,:,feet]-truth[:,:,feet])/.1).square().mean()
    velocity=((torch.diff(points[:,:,feet],dim=1)-torch.diff(truth[:,:,feet],dim=1))*30).square().mean()
    return dict(reconstruction=reconstruction,boundary=boundary,world_fk=position+velocity)

def readout(pred,target,scale,bias,origin,parents,offsets):
    p=(pred-bias)/scale;g=(target-bias)/scale
    feet=[7,8,10,11];a=fk(pred,scale,bias,origin,parents,offsets);b=fk(target,scale,bias,origin,parents,offsets)
    return dict(reconstruction=float((pred[:,:,4:]-target[:,:,4:]).square().mean()),
        root_rmse_m=float((p[:,:,4:7]-g[:,:,4:7]).square().mean().sqrt()),
        foot_rmse_m=float((a[:,:,feet]-b[:,:,feet]).square().mean().sqrt()),
        jerk_max_mps3=float((torch.diff(a,n=3,dim=1)*30**3).norm(dim=-1).max()))
