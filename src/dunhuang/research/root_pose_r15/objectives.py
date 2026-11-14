"""Train-oracle objectives and differentiable Native-parity FK."""
import torch
from dunhuang.research.boundary_state_r14.conflict import fk
from dunhuang.research.motion_bridge.model import rotations

def parts(pred,target,scale,bias,origin,parents,offsets):
    p=(pred-bias)/scale;g=(target-bias)/scale
    mask=torch.zeros_like(pred,dtype=torch.bool);mask[:,60:90,4:]=True
    candidate=torch.where(mask,pred,target)
    P=rotations(p[:,60:90,7:13]);G=rotations(g[:,60:90,7:13])
    root=((p[:,60:90,4:7]-g[:,60:90,4:7])/.1).square().mean()+(P-G).square().sum((-1,-2)).mean()/.3**2
    cp=(candidate-bias)/scale
    for n,unit in ((1,3.),(2,30.)):
        delta=torch.diff(cp[:,:,4:7],n=n,dim=1)*30**n
        truth=torch.diff(g[:,:,4:7],n=n,dim=1)*30**n
        # The crossing templates at each actual seam, not derivatives of detached windows.
        indices=[60-n,90-n]
        root=root+((delta[:,indices]-truth[:,indices])/unit).square().mean()
    points=fk(candidate,scale,bias,origin,parents,offsets)
    truth=fk(target,scale,bias,origin,parents,offsets)
    feet=[7,8,10,11]
    position=((points[:,60:90,feet]-truth[:,60:90,feet])/.1).square().mean()
    velocity=((torch.diff(points[:,:,feet],dim=1)[:,59:90]-torch.diff(truth[:,:,feet],dim=1)[:,59:90])*30).square().mean()
    return dict(root=root,world_foot=position+velocity)
