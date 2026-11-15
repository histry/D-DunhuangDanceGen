"""Differentiable row6D FK on the official Lodge skeleton, with observable targets."""
import torch
import torch.nn.functional as F

def rotations(x):
    x=x.reshape(*x.shape[:-1],22,6)
    a=F.normalize(x[...,:3],dim=-1,eps=1e-6)
    b=F.normalize(x[...,3:]-(a*x[...,3:]).sum(-1,keepdim=True)*a,dim=-1,eps=1e-6)
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),-2)

def fk22(raw,skeleton):
    R=rotations(raw[...,7:]);J=skeleton.J[0,:22].to(raw)
    p=[J[0]+raw[...,4:7]];world=[R[...,0,:,:]]
    for j in range(1,22):
        parent=int(skeleton.parents[j])
        p.append(p[parent]+(world[parent]@(J[j]-J[parent])[...,None])[...,0])
        world.append(world[parent]@R[...,j,:,:])
    return torch.stack(p,-2)

def relations(points):
    return torch.stack((points[...,20,:]-points[...,16,:],points[...,21,:]-points[...,17,:],points[...,20,:]-points[...,21,:],points[...,18,:]-points[...,16,:],points[...,19,:]-points[...,17,:]),-2)

def supported_losses(pred,target,valid,fps=30):
    if pred.shape!=target.shape or valid.shape!=pred.shape[:-1]:raise ValueError("Geometry target/mask")
    mask=valid[...,None].expand_as(pred)
    position=F.smooth_l1_loss(pred,target,reduction="none",beta=.05)
    vm=valid[:,1:]&valid[:,:-1]
    velocity=F.smooth_l1_loss(torch.diff(pred,dim=1)*fps,torch.diff(target,dim=1)*fps,reduction="none",beta=.5)
    return (position*mask).sum()/mask.sum().clamp_min(1),(velocity*vm[...,None]).sum()/(vm.sum()*3).clamp_min(1)
