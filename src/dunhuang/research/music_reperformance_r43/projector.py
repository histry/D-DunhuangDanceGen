"""Regional geodesic edits; the R38 reverse integrator and music weights remain frozen."""
import torch
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle,axis_angle_to_matrix
from dunhuang.research.stabilize_r39.projection import continuous_root


class RegionalProjector:
    def __init__(self,contract,root_m):self.contract=contract.validate();self.root_m=root_m
    def __call__(self,rt,proposal,anchor,legacy_bridge_budget):
        shape=(len(anchor),256,22,6)
        A=rotation_6d_to_matrix(anchor[:,:,7:].reshape(shape));P=rotation_6d_to_matrix(proposal[:,:,7:].reshape(shape))
        v=matrix_to_axis_angle(A.transpose(-1,-2)@P);limit=torch.as_tensor(self.contract.body_budget,device=v.device,dtype=v.dtype)[None]
        v=v*torch.minimum(torch.ones_like(v[...,0]),limit/v.norm(dim=-1).clamp_min(1e-8))[...,None]
        body=(A@axis_angle_to_matrix(v))[...,:2,:].reshape(shape)
        unchanged=(proposal[:,:,7:].reshape(shape)==anchor[:,:,7:].reshape(shape)).all(-1) | (limit==0)
        body=torch.where(unchanged[...,None],anchor[:,:,7:].reshape(shape),body)
        out=anchor.clone();out[:,:,7:]=body.reshape(len(anchor),256,132)
        envelope=torch.as_tensor(self.contract.root_envelope,device=anchor.device,dtype=anchor.dtype)
        scale=rt.n.legacy.scaler.scale_[4:7].to(anchor)
        out[:,:,4:7]=continuous_root(proposal[:,:,4:7],anchor[:,:,4:7],envelope,scale,self.root_m)
        # C4 is a model-state proxy, not an observed contact constraint.
        return out


def sample(rt,music,ref,context,anchor,parent,seeds,contract,root_m):
    from dunhuang.research.bridge_r38 import sampler
    old=sampler.project;sampler.project=RegionalProjector(contract,root_m)
    try:return sampler.sample_bridge(rt,music,ref,context,anchor,parent,seeds)
    finally:sampler.project=old
