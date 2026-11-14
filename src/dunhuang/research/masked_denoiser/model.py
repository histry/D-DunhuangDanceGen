"""Direct masked motion input and selected motion-denoiser finetuning.

Projection is algebraically Linear([where(mask, observed, x_t), observed*mask,
mask]) with split kernels to preserve exact original empty-mask initialization.
No bounded Decoder residual, output pose paste, or trainable music encoder.
"""
import hashlib
import torch
from torch import nn
from dunhuang.research.motion_bridge.model import rotations

class MaskMotionInput(nn.Module):
    def __init__(self,original):
        super().__init__()
        if not isinstance(original,nn.Linear) or original.in_features!=151:
            raise ValueError("Expected original 151D EDGE motion input")
        self.base=original
        self.observed=nn.Linear(302,original.out_features,bias=False,device=original.weight.device,dtype=original.weight.dtype)
        nn.init.zeros_(self.observed.weight)
        self.context=None
    def forward(self,xt):
        if self.context is None:return self.base(xt)
        value,mask=self.context
        clean=torch.where(mask,value,torch.zeros_like(xt))
        direct=torch.where(mask,value,xt)
        return self.base(direct)+self.observed(torch.cat((clean,mask.to(xt.dtype)),-1))

class MaskConditionedDenoiser(nn.Module):
    def __init__(self,model,prefixes):
        super().__init__();self.net=model;model.input_projection=MaskMotionInput(model.input_projection)
        self.trainable_names=[]
        for name,p in model.named_parameters():
            allowed=any(name.startswith(prefix) for prefix in prefixes) and "rotary" not in name
            p.requires_grad_(allowed)
            if allowed:self.trainable_names.append(name)
        if not self.trainable_names:raise ValueError("No selected motion parameters")
        self.net.eval()
    def train(self,mode=True):
        super().train(mode);self.net.eval() # Frozen music and original dropout stay deterministic.
        return self
    def forward(self,xt,music,t,value,mask,*,clip=True):
        if xt.shape!=value.shape or mask.shape!=xt.shape or mask.dtype!=torch.bool or xt.shape[1:]!=(150,151):
            raise ValueError("150x151 observed-motion contract required")
        if music.shape!=(len(xt),150,4800) or t.shape!=(len(xt),):
            raise ValueError("Real aligned 150x4800 Jukebox and one timestep per window required")
        if not torch.isfinite(value[mask]).all():raise ValueError("Nonfinite observed motion")
        observed=mask.clone();observed[:,:,:4]=False
        projection=self.net.input_projection
        if projection.context is not None:raise RuntimeError("Reentrant motion context")
        projection.context=(value,observed)
        try:
            # Music-only CFG: observed motion is provided to BOTH branches.
            unc=self.net(xt,music,t,cond_drop_prob=1.)
            cond=self.net(xt,music,t,cond_drop_prob=0.)
            raw=unc+2*(cond-unc)
            return raw.clamp(-1,1) if clip else raw
        finally:projection.context=None
    def selected(self):return [p for p in self.parameters() if p.requires_grad]
    def state(self):return {n:p.detach().cpu().clone() for n,p in self.net.named_parameters() if p.requires_grad}
    def load_selected(self,state):
        names=set(self.trainable_names)
        if set(state)!=names:raise ValueError("Checkpoint selected-motion names differ")
        with torch.no_grad():
            for n,p in self.net.named_parameters():
                if n in names:
                    if p.shape!=state[n].shape or not torch.isfinite(state[n]).all():raise ValueError("Invalid selected-motion tensor")
                    p.copy_(state[n])
    def digest(self,frozen=False):
        h=hashlib.sha256()
        for n,p in sorted(self.net.named_parameters()):
            if frozen and p.requires_grad:continue
            h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

def known_loss(pred,value,known,scale,bias):
    """The direction teacher never reads a hidden target coordinate."""
    k=known.clone();k[:,:,:4]=False
    # Mask before all target arithmetic; hidden NaNs must have no effect.
    safe=torch.where(k,value,bias.expand_as(value))
    p=(pred-bias)/scale;g=(safe-bias)/scale
    root=((p[:,:,4:7]-g[:,:,4:7])/.1).square()[k[:,:,4:7]].mean()
    rk=k[:,:,7:].reshape(-1,150,24,6).all(-1)
    identity=pred.new_tensor([1,0,0,0,1,0])
    target=torch.where(rk[:,:,:,None],g[:,:,7:].reshape(-1,150,24,6),identity)
    P=rotations(p[:,:,7:].reshape(-1,150,24,6));G=rotations(target)
    pose=((P-G).square().sum((-1,-2))*.5/.3**2)[rk].mean()
    if not rk.any() or not k[:,:,4:7].any():raise ValueError("Teacher needs real observed Root and complete rotations")
    return root+pose
