"""Explicit observed motion and mask -> frozen EDGE motion-token residual.

No pretrained HumanML3D weights, output-pose writer or contact supervision.
B2 and B3 have identical parameter counts; B3 replaces a pointwise block by
attention to BOTH observed sides. Missing observations are never zero truth.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def rotations(x):
    a,b=x[...,:3],x[...,3:]
    u=F.normalize(a,dim=-1,eps=1e-8)
    v=F.normalize(b-(u*b).sum(-1,keepdim=True)*u,dim=-1,eps=1e-8)
    return torch.stack((u,v,torch.cross(u,v,dim=-1)),dim=-2)

def positions(n,d,device):
    t=torch.arange(n,device=device)[:,None]
    f=torch.exp(torch.arange(0,d,2,device=device)*(-math.log(10000.)/d))
    z=torch.zeros(n,d,device=device);z[:,0::2]=torch.sin(t*f);z[:,1::2]=torch.cos(t*f)
    return z

def observation(value,known,scale,bias):
    from pytorch3d.transforms import matrix_to_axis_angle
    if value.shape!=known.shape or known.dtype!=torch.bool or value.shape[1:]!=(150,151):
        raise ValueError("Explicit 150x151 observation contract required")
    k=known.clone();k[:,:,:4]=False
    # Never evaluate rotations at arbitrary hidden values.
    native=(value-bias)/scale
    identity=value.new_tensor([1,0,0,0,1,0]).repeat(24)
    raw=torch.where(k,native,torch.cat((native.new_zeros(7),identity))[None,None])
    pose=rotations(raw[:,:,7:].reshape(-1,150,24,6))
    observed_pose=k[:,:,7:].reshape(-1,150,24,6).all(-1)
    valid_pose=observed_pose[:,1:]&observed_pose[:,:-1]
    angular=matrix_to_axis_angle(pose[:,:-1].transpose(-1,-2)@pose[:,1:])*30
    root=torch.diff(raw[:,:,4:7],dim=1)*30
    valid_root=k[:,1:,4:7]&k[:,:-1,4:7]
    vel=F.pad(torch.cat((root,angular.flatten(-2)),-1),(0,0,1,0))
    vm=F.pad(torch.cat((valid_root,valid_pose.repeat_interleave(3,-1)),-1),(0,0,1,0))
    content=torch.cat((torch.where(k,native,0.)[:,:,4:],torch.where(vm,vel,0.)),-1)
    mask=torch.cat((k[:,:,4:],vm),-1)
    return content,mask,k.any(-1)

class MotionBridge(nn.Module):
    def __init__(self,scale,bias,inv_rms,arm="B3",cap=.1):
        super().__init__()
        if arm not in ("B2","B3"):raise ValueError("Unknown Bridge arm")
        self.arm=arm;self.cap=float(cap)
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))
        self.register_buffer("bias",torch.as_tensor(bias,dtype=torch.float32))
        self.register_buffer("inv_rms",torch.as_tensor(inv_rms,dtype=torch.float32))
        if self.scale.shape!=(151,) or self.inv_rms.shape!=(222,) or (self.scale<=0).any() or not torch.isfinite(self.inv_rms).all():
            raise ValueError("Invalid scaler or Train-only feature statistics")
        self.observed=nn.Linear(444,128)
        self.noisy=nn.Linear(151,128);self.music=nn.Linear(512,128);self.time=nn.Linear(32,128)
        self.sides=nn.Embedding(2,128);self.pool=nn.Linear(256,128)
        self.norm=nn.LayerNorm(128);self.output=nn.Linear(128,512)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        if arm=="B3":self.block=nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
        else:self.block=nn.Sequential(*[nn.Linear(128,128) for _ in range(4)])
    def forward(self,value,known,xt,music,t):
        content,mask,tokens=observation(value,known,self.scale,self.bias)
        content=content*self.inv_rms
        pos=positions(150,128,value.device)[None]
        side=(torch.arange(150,device=value.device)>=75).long()
        fixed=pos+self.sides(side)[None]
        base=self.noisy(xt)+self.music(music)+self.time(positions(1000,32,value.device)[t])[:,None]+pos
        # An empty context gets a neutral temporary key; the final gate is EXACT zero.
        active=tokens.any(-1);keys=tokens.clone();keys[~active,0]=True
        def response(c):
            memory=self.observed(torch.cat((c,mask.to(c.dtype)),-1))+fixed
            means=[]
            for s in (0,1):
                keep=tokens&(side[None]==s)
                means.append((memory*keep[:,:,None]).sum(1)/keep.sum(1).clamp(min=1)[:,None])
            query=base+self.pool(torch.cat(means,-1))[:,None]
            if self.arm=="B3":h=self.block(query,memory,memory,key_padding_mask=~keys,need_weights=False)[0]
            else:
                h=query
                for layer in self.block:h=F.gelu(layer(h))
            return self.output(self.norm(query+h))
        delta=response(content)-response(torch.zeros_like(content))
        rms=delta.square().mean((1,2),keepdim=True).clamp(min=1e-16).sqrt()
        delta=delta*torch.clamp(self.cap/rms.clamp(min=1e-8),max=1.)
        return delta*active[:,None,None]

class Injection:
    """Conditional branch only; unconditional EDGE and hard observations untouched."""
    def __init__(self,model,site):
        self.model=model;self.enabled=False;self.delta=None
        module={"music":model.cond_encoder,"motion":model.input_projection,
                "decoder":model.seqTransDecoder.stack[0].multihead_attn}[site]
        def hook(module,args,out):
            if not self.enabled:return out
            if isinstance(out,tuple):return (out[0]+self.delta,)+out[1:]
            return out+self.delta
        self.handle=module.register_forward_hook(hook)
    def predict(self,xt,music,t,delta):
        self.enabled=False
        with torch.no_grad():unc=self.model(xt,music,t,cond_drop_prob=1.)
        self.delta=delta;self.enabled=delta is not None
        try:cond=self.model(xt,music,t,cond_drop_prob=0.)
        finally:self.enabled=False
        return (unc+2*(cond-unc)).clamp(-1,1)
    def close(self):self.handle.remove()

def music_tokens(model,music):
    with torch.no_grad():return model.cond_encoder(model.abs_pos_encoding(model.cond_projection(music)))

def soft_control(value,known,wrong,control):
    v=value.clone();k=known.clone()
    if control=="removed":k[:]=False
    elif control=="removed_left":k[:,:75]=False
    elif control=="removed_right":k[:,75:]=False
    elif control=="wrong_left":v[:,:75]=wrong[:,:75]
    elif control=="wrong_right":v[:,75:]=wrong[:,75:]
    elif control!="correct":raise ValueError("Unknown soft condition control")
    return v,k
