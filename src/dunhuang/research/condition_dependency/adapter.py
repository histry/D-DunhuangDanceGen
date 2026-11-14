"""Parameter-matched content-centered Root-relative / SO(3) / timestep condition."""
import torch
from dunhuang.research.boundary_context.adapter import BoundaryContextAdapter

class ContentContextAdapter(BoundaryContextAdapter):
    def __init__(self,scale,bias):
        super().__init__()
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))
        self.register_buffer("bias",torch.as_tensor(bias,dtype=torch.float32))

    def features(self,value,known,t):
        from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle
        if value.shape!=known.shape or known.dtype!=torch.bool or known[:,:,:4].any():raise ValueError("Observed Root/rotation channels only")
        if t.shape!=(len(value),):raise ValueError("One explicit diffusion time per window required")
        v=torch.where(known,value,torch.zeros_like(value))
        # Anchor is an actual observed frame shared by every registered training mask.
        support=known[:,59,4:7].all(-1)
        if bool((known.any((1,2))&~support).any()):raise ValueError("Frame59 Root observation required for relative coordinate")
        v=v.clone();v[:,:,4:6]-=v[:,59:60,4:6].clone();v=torch.where(known,v,torch.zeros_like(v))
        d1=torch.zeros_like(value);d2=torch.zeros_like(value)
        pair=known[:,1:,4:7]&known[:,:-1,4:7];triple=pair[:,1:]&pair[:,:-1]
        d1[:,1:,4:7]=((value[:,1:,4:7]-value[:,:-1,4:7])/self.scale[4:7])*30*pair
        d2[:,2:,4:7]=(d1[:,2:,4:7]-d1[:,1:-1,4:7])*30*triple
        observed=known[:,:,7:].reshape(-1,150,24,6).all(-1)
        raw=((value-self.bias)/self.scale)[:,:,7:].reshape(-1,150,24,6)
        ident=torch.tensor([1,0,0,0,1,0],device=value.device,dtype=value.dtype)
        R=rotation_6d_to_matrix(torch.where(observed[...,None],raw,ident))
        w=matrix_to_axis_angle(R[:,:-1].transpose(-1,-2)@R[:,1:])*30
        valid=observed[:,:-1]&observed[:,1:];w=torch.where(valid[...,None],w,torch.zeros_like(w))
        acc=(w[:,1:]-w[:,:-1])*30;acc=torch.where((valid[:,1:]&valid[:,:-1])[...,None],acc,torch.zeros_like(acc))
        d1[:,1:,7:]=torch.cat([w,torch.zeros_like(w)],-1).reshape(-1,149,144)
        d2[:,2:,7:]=torch.cat([acc,torch.zeros_like(acc)],-1).reshape(-1,148,144)
        u=torch.linspace(0,1,150,device=value.device,dtype=value.dtype)[None,:,None].expand(len(v),-1,-1)
        tau=(t.to(value.dtype)/999)[:,None,None].expand(-1,150,-1)
        return torch.cat([v,known.to(value.dtype),d1,d2,u,tau],-1),support

    def forward(self,value,known,music,t):
        f,support=self.features(value,known,t)
        null=f.clone();null[:,:,:151]=0;null[:,:,302:604]=0
        def core(x):return self.output(self.encoder(self.input(x)+self.music(music)))
        # Same observation mask, time and music in both branches: no mask/music-only shortcut.
        result=core(f)-core(null)
        return torch.where(support[:,None,None],result,torch.zeros_like(result))
