"""One genuine225-token EDGE call; all selected parameters keep their source values."""
import hashlib
import torch
from torch import nn
from dunhuang.research.masked_denoiser.model import MaskConditionedDenoiser
from dunhuang.research.global_clock_r17.model import SharedState

class JointDenoiser(MaskConditionedDenoiser):
    def __init__(self,original):
        nn.Module.__init__(self)
        self.net=original.net;self.trainable_names=list(original.trainable_names)
        if self.net.rotary is None or not isinstance(self.net.abs_pos_encoding,nn.Identity):
            raise ValueError("Joint225 requires the pinned RoPE backend")
        old=self.net.null_cond_embed.detach()
        if old.shape!=(1,150,512):raise ValueError("Expected pinned150 null embedding")
        extended=torch.nn.functional.interpolate(old.transpose(1,2),size=225,mode="linear",align_corners=True).transpose(1,2).contiguous()
        self.net.null_cond_embed=nn.Parameter(extended,requires_grad=False)
        self.null_adaptation=dict(policy="linear_time_interpolation_align_corners_true",from_shape=list(old.shape),
            to_shape=list(extended.shape),added_frozen_parameters=75*512,
            source_sha256=hashlib.sha256(old.cpu().numpy().tobytes()).hexdigest(),
            adapted_sha256=hashlib.sha256(extended.cpu().numpy().tobytes()).hexdigest())
        self.net.eval()
    def forward(self,xt,music,t,value,mask,*,clip=True):
        if xt.shape!=value.shape or mask.shape!=xt.shape or xt.shape[1:]!=(225,151) or mask.dtype!=torch.bool:
            raise ValueError("One real225-frame motion/observation contract required")
        if music.shape!=(len(xt),225,4800) or t.shape!=(len(xt),):raise ValueError("225-frame music clock required")
        if not torch.isfinite(value[mask]).all():raise ValueError("Nonfinite observation")
        known=mask.clone();known[:,:,:4]=False;proj=self.net.input_projection
        if proj.context is not None:raise RuntimeError("Reentrant joint context")
        proj.context=(value,known)
        try:
            unc=self.net(xt,music,t,cond_drop_prob=1.)
            cond=self.net(xt,music,t,cond_drop_prob=0.)
            raw=unc+2*(cond-unc)
            return raw.clamp(-1,1) if clip else raw
        finally:proj.context=None

class JointState(SharedState):
    def __init__(self,width=64):
        super().__init__("self",width);self.memory_mode="joint"
    def forward(self,hidden):
        if self.context is None:raise RuntimeError("Joint state absent")
        if self.control=="removed":return hidden
        if self.control!="correct":raise ValueError("Joint uses explicit endpoint interventions")
        state,clock=self.context
        memory=self.encoder(state)
        score=self.q(hidden)@self.k(memory).transpose(-1,-2)/(memory.shape[-1]**.5)
        score=score-abs(clock[:,:,None]-clock[:,None,:])/75.
        return hidden+self.out(score.softmax(-1)@self.v(memory))

class JointPredictor:
    def __init__(self,wrapper,head):
        self.wrapper=wrapper;self.adapter=head
        self.handle=wrapper.net.input_projection.register_forward_hook(lambda m,i,o:head(o))
    def __call__(self,x,music,value,known,meta,t,control="correct"):
        if x.shape!=(1,225,151) or music.shape!=(2,150,4800):raise ValueError("Sealed global225 data required")
        if not torch.equal(music[0,75:],music[1,:75]):raise ValueError("Music ownership mismatch")
        global_music=torch.cat((music[0],music[1,75:]),0)[None]
        safe=torch.where(known,value,torch.zeros_like(value));effective=torch.where(known,safe,x)
        state=torch.cat((effective[:,:,4:],known[:,:,4:].to(x.dtype),meta),-1)
        clock=torch.arange(225,device=x.device,dtype=x.dtype)[None]
        self.adapter.context=(state,clock);self.adapter.control=control
        try:
            p=self.wrapper(x,global_music,t,safe,known)
            return p,p
        finally:self.adapter.context=None
    def close(self):self.handle.remove()
