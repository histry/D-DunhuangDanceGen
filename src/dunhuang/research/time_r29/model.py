"""Parameter-identical C1 with an explicit Core-only target-to-reference clock."""
import math
import torch
from dunhuang.research.state_r27.model import ReferenceModel

def mapping(batch,device,wrong=False):
    x=(torch.arange(256,device=device,dtype=torch.float32)-64).clamp(0,95)
    if wrong:x=(x+24).clamp(0,95)
    return x[None].expand(batch,-1).clone()

def core_bias(mapped,beta):
    if mapped.ndim!=2 or mapped.shape[1]!=256 or not torch.isfinite(mapped).all():
        raise ValueError("Finite Bx256 target-to-source clock required")
    core=mapped[:,64:160]
    if core.min()<0 or core.max()>95 or (core.diff(dim=1)<0).any():raise ValueError("Core clock must be nondecreasing within source96")
    distance=(torch.arange(96,device=mapped.device)[None,None]-mapped[:,:,None])/4.
    value=-float(beta)*distance.square()
    active=torch.zeros(256,dtype=torch.bool,device=mapped.device);active[64:160]=True
    value=torch.where(active[None,:,None],value,torch.zeros_like(value))
    return torch.cat((torch.zeros_like(value[:,:,:1]),value),-1)

class TimeReference(ReferenceModel):
    def __init__(self,beta=4.):
        super().__init__(False);self.beta=float(beta);self.record_attention=False;self.attention_readouts={}
    def set_reference(self,features,valid,clock,ref_phase,query_phase,mapped_source=None):
        if mapped_source is None:mapped_source=mapping(len(features),features.device)
        core_bias(mapped_source,self.beta) # Validate even when neutral; never silently clamp caller maps.
        super().set_reference(features,valid,clock,ref_phase,query_phase,mapped_source)
        self.attention_readouts={}
    def clear(self):
        super().clear();self.attention_readouts={}
    def inject(self,index,hidden):
        if self.context is None:return hidden
        memory,keys,present,qp=self.context
        if not present.any():return hidden
        if self.beta==0 and not self.record_attention:return super().inject(index,hidden)
        q=self.q[index](hidden)+self.qphase[index](qp)
        logits=q@self.k[index](memory).transpose(-1,-2)/math.sqrt(q.shape[-1])
        if self.beta:logits=logits+core_bias(self.mapped_source,self.beta)
        attention=logits.masked_fill(~keys[:,None],-torch.inf).softmax(-1)
        residual=self.out[index](attention@self.v[index](memory))
        result=torch.where(present[:,None,None],hidden+residual,hidden)
        if self.record_attention:
            local=attention[:,64:160,1:];mass=local.sum(-1)
            indices=torch.arange(96,device=hidden.device,dtype=hidden.dtype)
            expected=(local*indices[None,None]).sum(-1)/mass.clamp_min(torch.finfo(hidden.dtype).tiny)
            true=indices[None].expand(len(hidden),-1)
            nearby=(indices[None,None]-true[:,:,None]).abs()<=4
            self.attention_readouts[str(self.layers[index])]=dict(
                global_core=attention[:,64:160,0].mean(-1).detach().cpu().tolist(),
                global_outside=torch.cat((attention[:,:64,0],attention[:,160:,0]),1).mean(-1).detach().cpu().tolist(),
                mass_within_true_clock4=(local*nearby).sum(-1).mean(-1).detach().cpu().tolist(),
                conditional_local_clock_mae=(expected-true).abs().mean(-1).detach().cpu().tolist(),
                expected_local_time=expected.detach().cpu().tolist(),
                residual_rms=residual.detach().square().mean((1,2)).sqrt().cpu().tolist())
        return result
