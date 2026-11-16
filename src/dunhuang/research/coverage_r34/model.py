"""Variable observed reference length with an exact R33 96-token compatibility route."""
import math
import torch
from dunhuang.research.time_r29.model import TimeReference
from dunhuang.research.lodge_adapter_r25.model import PARTS

def time_bias(mapped,n,control,beta):
    if mapped.ndim!=2 or mapped.shape[1]!=256 or control.shape!=mapped.shape or control.dtype!=torch.bool:
        raise ValueError("Explicit Bx256 clock and control mask")
    if not torch.isfinite(mapped).all() or mapped.min()<0 or mapped.max()>n-1 or (mapped.diff(dim=1)<0).any():
        raise ValueError("Finite nondecreasing clock within observed source required")
    d=(torch.arange(n,device=mapped.device)[None,None]-mapped[:,:,None])/4.
    value=torch.where(control[:,:,None],-float(beta)*d.square(),torch.zeros_like(d))
    return torch.cat((torch.zeros_like(value[:,:,:1]),value),-1)

class WindowReference(TimeReference):
    def set_reference(self,features,valid,clock,ref_phase,query_phase,mapped_source=None,control_mask=None):
        n=features.shape[1];self.legacy=n==96 and control_mask is None
        if self.legacy:
            return super().set_reference(features,valid,clock,ref_phase,query_phase,mapped_source)
        if features.ndim!=4 or features.shape[2:]!=(22,9) or n<2 or valid.shape!=features.shape[:-1] or valid.dtype!=torch.bool:
            raise ValueError("Explicit finite BN22x9 reference and joint validity")
        if clock.shape!=(*features.shape[:2],2) or not torch.isfinite(clock).all() or ref_phase.shape!=(*features.shape[:2],8) or query_phase.shape!=(len(features),256,8):
            raise ValueError("Reference clock/phase shape")
        if mapped_source is None or control_mask is None:raise ValueError("Variable source requires map and requested-control scope")
        time_bias(mapped_source,n,control_mask,self.beta)
        if len(mapped_source)!=len(features) or not torch.isfinite(features[valid]).all():raise ValueError("Source batch or observations differ")
        safe=torch.where(valid[:,:,:,None],features,torch.zeros_like(features))
        inp=torch.cat((safe,valid[:,:,:,None].to(safe.dtype)),-1)
        parts=[layer(inp[:,:,p].flatten(2)) for layer,p in zip(self.parts,PARTS)]
        frame=valid.any(-1);present=frame.any(-1)
        mem=self.input(torch.cat(parts+[clock,torch.zeros_like(ref_phase)],-1))*frame[:,:,None]
        for conv in self.temporal:mem=mem+torch.nn.functional.silu(conv(mem.transpose(1,2)).transpose(1,2))*frame[:,:,None]
        mem=self.norm(mem)
        global_token=(mem*frame[:,:,None]).sum(1)/frame.sum(1).clamp_min(1)[:,None]
        self.context=(torch.cat((global_token[:,None],mem),1),torch.cat((torch.ones_like(present[:,None]),frame),1),present,torch.zeros_like(query_phase))
        self.mapped_source=mapped_source;self.control_mask=control_mask;self.readouts={};self.attention_readouts={}
    def inject(self,index,hidden):
        if self.context is None:return hidden
        if self.legacy:return super().inject(index,hidden)
        memory,keys,present,qp=self.context
        if not present.any():return hidden
        q=self.q[index](hidden)+self.qphase[index](qp)
        logits=q@self.k[index](memory).transpose(-1,-2)/math.sqrt(q.shape[-1])
        logits=logits+time_bias(self.mapped_source,memory.shape[1]-1,self.control_mask,self.beta)
        weights=logits.masked_fill(~keys[:,None],-torch.inf).softmax(-1)
        return torch.where(present[:,None,None],hidden+self.out[index](weights@self.v[index](memory)),hidden)
