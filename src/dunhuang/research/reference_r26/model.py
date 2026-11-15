"""Ordered part slots fix the R25 permutation ambiguity; Lodge is untouched."""
import math
import torch
from torch import nn
from dunhuang.research.lodge_adapter_r25.model import ReferenceAdapter,PARTS,digest,grad_readout

class Adapter(ReferenceAdapter):
    def __init__(self,ordered,**kwargs):
        super().__init__(False,**kwargs)
        self.ordered=ordered
        self.time_bias_strength=0.
        if ordered:
            del self.joint,self.joint_id
            self.parts=nn.ModuleList([nn.Sequential(nn.Linear(len(p)*10,32),nn.SiLU(),nn.Linear(32,kwargs["part_width"])) for p in PARTS])

    def set_reference(self,features,valid,clock,ref_phase,query_phase,mapped_source=None):
        if features.shape[1:]!=(96,22,9) or valid.shape!=features.shape[:-1]:
            raise ValueError("Reference96x22x9 and joint validity required")
        if not torch.isfinite(clock).all() or clock.shape!=(*features.shape[:2],2):
            raise ValueError("Finite explicit source clock required")
        present=valid.any((1,2));frame=valid.any(-1)
        safe=torch.where(valid[:,:,:,None],features,torch.zeros_like(features))
        inp=torch.cat((safe,valid[:,:,:,None].to(safe.dtype)),-1)
        if self.ordered:
            parts=[layer(inp[:,:,p].flatten(2)) for layer,p in zip(self.parts,PARTS)]
        else:
            joint=(self.joint(inp)+self.joint_id[None,None])*valid[:,:,:,None]
            parts=[joint[:,:,p].sum(2)/valid[:,:,p].sum(2).clamp_min(1)[:,:,None] for p in PARTS]
        # C0-C2 share the same clock; phase proxies are sidecars, not extra inputs.
        mem=self.input(torch.cat(parts+[clock,torch.zeros_like(ref_phase)],-1))*frame[:,:,None]
        for conv in self.temporal:
            mem=mem+torch.nn.functional.silu(conv(mem.transpose(1,2)).transpose(1,2))*frame[:,:,None]
        mem=self.norm(mem)
        global_token=(mem*frame[:,:,None]).sum(1)/frame.sum(1).clamp_min(1)[:,None]
        memory=torch.cat((global_token[:,None],mem),1)
        keys=torch.cat((torch.ones_like(present[:,None]),frame),1)
        self.context=(memory,keys,present,torch.zeros_like(query_phase))
        self.readouts={}
        self.mapped_source=mapped_source

    def inject(self,index,hidden):
        if not self.time_bias_strength:
            return super().inject(index,hidden)
        if self.context is None:return hidden
        memory,keys,present,qp=self.context
        if not present.any():return hidden
        if self.mapped_source is None:raise ValueError("Explicit monotone source map required")
        q=self.q[index](hidden)
        logits=q@self.k[index](memory).transpose(-1,-2)/math.sqrt(q.shape[-1])
        logits=logits+relative_time_bias(self.mapped_source,96,self.time_bias_strength)
        attention=logits.masked_fill(~keys[:,None],-torch.inf).softmax(-1)
        residual=self.out[index](attention@self.v[index](memory))
        return torch.where(present[:,None,None],hidden+residual,hidden)

def relative_time_bias(mapped_source,n,strength):
    if mapped_source.ndim!=2 or not torch.isfinite(mapped_source).all() or (mapped_source.diff(dim=1)<0).any():
        raise ValueError("Finite nondecreasing target-to-source map required")
    if mapped_source.min()<0 or mapped_source.max()>n-1:raise ValueError("Map outside reference")
    distance=(torch.arange(n,device=mapped_source.device)[None,None]-mapped_source[:,:,None])/4.
    # The global event token remains unpenalized; this is not a hard local mask.
    return torch.cat((torch.zeros_like(distance[:,:,:1]),-strength*distance.square()),-1)
