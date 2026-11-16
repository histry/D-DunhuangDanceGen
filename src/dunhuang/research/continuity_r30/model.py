"""Parameter-identical local/global reference reading; frozen Lodge stays untouched."""
import math
import torch
from dunhuang.research.time_r29.model import TimeReference,core_bias

def interpolate(memory,valid,clock):
    if memory.ndim!=3 or clock.shape!=(memory.shape[0],256) or valid.shape!=memory.shape[:2]:
        raise ValueError("B96xD memory, B96 support and B256 clock required")
    core_bias(clock,4.)
    lo=clock.floor().long();hi=clock.ceil().long();alpha=(clock-lo)[...,None]
    batch=torch.arange(len(memory),device=memory.device)[:,None]
    supported=valid[batch,lo] & valid[batch,hi]
    value=memory[batch,lo]*(1-alpha)+memory[batch,hi]*alpha
    return value,supported

class Reference(TimeReference):
    def __init__(self,aligned=False):
        super().__init__(4.);self.aligned=aligned;self.detail=False;self.details={};self.intervention=None
    def clear(self):
        super().clear();self.details={}
    def inject(self,index,hidden):
        if self.context is None or not self.context[2].any():return hidden
        if not self.aligned and not self.detail and self.intervention is None:return super().inject(index,hidden)
        memory,keys,present,qp=self.context
        q=self.q[index](hidden)+self.qphase[index](qp)
        logits=q@self.k[index](memory).transpose(-1,-2)/math.sqrt(q.shape[-1])
        attention=(logits+core_bias(self.mapped_source,self.beta)).masked_fill(~keys[:,None],-torch.inf).softmax(-1)
        values=self.v[index](memory);weighted=attention@values;gate=None
        if self.aligned:
            local,supported=interpolate(memory[:,1:],keys[:,1:],self.mapped_source)
            pair=torch.stack((memory[:,:1].expand_as(local),local),2)
            pair_logits=(q[:,:,None]*self.k[index](pair)).sum(-1)/math.sqrt(q.shape[-1])
            pair_logits[:,:,1]=pair_logits[:,:,1].masked_fill(~supported,-torch.inf)
            gate=pair_logits.softmax(-1)
            paired=(gate[...,None]*self.v[index](pair)).sum(2)
            weighted=torch.cat((weighted[:,:64],paired[:,64:160],weighted[:,160:]),1)
        if self.intervention is not None and self.layers[index]==self.intervention[0]:
            frame=self.intervention[1];weighted=weighted.clone();weighted[:,frame]=weighted[:,frame-1]
        residual=self.out[index](weighted)
        if self.detail:
            self.details[str(self.layers[index])]=dict(attention=attention.detach().cpu().numpy(),
                weighted_v=weighted.detach().cpu().numpy(),residual=residual.detach().cpu().numpy(),
                gate=None if gate is None else gate.detach().cpu().numpy())
        return torch.where(present[:,None,None],hidden+residual,hidden)
