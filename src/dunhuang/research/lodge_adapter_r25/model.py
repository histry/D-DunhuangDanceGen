"""Part-aware full sequence memory, global token, and zero-initialized layer controls."""
import math,hashlib
import torch
from torch import nn

PARTS=((0,3,6,9,12,15),(13,16,18,20),(14,17,19,21),(1,4,7,10),(2,5,8,11))

class ReferenceAdapter(nn.Module):
    def __init__(self,phase,width=64,part_width=16,layers=(1,4,7),hidden=512):
        super().__init__();self.phase=phase;self.layers=tuple(layers)
        self.joint=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,part_width))
        self.joint_id=nn.Parameter(torch.zeros(22,part_width))
        self.input=nn.Linear(5*part_width+10,width)
        self.temporal=nn.ModuleList([nn.Conv1d(width,width,3,padding=1) for _ in range(2)])
        self.norm=nn.LayerNorm(width)
        self.q=nn.ModuleList([nn.Linear(hidden,width) for _ in layers])
        self.qphase=nn.ModuleList([nn.Linear(8,width,bias=False) for _ in layers])
        self.k=nn.ModuleList([nn.Linear(width,width) for _ in layers]);self.v=nn.ModuleList([nn.Linear(width,width) for _ in layers])
        self.out=nn.ModuleList([nn.Linear(width,hidden) for _ in layers])
        for layer in self.out:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.context=None;self.handles=[];self.readouts={};self.capture=False
    def set_reference(self,features,valid,clock,ref_phase,query_phase):
        if features.ndim!=4 or features.shape[2:]!=(22,9) or valid.shape!=features.shape[:-1]:raise ValueError("Full valid reference22x9 required")
        if ref_phase.shape!=(*features.shape[:2],8) or query_phase.shape!=(len(features),256,8):raise ValueError("Explicit phase/target clock contract")
        present=valid.any((1,2));frame=valid.any(-1);safe=torch.where(valid[:,:,:,None],features,torch.zeros_like(features))
        inp=torch.cat((safe,valid[:,:,:,None].to(safe.dtype)),-1)
        joint=self.joint(inp)+self.joint_id[None,None];joint=joint*valid[:,:,:,None]
        parts=[joint[:,:,p].sum(2)/valid[:,:,p].sum(2).clamp_min(1)[:,:,None] for p in PARTS]
        rp=torch.where(frame[:,:,None],ref_phase,torch.zeros_like(ref_phase)) if self.phase else torch.zeros_like(ref_phase)
        qp=query_phase if self.phase else torch.zeros_like(query_phase)
        qp=torch.where(present[:,None,None],qp,torch.zeros_like(qp))
        mem=self.input(torch.cat(parts+[clock,rp],-1))
        mem=mem*frame[:,:,None]
        for conv in self.temporal:mem=mem+torch.nn.functional.silu(conv(mem.transpose(1,2)).transpose(1,2))*frame[:,:,None]
        mem=self.norm(mem);global_token=(mem*frame[:,:,None]).sum(1)/frame.sum(1).clamp_min(1)[:,None]
        memory=torch.cat((global_token[:,None],mem),1)
        keys=torch.cat((torch.ones_like(present[:,None]),frame),1)
        self.context=(memory,keys,present,qp);self.readouts={}
    def inject(self,index,hidden):
        if self.context is None:return hidden
        memory,keys,present,qp=self.context
        if not present.any():return hidden
        q=self.q[index](hidden)+self.qphase[index](qp)
        attention=(q@self.k[index](memory).transpose(-1,-2)/math.sqrt(q.shape[-1])).masked_fill(~keys[:,None],-torch.inf).softmax(-1)
        residual=self.out[index](attention@self.v[index](memory))
        result=torch.where(present[:,None,None],hidden+residual,hidden)
        if self.capture:
            self.readouts[str(self.layers[index])]=dict(input_rms=float(hidden.detach().square().mean().sqrt()),residual_rms=float(residual.detach().square().mean().sqrt()))
        return result
    def attach(self,model):
        if self.handles:raise RuntimeError("Adapter already attached")
        if len(model.seqTransDecoder.stack)!=9:raise ValueError("Pinned Lodge9+0 required")
        for i,layer in enumerate(self.layers):
            self.handles.append(model.seqTransDecoder.stack[layer].register_forward_hook(lambda m,a,o,i=i:self.inject(i,o)))
    def clear(self):self.context=None
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[];self.clear()

def digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def grad_readout(adapter):
    prefixes={"joint":"joint","sequence":"temporal","phase_query":"qphase","projections":"out","reference_input":"input"}
    return {key:float(sum((p.grad.detach().square().sum() for name,p in adapter.named_parameters() if name.startswith(prefix) and p.grad is not None),torch.tensor(0.,device=next(adapter.parameters()).device)).sqrt()) for key,prefix in prefixes.items()}
