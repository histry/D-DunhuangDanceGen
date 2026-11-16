"""Independent context residual, zero initialized; no cross-gap temporal convolution."""
import math
import torch
from torch import nn

class ContextBranch(nn.Module):
    def __init__(self,hidden=512,width=64,layers=3):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(285,128),nn.SiLU(),nn.Linear(128,width))
        self.q=nn.ModuleList([nn.Linear(hidden,width) for _ in range(layers)])
        self.k=nn.ModuleList([nn.Linear(width,width) for _ in range(layers)]);self.v=nn.ModuleList([nn.Linear(width,width) for _ in range(layers)])
        self.out=nn.ModuleList([nn.Linear(width,hidden) for _ in range(layers)])
        for layer in self.out:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.packet=None
    def set_context(self,packet):
        packet.validate();self.packet=packet
    def clear_context(self):self.packet=None
    def delta(self,index,hidden):
        if self.packet is None:return torch.zeros_like(hidden)
        p=self.packet;keys=p.valid.any(-1);present=keys.any(-1)
        if not present.any():return torch.zeros_like(hidden)
        memory=self.encoder(p.encoded_input())
        # The synthetic safe key prevents NaN only; absent rows are explicitly bypassed.
        safe=keys.clone();safe[~present,0]=True
        logits=self.q[index](hidden)@self.k[index](memory).transpose(-1,-2)/math.sqrt(memory.shape[-1])
        weights=logits.masked_fill(~safe[:,None],-torch.inf).softmax(-1)
        delta=self.out[index](weights@self.v[index](memory))
        return torch.where(present[:,None,None],delta,torch.zeros_like(delta))
