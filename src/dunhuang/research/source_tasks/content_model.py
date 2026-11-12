"""Shared-parameter 15/30/dual-rate SDMR architecture, fresh task weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from dunhuang.alignment.model import Encoder

class RateEncoder(Encoder):
    def __init__(self,channels,rate):
        super().__init__(channels)
        if rate not in ("15","30","dual"):raise ValueError("Registered rate required")
        self.rate=rate
        # Equal deterministic regularization; no architecture-specific dropout RNG.
        for module in self.modules():
            if isinstance(module,nn.Dropout):module.p=0.
            if isinstance(module,nn.MultiheadAttention):module.dropout=0.
    def project(self,x,valid,stride):
        x=x[:,::stride];valid=valid[:,::stride]
        position=torch.arange(x.shape[1],device=x.device)[:,None]*stride/30
        frequency=torch.exp(torch.arange(0,128,2,device=x.device)*(-math.log(10000.)/128))
        phase=position*frequency;encoding=torch.zeros_like(x[0]);encoding[:,::2]=phase.sin();encoding[:,1::2]=phase.cos()
        x=self.transformer(x+encoding,src_key_padding_mask=~valid)
        logits=self.pool(x).squeeze(-1).masked_fill(~valid,-1e4)
        global_feature=F.normalize(self.output((logits.softmax(-1)[...,None]*x).sum(1)),dim=-1)
        if stride==2:x=x.repeat_interleave(2,1);valid=valid.repeat_interleave(2,1)
        return global_feature,F.normalize(x,dim=-1),valid
    def forward(self,x,mask):
        x,valid=self.temporal_features(x,mask)
        if self.rate=="15":return self.project(x,valid,2)
        if self.rate=="30":return self.project(x,valid,1)
        a=self.project(x,valid,2);b=self.project(x,valid,1)
        return F.normalize(a[0]+b[0],dim=-1),F.normalize(a[1]+b[1],dim=-1),a[2]&b[2]

class ContentModel(nn.Module):
    def __init__(self,rate):
        super().__init__();self.music=RateEncoder(50,rate);self.motion=RateEncoder(74,rate)
    def score(self,m,d,weight):
        common=m[2][:,None]&d[2];local=(m[1][:,None]*d[1]).sum(-1)
        local=(local*common).sum(-1)/common.sum(-1).clamp_min(1)
        return (m[0][:,None]*d[0]).sum(-1)+weight*local
