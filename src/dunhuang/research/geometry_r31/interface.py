"""Explicit new action contract; music and frozen prior weights are preserved, old qualification is not."""
import torch
from torch import nn
from dunhuang.research.state_r27.common import restore
from .common import runtime as original_runtime

class CanonicalNormalizer:
    def __init__(self,legacy):self.legacy=legacy
    def normalize(self,raw):
        old=self.legacy.normalize(raw.clone())
        return torch.cat((old[...,:7],raw[...,7:].clamp(-1,1)),-1)
    def unnormalize(self,value):
        raw=self.legacy.unnormalize(value.clone())
        return torch.cat((raw[...,:7],value[...,7:].clamp(-1,1)),-1)

class ActionIO(nn.Module):
    def __init__(self,legacy,canonical,alphas=None):
        super().__init__();self.canonical=canonical
        self.input_delta=nn.Linear(132,132);self.output_delta=nn.Linear(132,132)
        for layer in (self.input_delta,self.output_delta):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer("scale",legacy.scaler.scale_[7:].detach().clone())
        self.register_buffer("offset",legacy.scaler.min_[7:].detach().clone())
        self.register_buffer("alphas",torch.ones(1000) if alphas is None else alphas.detach().clone())
    def encode(self,value,times):
        body=value[...,7:]
        if self.canonical:
            alpha=self.alphas[times][:,None,None]
            divisor=(alpha+(1-alpha)*self.scale.square()).sqrt()
            base=(body*self.scale+alpha.sqrt()*self.offset)/divisor
        else:base=body
        return torch.cat((value[...,:7],base+self.input_delta(body)),-1)
    def decode(self,value):
        body=value[...,7:];base=(body-self.offset)/self.scale if self.canonical else body
        return torch.cat((value[...,:7],base+self.output_delta(body)),-1)

class Bundle(nn.Module):
    capacity=False
    def __init__(self,reference,io,model):
        super().__init__();self.reference=reference;self.io=io
        self.original_forward=model.forward
        def forward(x,*args,**kwargs):return io.decode(self.original_forward(io.encode(x,args[2]),*args,**kwargs))
        model.forward=forward;object.__setattr__(self,"model_object",model)
    def clear(self):self.reference.clear()
    def set_reference(self,**kwargs):self.reference.set_reference(**kwargs)
    def close(self):
        self.model_object.forward=self.original_forward;self.reference.close()

def runtime(cfg,canonical,checkpoint=None):
    d,n,s,f,e,info,reference=original_runtime(cfg)
    io=ActionIO(n,canonical,d.alphas_cumprod).cuda()
    bundle=Bundle(reference,io,d.model)
    if checkpoint:
        value=torch.load(checkpoint,map_location="cpu",weights_only=True)
        assert value["arm"]==("CanonicalIO" if canonical else "LegacyIO")
        restore(bundle,checkpoint)
    decoder=CanonicalNormalizer(n) if canonical else n
    assert sum(p.numel() for p in bundle.parameters())==300440
    return d,decoder,s,f,e,info,bundle
