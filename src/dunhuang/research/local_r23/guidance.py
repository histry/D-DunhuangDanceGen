"""Read both live branches; change only the frozen music guidance coefficient."""
import numpy as np
import torch
from dunhuang.research.temporal_r20.trace import blend
from dunhuang.research.context_r21.sampler import domain_metrics

def correlation(unc,cond,target,known):
    mask=~known[:,:,4:]
    eu=(unc[:,:,4:]-target[:,:,4:])[mask].double()
    ec=(cond[:,:,4:]-target[:,:,4:])[mask].double()
    uu=(eu*eu).mean();cc=(ec*ec).mean();dot=(eu*ec).mean()
    eg=2*ec-eu;identity=4*cc+uu-4*dot
    a=eu-eu.mean();b=ec-ec.mean()
    return dict(elements=eu.numel(),unc_mse=float(uu),cond_mse=float(cc),error_dot_mean=float(dot),
        error_cosine=float(dot/(uu*cc).sqrt().clamp_min(1e-30)),
        error_pearson=float((a*b).sum()/(a.square().sum()*b.square().sum()).sqrt().clamp_min(1e-30)),
        cfg2_raw_mse=float(eg.square().mean()),identity_absolute_error=float(abs(eg.square().mean()-identity)),
        guidance_difference_energy=float((ec-eu).square().mean()),C4_included=False)

class FixedGuidance:
    def __init__(self,predict,weight,s,cache,stages):
        if weight not in (1,2):raise ValueError("Only preregistered CFG1 and CFG2")
        self.old=predict;self.weight=weight;self.adapter=predict.adapter;self.telemetry=predict.telemetry
        self.s=s;self.cache=cache;self.stages=set(stages);self.saved={};self.metrics=[]
    def __call__(self,x,music,value,known,meta,t):
        self.old.telemetry.enabled=False;self.old.capture_branches=True
        baseline,_=self.old(x,music,value,known,meta,t)
        unc,cond=self.old.branches
        raw=unc+self.weight*(cond-unc);pred=blend(raw.clamp(-1,1))
        if self.weight==2:assert torch.equal(pred,baseline)
        moment=int(t.item())
        if moment in self.stages:
            y,m,o,v,k,h,metadata,ts,xt=self.s
            stages={}
            for label,z in (("unc",unc),("cond",cond),("guided_raw",raw),("guided_clip",raw.clamp(-1,1))):
                self.saved[str(moment)+"_"+label]=z.detach().cpu().numpy()
                stages[label]=domain_metrics(blend(z),self.cache,o,h)
            self.metrics.append(dict(t=moment,correlation=correlation(unc,cond,y,k),geometry=stages,
                saturation_fraction=float((raw[:,:,4:]!=raw[:,:,4:].clamp(-1,1)).float().mean())))
        return pred,pred
