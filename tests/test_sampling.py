import numpy as np
import pytest
torch=pytest.importorskip('torch')
from dunhuang.sampling import constrained_ddim

class Denoiser:
    betas=torch.ones(1000)
    n_timestep=1000
    guidance_weight=2.
    clip_denoised=True
    alphas_cumprod=torch.linspace(.999,.001,1000)
    calls=0
    def model_predictions(self,x,cond,t,**kwargs):
        self.calls+=1
        return torch.zeros_like(x),torch.tanh(x*.1+cond[:,:,:1]*.01)

def test_anchors_and_overlap_applied_during_all_steps():
    values=torch.zeros((2,150,151));mask=torch.zeros_like(values,dtype=torch.bool)
    values[0,100,7:13]=.4;values[1,25,7:13]=.4
    mask[0,100,7:13]=True;mask[1,25,7:13]=True
    d=Denoiser();x,report=constrained_ddim(d,torch.ones(2,150,4800),values,mask)
    assert d.calls==50
    assert torch.equal(x[mask],values[mask])
    assert torch.equal(x[0,75:],x[1,:75])
    assert report['raw_denoiser_anchor_max_abs_by_step'][-1]>0

def test_conflicting_overlaps_fail_before_model_call():
    values=torch.zeros((2,150,151));mask=torch.zeros_like(values,dtype=torch.bool);mask[0,100,7]=True
    with pytest.raises(ValueError):constrained_ddim(Denoiser(),torch.ones(2,150,4800),values,mask)


def test_observer_cannot_change_samples_or_noise_stream():
    values=torch.zeros(1,150,151);mask=torch.zeros_like(values,dtype=torch.bool)
    cond=torch.ones(1,150,4800)
    torch.manual_seed(3);expected,r1=constrained_ddim(Denoiser(),cond,values,mask,guidance='none')
    observed=[]
    def observer(step,t,predicted,clean):
        observed.append((step,t))
        predicted.fill_(100);clean.fill_(-100)
    torch.manual_seed(3);actual,r2=constrained_ddim(Denoiser(),cond,values,mask,guidance='none',observer=observer)
    assert torch.equal(expected,actual)
    assert r1['noise_sha256']==r2['noise_sha256'] and len(observed)==50
