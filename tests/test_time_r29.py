import torch
import pytest
from dunhuang.research.time_r29.model import TimeReference,mapping,core_bias
from dunhuang.research.state_r27.model import ReferenceModel

def context(a):
    a.context=(torch.randn(2,97,64),torch.ones(2,97,dtype=torch.bool),torch.tensor([True,False]),torch.zeros(2,256,8))
    a.mapped_source=mapping(2,"cpu")

def test_clock_bias_is_neutral_outside_Core_and_for_global_token():
    clock=mapping(2,"cpu");b=core_bias(clock,4)
    assert torch.count_nonzero(b[:,:64])==0 and torch.count_nonzero(b[:,160:])==0 and torch.count_nonzero(b[:,:,0])==0
    assert b[0,64,1]==0 and b[0,159,96]==0
    assert b[0,64,25]<-100
    warp=clock.clone();warp[:,64:160]=torch.linspace(0,95,96).square()/95
    assert torch.isfinite(core_bias(warp,4)).all()

def test_invalid_clocks_fail_without_silent_correction():
    for x in (torch.zeros(2,255),torch.full((2,256),float("nan")),mapping(2,"cpu").flip(1),mapping(2,"cpu")+96):
        with pytest.raises(ValueError):core_bias(x,4)

def test_neutral_instrumentation_is_bitwise_original_and_adds_no_parameters():
    torch.manual_seed(1);old=ReferenceModel(False)
    for layer in old.out:torch.nn.init.normal_(layer.weight,0,.02)
    new=TimeReference(0);new.load_state_dict(old.state_dict(),strict=True)
    context(old);new.context=old.context;new.mapped_source=old.mapped_source;new.record_attention=True
    x=torch.randn(2,256,512)
    assert torch.equal(old.inject(0,x),new.inject(0,x))
    assert old.state_dict().keys()==new.state_dict().keys()
    assert sum(p.numel() for p in new.parameters())==265328

def test_core_clock_changes_only_present_reference_query_injection():
    torch.manual_seed(3);a=TimeReference(4)
    for layer in a.out:torch.nn.init.normal_(layer.weight,0,.02)
    context(a);x=torch.randn(2,256,512)
    correct=a.inject(0,x);a.mapped_source=mapping(2,"cpu",True);wrong=a.inject(0,x)
    assert torch.equal(correct[:,:64],wrong[:,:64]) and torch.equal(correct[:,160:],wrong[:,160:])
    assert torch.equal(correct[1],x[1]) and torch.equal(wrong[1],x[1])
    assert (correct[0,64:160]-wrong[0,64:160]).norm()>0
    a.clear();assert torch.equal(a.inject(0,x),x)


def test_Root_and_body_jerk_vectors_cancel_but_norms_do_not_add():
    import numpy as np
    from dunhuang.research.time_r29.dynamics import decomposition
    t=np.arange(256,dtype=np.float64);p=np.zeros((256,22,3))
    p[:,0,0]=t**3*1e-9
    result=decomposition(p);r=result["regions"]["core_interior"]
    assert result["vector_closure_max_mps3"]<1e-9 and r["templates"]==93
    assert r["Root_translation"]>0 and r["Root_relative_body"]>0
    assert r["world"]<r["Root_translation"]+r["Root_relative_body"]
