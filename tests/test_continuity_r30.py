import numpy as np
import pytest
import torch
from dunhuang.research.continuity_r30.model import Reference,interpolate
from dunhuang.research.time_r29.model import TimeReference,mapping
from dunhuang.research.continuity_r30.metrics import paired,absolute

def setup(a):
    a.context=(torch.randn(2,97,64),torch.ones(2,97,dtype=torch.bool),torch.tensor([True,False]),torch.zeros(2,256,8))
    a.mapped_source=mapping(2,"cpu")

def test_instrumentation_preserves_frozen_reading_exactly():
    torch.manual_seed(31);old=TimeReference(4);new=Reference(False);new.load_state_dict(old.state_dict())
    for j,layer in enumerate(old.out):
        torch.nn.init.normal_(layer.weight,0,.02);new.out[j].load_state_dict(layer.state_dict())
    setup(old);new.context=old.context;new.mapped_source=old.mapped_source;new.detail=True
    x=torch.randn(2,256,512)
    assert torch.equal(old.inject(0,x),new.inject(0,x))
    assert sum(p.numel() for p in new.parameters())==265328

def test_aligned_reading_is_exact_interpolation_and_parameter_identical():
    memory=torch.arange(96.).view(1,96,1).expand(2,-1,-1).clone().requires_grad_()
    clock=mapping(2,"cpu");clock[:,64:160]=torch.linspace(0,95,96).square()/95
    value,support=interpolate(memory,torch.ones(2,96,dtype=torch.bool),clock)
    assert torch.allclose(value[:,:,0],clock) and support.all()
    value.sum().backward();assert torch.isfinite(memory.grad).all() and memory.grad.abs().sum()>0
    assert Reference(True).state_dict().keys()==Reference(False).state_dict().keys()

def test_missing_support_and_NoRef_never_invent_reference():
    m=torch.randn(2,96,4);valid=torch.ones(2,96,dtype=torch.bool);valid[:,4]=False;c=mapping(2,"cpu")
    _,support=interpolate(m,valid,c);assert not support[:,68].any()
    a=Reference(True);x=torch.randn(2,256,512);assert torch.equal(a.inject(0,x),x)
    with pytest.raises(ValueError):interpolate(m,valid,c+96)

def test_aligned_operator_only_changes_Core_present_rows():
    torch.manual_seed(34);a=Reference(False)
    for layer in a.out:torch.nn.init.normal_(layer.weight,0,.02)
    setup(a);x=torch.randn(2,256,512);old=a.inject(0,x);a.aligned=True;new=a.inject(0,x)
    assert torch.equal(old[:,:64],new[:,:64]) and torch.equal(old[:,160:],new[:,160:])
    assert torch.equal(new[1],x[1]) and (old[0,64:160]-new[0,64:160]).norm()>0

def test_paired_derivatives_use_same_support_and_physical_units():
    t=np.arange(256)/30;p=np.zeros((256,22,3));p[:,:,0]=t[:,None];truth=np.zeros_like(p)
    x=paired(p,truth)
    assert x["Root_translation"]["1"]["rmse"]==pytest.approx(1/np.sqrt(3))
    assert x["Root_translation"]["2"]["rmse"]<1e-10
    assert x["Root_relative_body"]["1"]["rmse"]==0 and x["world"]["3"]["templates"]==93
    assert absolute(p)["root_relative_activity_mps"]==0
    with pytest.raises(ValueError):paired(p[:80],truth)
