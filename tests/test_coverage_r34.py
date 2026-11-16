import numpy as np
import pytest
import torch
from dunhuang.research.coverage_r34.model import WindowReference,time_bias
from dunhuang.research.time_r29.model import TimeReference
from dunhuang.research.coverage_r34.data import extend_clock

def inputs(n):
    return dict(features=torch.randn(2,n,22,9),valid=torch.ones(2,n,22,dtype=torch.bool),
        clock=torch.randn(2,n,2),ref_phase=torch.zeros(2,n,8),query_phase=torch.zeros(2,256,8))

def test_legacy_path_uses_identical_weights_operations_and_output():
    torch.manual_seed(14);parent=TimeReference(4.);child=WindowReference(4.)
    for layer in parent.out:torch.nn.init.normal_(layer.weight,std=.02)
    child.load_state_dict(parent.state_dict());p=inputs(96);parent.set_reference(**p);child.set_reference(**p)
    h=torch.randn(2,256,512)
    assert torch.equal(parent.inject(0,h),child.inject(0,h))
    assert sum(p.numel() for p in child.parameters())==265328

def test_full_reference_attends_entire_requested_clock_and_has_gradient():
    m=WindowReference(4.);p=inputs(256);p.update(mapped_source=torch.arange(256).float()[None].repeat(2,1),control_mask=torch.ones(2,256,dtype=torch.bool))
    for layer in m.out:torch.nn.init.normal_(layer.weight,std=.02)
    m.set_reference(**p);h=torch.randn(2,256,512,requires_grad=True);out=m.inject(1,h)
    assert out.shape==h.shape and not torch.equal(h,out);out.square().mean().backward()
    assert m.parts[0][0].weight.grad.abs().sum()>0 and torch.isfinite(h.grad).all()
    p["valid"][:]=False;m.set_reference(**p);assert torch.equal(h,m.inject(1,h))

def test_time_bias_has_explicit_control_scope_and_never_clamps_invalid_map():
    t=torch.arange(256).float()[None];mask=torch.zeros_like(t,dtype=torch.bool);mask[:,64:160]=True
    v=time_bias(t,256,mask,4)
    assert v.shape==(1,256,257) and (v[:,:64]==0).all() and (v[:,:,0]==0).all()
    assert v[0,100,101]==0 and v[0,100,10]<0
    with pytest.raises(ValueError):time_bias(t+1,256,mask,4)
    with pytest.raises(ValueError):time_bias(t.flip(1),256,mask,4)

def test_extend_reads_surrounding_recording_without_hold_or_loop():
    core=np.arange(96)*1.2+1000;full=extend_clock(core)
    assert np.array_equal(full[64:160],core) and np.allclose(np.diff(full),1.2)
    assert full[0]<core[0] and full[-1]>core[-1]
    with pytest.raises(ValueError):extend_clock(np.zeros(96))

def test_region_mask_keeps_unknowns_out_of_truth_and_counts_boundary_templates():
    from types import SimpleNamespace
    from dunhuang.research.coverage_r34.objective import region_loss
    class Normalizer:
        def unnormalize(self,x):return x
    from dunhuang.research.reference_r26.geometry import fk22
    body=torch.tensor([1.,0,0,0,1,0]).repeat(1,256,22).reshape(1,256,132)
    y=torch.zeros(1,256,139);y[:,:,7:]=body
    skeleton=SimpleNamespace(J=torch.arange(66).float().reshape(1,22,3)/100,parents=[-1]+list(range(21)))
    target=fk22(y,skeleton).detach();target=target-target[:,:,:1]
    valid=torch.ones(1,256,22,dtype=torch.bool);valid[:,10]=False
    partial=body.clone();partial[:,10]=float("nan");target[:,10]=float("nan")
    pred=y.clone().requires_grad_(True);weights=dict(body=1,coordinate_difference=.1,local_geometry=.2,prior_first7=.1,outside=.25)
    loss,parts,dyn,dp,_,counts=region_loss(pred,partial,target,y,Normalizer(),skeleton,weights,torch.ones(2,5),valid)
    assert torch.isfinite(loss+dyn) and float(loss)==pytest.approx(0,abs=1e-7)
    assert counts["D2_part0"]==(254-3)*6
    (loss+dyn).backward();assert torch.isfinite(pred.grad).all()
    assert pred.grad[:,10,7:].abs().sum()==0

@pytest.mark.parametrize("name",["common","model","data","objective","prepare","train","metrics","evaluate","context_contract","context_train","context_eval"])
def test_imports(name):
    import importlib
    importlib.import_module("dunhuang.research.coverage_r34."+name)

def test_context_never_reads_gap_and_preserves_global_clock():
    from dunhuang.research.coverage_r34.context_contract import context
    x=torch.rand(2,256,139)*.5;y=x.clone();y[:,80:144]=100
    a=context(x,[80,144],window_origin=[1000,2000]);b=context(y,[80,144],window_origin=[1000,2000])
    assert torch.equal(a.encoded_input(),b.encoded_input())
    assert a.encoded_input().shape==(2,16,285)
    assert torch.equal(a.global_frame_index,a.frame_index+torch.tensor([1000,2000])[:,None])
    a.frame_index[0,0]=100
    with pytest.raises(ValueError):a.validate()

def test_partial_and_wrong_context_root_does_not_become_observed():
    from dunhuang.research.coverage_r34.context_contract import context
    x=torch.rand(2,256,139)
    p=context(x,[64,160],partial_body=True);assert not p.valid[:,:,:7].any()
    assert torch.isnan(p.values[:,:,:7]).all() and torch.isfinite(p.encoded_input()).all()
    q=context(x,[64,160],"wrong");assert not q.valid[:,:,:7].any()
    assert torch.equal(q.values[0,:,7:],x[1,q.frame_index[0],7:])
    left=context(x,[64,160],"no_left");assert not left.valid[:,:8].any() and left.valid[:,8:].all()
