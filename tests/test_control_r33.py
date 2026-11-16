import pytest
import torch
from types import SimpleNamespace
from dunhuang.research.control_r33.contracts import ContextPacket,TargetSupport,SourcePacket
from dunhuang.research.control_r33.context import ContextBranch
from dunhuang.research.control_r33.controller import Controller
from dunhuang.research.control_r33.runtime import explicit_forward

def context():
    x=torch.zeros(2,16,139);idx=torch.tensor(list(range(56,64))+list(range(160,168)))[None].expand(2,-1)
    return ContextPacket(x,torch.ones_like(x,dtype=torch.bool),idx,torch.tensor([0]*8+[1]*8)[None].expand(2,-1),torch.zeros_like(idx),"same_window")

def test_missing_condition_is_safe_but_not_fake_truth():
    p=context();p.values[:,:,:7]=float("nan");p.valid[:,:,:7]=False
    assert p.encoded_input().shape==(2,16,285) and torch.isfinite(p.encoded_input()).all()
    y=torch.full((2,256,139),float("nan"));truth=TargetSupport(y,torch.zeros_like(y,dtype=torch.bool),torch.ones_like(y,dtype=torch.bool))
    with pytest.raises(ValueError):truth.complete_for_q()
    assert not truth.teacher_preservation_mask(torch.zeros_like(y,dtype=torch.bool)).any()

def test_context_coordinates_masks_and_sides_fail_closed():
    p=context();p.coordinate_uid=""
    with pytest.raises(ValueError):p.validate()
    p=context();p.frame_index=p.frame_index.clone();p.frame_index[0,0]=100
    with pytest.raises(ValueError):p.validate()
    p=context();p.values[0,0,0]=float("nan")
    with pytest.raises(ValueError):p.validate()

def test_zero_context_and_absent_rows_bypass():
    p=context();p.valid[0]=False;p.values[0]=float("nan")
    m=ContextBranch();m.set_context(p);h=torch.randn(2,256,512)
    assert torch.equal(m.delta(0,h),torch.zeros_like(h))
    for layer in m.out:torch.nn.init.constant_(layer.bias,1.)
    delta=m.delta(0,h);assert torch.equal(delta[0],torch.zeros_like(h[0])) and torch.equal(delta[1],torch.ones_like(h[1]))
    m.clear_context();assert torch.equal(m.delta(0,h),torch.zeros_like(h))

def test_two_residuals_read_same_original_hidden_and_switches_independent():
    class Source(torch.nn.Module):
        def clear(self):pass
        def inject(self,i,h):return h+2
    class Context(torch.nn.Module):
        packet=True
        def delta(self,i,h):return h*3
        def clear_context(self):self.packet=None
        def set_context(self,p):self.packet=p
    c=Controller(Source(),Source(),Context());c.route="general";h=torch.ones(2,4,8)
    assert torch.equal(c.inject(0,h),4*h+2)
    c.clear_source();assert c.context_branch.packet is True
    c.route="general";c.clear_context();assert torch.equal(c.inject(0,h),h+2)

def test_catalog_router_is_explicit_not_learned():
    assert SourcePacket("aist",{},("s",)).branch()=="general"
    assert SourcePacket("change",{},("s",)).branch()=="change"
    assert SourcePacket("aist",{},("s",),"change_replay").branch()=="change"
    with pytest.raises(ValueError):SourcePacket("unknown",{},("s",)).branch()

def test_explicit_time_binding_supports_keyword_and_positional():
    class IO:
        def encode(self,x,t):return x+t[:,None,None]
        def decode(self,x):return x
    def original(x,cond,genre,t,cond_drop_prob=0):return x+t[:,None,None]
    f=explicit_forward(IO(),original);x=torch.zeros(2,4,139);t=torch.tensor([1,2])
    assert torch.equal(f(x,None,None,t),f(x,cond=None,genre=None,t=t))
    with pytest.raises(ValueError):f(x,None,None,t.float())


def test_paired_dynamics_keeps_real_motion_and_translation_invariance():
    from dunhuang.research.control_r33.objective import body_dynamics
    t=torch.arange(96,dtype=torch.float64)/30
    target=torch.zeros(2,96,22,3,dtype=torch.float64);target[:,:,:,0]=torch.sin(t*5)[None,:,None]*torch.arange(22)[None,None,:]
    observed=torch.ones(2,96,22,dtype=torch.bool)
    loss,_,counts=body_dynamics(target.clone()+3,target,observed,torch.ones(2,5))
    assert float(loss)<1e-18 and counts['D2_part1']==2*94*4
    shifted=target.clone();shifted[:,40,20,1]+=.2;shifted.requires_grad_(True)
    value,parts,_=body_dynamics(shifted,target,observed,torch.ones(2,5));assert value>0
    value.backward();assert torch.isfinite(shifted.grad).all()


def test_dynamic_unknown_frames_do_not_become_zero_targets():
    from dunhuang.research.control_r33.objective import derivative_support,body_dynamics
    valid=torch.ones(1,96,22,dtype=torch.bool);valid[:,40]=False
    m=derivative_support(valid,2);assert not m[:,38:41].any() and m[:,:38].all()
    target=torch.zeros(1,96,22,3);target[:,40]=float('nan');pred=torch.zeros_like(target);pred[:,40]=100.
    value,_,_=body_dynamics(pred,target,valid,torch.ones(2,5));assert value==0
    with pytest.raises(ValueError):body_dynamics(pred,target,torch.zeros_like(valid),torch.ones(2,5))


@pytest.mark.parametrize('name',['contracts','context','controller','runtime','preflight','objective','calibrate','prepare_held','train','evaluate'])
def test_r33_cli_modules_import(name):
    import importlib
    importlib.import_module('dunhuang.research.control_r33.'+name)


def test_dynamic_derivative_uses_thirty_hertz_vector_error():
    from dunhuang.research.control_r33.objective import body_dynamics
    target=torch.zeros(1,96,22,3,dtype=torch.float64);p=target.clone();p[0,:,20,0]=torch.arange(96,dtype=torch.float64)/30
    _,parts,_=body_dynamics(p,target,torch.ones(1,96,22,dtype=torch.bool),torch.ones(2,5))
    assert float(parts['D1'])==pytest.approx(1/120,abs=1e-12)
    assert float(parts['D2'])<1e-20


def test_held_beats_use_core_window_then_local_clock():
    from dunhuang.research.control_r33.inventory import core_beat_positions
    import numpy as np
    music=np.zeros((2,256,35));music[0,[17,70,159,160],-1]=1
    assert core_beat_positions(music)==[6,95]
    with pytest.raises(ValueError):core_beat_positions(music[:,:96])
