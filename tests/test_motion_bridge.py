import numpy as np
import pytest
torch=pytest.importorskip("torch")
from dunhuang.research.motion_bridge.model import MotionBridge,rotations,Injection,soft_control
from dunhuang.research.motion_bridge.objective import terms,total

def sample():
    x=torch.zeros(2,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24)
    x[:,:,4]=torch.arange(150)*.01;k=torch.zeros_like(x,dtype=torch.bool);k[:,52:60,4:]=True;k[:,90:98,4:]=True
    return x,k

def test_raw_known_error_is_not_erased_by_mask_before_training():
    y,k=sample();p=y.clone();p[:,:,4]+=.2;p.requires_grad_()
    loss=terms(p,y,k,torch.ones(151),torch.zeros(151))["anchor_root"]
    loss.backward()
    assert loss>0 and p.grad[k].abs().sum()>0
    assert torch.isfinite(p.grad).all()

def test_rotations_and_seam_losses_have_finite_gradient_at_identity():
    y,k=sample();p=y.clone().requires_grad_()
    parts=terms(p,y,k,torch.ones(151),torch.zeros(151))
    total(parts,{n:1 for n in parts}).backward()
    assert torch.isfinite(p.grad).all()
    assert torch.allclose(rotations(y[:,:,7:].reshape(2,150,24,6)),torch.eye(3).expand(2,150,24,3,3))

def test_mask_controls_are_independent_of_hard_context():
    y,k=sample();v=y.clone();m=k.clone()
    for c in ("correct","removed","removed_left","removed_right","wrong_left","wrong_right"):
        soft_control(y,k,y+2,c)
        assert torch.equal(y,v) and torch.equal(k,m)
    with pytest.raises(ValueError):soft_control(y,k,y,"typo")

def test_parameter_matching_and_common_initialization():
    torch.manual_seed(11);a=MotionBridge(np.ones(151),np.zeros(151),np.ones(222),"B2")
    torch.manual_seed(11);b=MotionBridge(np.ones(151),np.zeros(151),np.ones(222),"B3")
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for n,p in a.named_parameters():
        if not n.startswith("block."):assert torch.equal(p,dict(b.named_parameters())[n])

def test_observation_masks_hide_unknown_and_zero_head_can_learn():
    pytest.importorskip("pytorch3d")
    y,k=sample();a=MotionBridge(np.ones(151),np.zeros(151),np.ones(222),"B3")
    music=torch.zeros(2,150,512);t=torch.tensor([999,19])
    d=a(y,k,y,music,t);assert torch.equal(d,torch.zeros_like(d))
    d.sum().backward()
    assert a.output.weight.grad.norm()>0 and torch.isfinite(a.output.weight.grad).all()
    with torch.no_grad():torch.nn.init.normal_(a.output.weight,std=.01)
    hidden=y.clone();hidden[~k]=100
    assert torch.equal(a(y,k,y,music,t),a(hidden,k,y,music,t))
    assert torch.equal(a(y,torch.zeros_like(k),y,music,t),torch.zeros_like(d))
    assert a(y,k,y,music,t).square().mean((1,2)).sqrt().max()<=.100001

def test_conditional_injection_never_touches_unconditional_and_removes_hook():
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__();self.input_projection=torch.nn.Identity();self.cond_encoder=torch.nn.Identity()
            self.seqTransDecoder=type("Stack",(),{"stack":[type("Layer",(),{"multihead_attn":torch.nn.Identity()})()]})()
        def forward(self,x,m,t,cond_drop_prob):
            z=self.input_projection(x)
            return z if cond_drop_prob==1 else z+self.cond_encoder(m)
    model=Stub();hook=Injection(model,"motion");x=torch.zeros(1,2,3);m=x+.1
    assert torch.allclose(hook.predict(x,m,torch.tensor([0]),x+.2),x+.6)
    assert not hook.enabled;hook.close()
    assert not model.input_projection._forward_hooks
