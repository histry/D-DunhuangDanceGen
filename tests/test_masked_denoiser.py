import numpy as np
import pytest
torch=pytest.importorskip("torch")
from torch import nn
from dunhuang.research.masked_denoiser.model import MaskMotionInput,MaskConditionedDenoiser,known_loss

def tensors():
    x=torch.zeros(2,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24)
    k=torch.zeros_like(x,dtype=torch.bool);k[:,52:60,4:]=True;k[:,90:98,4:]=True
    return x,k

class Stub(nn.Module):
    def __init__(self):
        super().__init__();self.input_projection=nn.Linear(151,8);self.music=nn.Linear(1,8);self.final_layer=nn.Linear(8,151)
    def forward(self,x,m,t,cond_drop_prob):
        return self.final_layer(self.input_projection(x)+self.music(m[:,:,:1])*(1-cond_drop_prob))

def test_direct_input_masks_hidden_nan_and_reads_real_clean_motion():
    x,k=tensors();base=nn.Linear(151,8);projection=MaskMotionInput(base)
    projection.context=(x,k);a=projection(x+1)
    poisoned=x.clone();poisoned[~k]=float("nan");projection.context=(poisoned,k)
    assert torch.equal(a,projection(x+1))
    assert torch.equal(a,base(torch.where(k,x,x+1)))
    projection.context=(x,torch.zeros_like(k));assert torch.equal(projection(x+1),base(x+1))

def test_empty_initialization_matches_original_and_music_frozen():
    x,k=tensors();music=torch.zeros(2,150,4800);t=torch.tensor([19,999]);model=Stub()
    before=(model(x,music,t,1)+2*(model(x,music,t,0)-model(x,music,t,1))).clamp(-1,1)
    wrapper=MaskConditionedDenoiser(model,["input_projection.","final_layer."])
    assert torch.equal(before,wrapper(x,music,t,x,torch.zeros_like(k)))
    frozen=wrapper.digest(True);loss=wrapper(x,music,t,x,k).square().mean();loss.backward()
    assert all(p.grad is None for n,p in wrapper.net.named_parameters() if n.startswith("music."))
    assert wrapper.net.input_projection.observed.weight.grad.abs().sum()>0
    assert wrapper.net.input_projection.context is None and wrapper.digest(True)==frozen

def test_hidden_target_does_not_enter_known_direction_teacher():
    x,k=tensors();pred=(x+.1).requires_grad_()
    p=x.clone();p[~k]=float("nan")
    a=known_loss(pred,x,k,torch.ones(151),torch.zeros(151))
    b=known_loss(pred,p,k,torch.ones(151),torch.zeros(151))
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a,pred,retain_graph=True)[0];gb=torch.autograd.grad(b,pred)[0]
    assert torch.equal(ga,gb) and torch.isfinite(ga).all()

def test_checkpoint_scope_and_context_cleared_on_error():
    x,k=tensors();w=MaskConditionedDenoiser(Stub(),["input_projection.","final_layer."])
    with pytest.raises(ValueError):w.load_selected({})
    state=w.state();w.load_selected(state)
    with pytest.raises(ValueError):w(x,torch.zeros(2,90,4800),torch.ones(2,dtype=torch.long),x,k)
    assert w.net.input_projection.context is None
