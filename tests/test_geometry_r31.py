import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.geometry_r31.gauge import solve
from dunhuang.contracts import matrices

def rotations(T=8):
    return Rotation.random(T*22,random_state=31041).as_matrix().reshape(T,22,3,3)

def test_equivalent_box_encoding_preserves_rotations_and_stability():
    r=rotations();result=solve(r,np.full(132,-2.),np.full(132,2.),iterations=8)
    assert result["feasible"].all() and np.nanmin(result["coefficients"][:,:,[0,2]])>=.05
    decoded=matrices(result["code"],row=True)
    assert np.max(abs(decoded-r))<2e-6

def test_infeasible_alpha_frames_are_preserved_not_replaced():
    r=np.broadcast_to(np.eye(3),(4,22,3,3)).copy();lo=np.full((22,6),-2.);hi=np.full((22,6),2.)
    lo[0,1]=.1;hi[0,1]=.2
    x=solve(r,lo,hi,iterations=2)
    assert not x["feasible"][:,0].any() and np.isnan(x["code"][:,0]).all()
    assert x["feasible"][:,1:].all()

def test_stability_floor_rejects_vanishing_second_vector():
    r=np.broadcast_to(np.eye(3),(3,22,3,3)).copy();lo=np.full((22,6),-2.);hi=np.full((22,6),2.)
    lo[:,4]=0.;hi[:,4]=.001
    x=solve(r,lo,hi,iterations=1)
    assert not x["feasible"].any()

def test_narrow_feasible_box_keeps_geometry_while_changing_coordinates():
    r=np.broadcast_to(np.eye(3),(8,22,3,3)).copy()
    lo=np.full((22,6),-.5);hi=np.full((22,6),.5)
    x=solve(r,lo,hi,iterations=8)
    assert x["feasible"].all() and np.max(abs(x["code"]))<=.500001
    assert np.max(abs(matrices(x["code"],row=True)-r))<1e-6

def test_invalid_boxes_and_nonrotations_fail():
    with pytest.raises(ValueError):solve(rotations(),np.ones(132),np.zeros(132))
    with pytest.raises(ValueError):solve(np.ones((4,22,3,3)),np.full(132,-1),np.ones(132))

def test_constant_scaler_components_do_not_exclude_exact_identity():
    from dunhuang.research.geometry_r31.gauge import scaler_box
    source=np.tile(np.array([1.,0,0,0,1,0]),22)
    scale=np.ones(139)*2;offset=np.r_[np.zeros(7),-1-2*source]
    low,high,constant=scaler_box(scale,offset,np.r_[np.zeros(7),source],np.r_[np.ones(7),source],1e-5)
    assert constant.all()
    r=np.broadcast_to(np.eye(3),(3,22,3,3)).copy()
    x=solve(r,low,high,iterations=2)
    assert x["feasible"].all()

def test_new_normalizer_preserves_canonical_geometry_and_io_noise_limits():
    import torch
    from types import SimpleNamespace
    from dunhuang.research.geometry_r31.interface import ActionIO,CanonicalNormalizer
    class Legacy:
        scaler=SimpleNamespace(scale_=torch.linspace(1,3,139),min_=torch.linspace(-.3,.3,139))
        def normalize(self,x):return (x*self.scaler.scale_+self.scaler.min_).clamp(-1,1)
        def unnormalize(self,x):return (x.clamp(-1,1)-self.scaler.min_)/self.scaler.scale_
    n=Legacy();new=CanonicalNormalizer(n);x=torch.zeros(2,8,139)
    x[:,:,7:]=torch.tensor(rotations(8)[...,:2,:].reshape(8,132),dtype=torch.float32)[None]
    assert torch.allclose(new.unnormalize(new.normalize(x)),x,atol=1e-7)
    io=ActionIO(n,True,torch.tensor([0.,1.]))
    mapped=io.encode(x,torch.tensor([0,1]))
    assert torch.allclose(mapped[0],x[0])
    assert torch.allclose(mapped[1,:,7:],x[1,:,7:]*n.scaler.scale_[7:]+n.scaler.min_[7:])
    io2=ActionIO(n,False,torch.tensor([0.,1.]))
    assert torch.equal(io2.decode(io2.encode(x,torch.tensor([0,1]))),x)

def test_action_bundle_does_not_register_or_optimize_backbone():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from dunhuang.research.geometry_r31.interface import ActionIO,Bundle
    class Ref(nn.Module):
        def clear(self):pass
        def set_reference(self,**kwargs):pass
        def close(self):pass
    class Prior(nn.Module):
        def __init__(self):
            super().__init__();self.large=nn.Linear(139,512)
        def forward(self,x,*args,**kwargs):return x
    prior=Prior();old=prior.forward
    norm=SimpleNamespace(scaler=SimpleNamespace(scale_=torch.ones(139),min_=torch.zeros(139)))
    io=ActionIO(norm,False,torch.ones(1000));bundle=Bundle(Ref(),io,prior)
    assert "model_object" not in bundle._modules
    assert sum(p.numel() for p in bundle.parameters())==35112
    x=torch.randn(2,4,139);assert torch.equal(prior(x,None,None,torch.tensor([9,9])),x)
    bundle.close();assert prior.forward==old

@pytest.mark.parametrize("module",["qualify","train","evaluate","change","probe"])
def test_research_cli_modules_import(module):
    import importlib
    importlib.import_module("dunhuang.research.geometry_r31."+module)
