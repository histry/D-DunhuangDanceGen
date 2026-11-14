import numpy as np
import pytest
import torch
from dunhuang.research.global_clock_r17.contract import conditions
from dunhuang.research.objective_r19.objective import reconstruction,scopes,physical,geometry,static_weights,PHYSICS
from dunhuang.research.objective_r19.model import State,typed_permutation
from dunhuang.research.objective_r19.common import schedule

def target():
    y=torch.zeros(1,225,151);y[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(24)
    v,k,h=conditions(y[0].numpy(),"inbetween")
    return y,torch.tensor(v[None]),torch.tensor(k[None]),torch.tensor(h[None])

def test_loss_scope_counts_and_root_rotation_do_not_mix():
    y,v,k,h=target();p=y.clone();p[:,10,4]=1;p[:,0,4]=2;p[:,:,:4]=float("nan");p.requires_grad_()
    terms,counts=reconstruction(p,y,k)
    assert counts["unknown_root"]==209*3 and counts["observed_root"]==16*3
    assert float(terms["unknown_root"].detach())==pytest.approx(1/(209*3))
    assert float(terms["observed_root"].detach())==pytest.approx(4/(16*3))
    assert terms["unknown_rotation"]==0
    (terms["unknown"]+terms["observed"]).backward();assert torch.isfinite(p.grad).all()
    assert not p.grad[:,:,:4].any()

def test_physical_derivatives_keep_30hz_and_local_seam_counts():
    y,v,k,h=target();p=y.clone();p[:,8,4]=1
    scales=dict(seam_velocity=3.,seam_acceleration=30.,foot_position=.1,foot_velocity=1.)
    parts,counts=physical(p,y,v,k,h,torch.ones(151),torch.zeros(151),torch.zeros(1,3),
        [-1]+list(range(23)),torch.zeros(24,3),scales)
    assert parts["seam_velocity"]==pytest.approx(200/counts["seam_velocity"],rel=1e-6)
    assert parts["seam_acceleration"]==pytest.approx(5400/counts["seam_acceleration"],rel=1e-6)
    assert counts["foot_position"]==209*4*3
    _,_,sv,sa=scopes(k);assert not sv[:,40:180].any() and not sa[:,40:180].any()

def test_static_calibration_cannot_inflate_near_zero_gradients():
    n={name:dict(motion=0.,head=0.) for name in PHYSICS}
    w=static_weights(n,dict(motion=1.,head=1.))
    assert all(value==pytest.approx(.025) for value in w.values())
    n["foot_position"]=dict(motion=100.,head=200.)
    assert static_weights(n,dict(motion=1.,head=1.))["foot_position"]==pytest.approx(.025/200)
    assert all(np.isfinite(v) for v in static_weights(n,dict(motion=0.,head=0.)).values())

def test_rotation_condition_readout_does_not_change_decoder_or_values():
    y,v,k,h=target();p=y.clone();p[:,:,7:10]=torch.tensor([1e-5,0.,0.]);p[:,:,10:13]=torch.tensor([2e-5,0.,0.])
    before=p.clone();r=geometry(p,torch.ones(151),torch.zeros(151))
    assert torch.equal(p,before) and r["first_vector"]["min"]==pytest.approx(1e-5)
    assert r["perpendicular_vector"]["min"]==0 and r["perpendicular_vector"]["below_1e_3"]==225

def test_typed_content_preserves_observation_type_metadata_and_exclusive_support():
    x=torch.zeros(2,150,298)
    x[:,:,:147]=torch.arange(150)[None,:,None]
    x[:,75:,147:294]=1;allowed=torch.zeros(2,150,dtype=torch.bool);allowed[:,40:120]=True
    q=typed_permutation(x,allowed)
    assert torch.equal(q[:,:,147:],x[:,:,147:]) and torch.equal(q[~allowed],x[~allowed])
    assert q[0,40,0]==74 and q[0,75,0]==119
    assert q[0,40,147]==0 and q[0,75,147]==1

def test_typed_hook_has_no_overlap_gradient_leak_or_new_parameters():
    torch.manual_seed(19);a=State("unique",8);h=torch.randn(2,150,512)
    x=torch.randn(2,150,298,requires_grad=True)
    clock=torch.stack((torch.arange(150),torch.arange(75,225))).float()
    a.context=(x,clock);a.control="typed_content"
    with torch.no_grad():a.out.weight.fill_(.01)
    a(h)[0].sum().backward()
    assert not x.grad[0].any() and not x.grad[1,:75].any() and x.grad[1,75:,:147].abs().sum()>0
    from dunhuang.research.task_denoising_r18.model import State as Previous
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in Previous("unique",8).parameters())

def test_fixed_single_task_schedule_balances_cases_noise_and_hardness():
    cfg=dict(seed=19041,steps=240,cases=[0,1],timesteps=[999,519,119,19])
    rows=schedule(cfg);assert rows==schedule(cfg)
    for c in cfg["cases"]:
        for t in cfg["timesteps"]:
            pair=[r for r in rows if r["case"]==c and r["t"]==t]
            assert len(pair)==30 and sum(r["hard"] for r in pair)==15
