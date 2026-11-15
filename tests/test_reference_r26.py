import json
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from dunhuang.research.reference_r26.model import Adapter,relative_time_bias
from dunhuang.research.reference_r26.reference import view,events
from dunhuang.research.reference_r26.geometry import fk22,relations,supported_losses
from dunhuang.research.reference_r26.common import ledger

def packet():
    rng=np.random.default_rng(4)
    rot=Rotation.from_rotvec(rng.normal(0,.2,(96*22,3))).as_matrix()[:,:2,:].reshape(96,132)
    return {k:torch.tensor(v[None]) for k,v in view(rot).items()}

def make(ordered):
    torch.manual_seed(26041)
    return Adapter(ordered,width=16,part_width=4,layers=(1,),hidden=32)

def test_joint_binding_survives_full_visibility_and_swapped_wrist():
    p=packet();h=torch.randn(1,256,32)
    for ordered in [False,True]:
        a=make(ordered);a.set_reference(**p);before=a.context[0].clone()
        q={k:v.clone() for k,v in p.items()};q["features"][:,:,[16,20]]=q["features"][:,:,[20,16]]
        a.set_reference(**q);delta=(a.context[0]-before).abs().max()
        assert delta>1e-4 if ordered else delta<2e-6

def test_zero_control_missing_storage_and_common_initialization():
    old,new=make(False),make(True);common=set(old.state_dict())&set(new.state_dict())
    assert all(torch.equal(old.state_dict()[k],new.state_dict()[k]) for k in common)
    p=packet();h=torch.randn(1,256,32)
    new.set_reference(**p);assert torch.equal(new.inject(0,h),h)
    with torch.no_grad():new.out[0].weight.normal_()
    p["valid"][:]=False;p["features"][:]=float("nan")
    new.set_reference(**p);assert torch.equal(new.inject(0,h),h)

def test_constant_speed_and_static_have_no_fabricated_stages():
    for speed in [0.,.3]:
        angle=np.arange(96)*speed/30
        R=Rotation.from_euler("z",angle).as_matrix()[:,:2,:]
        rot=np.repeat(R[:,None],22,axis=1).reshape(96,132)
        assert all(x["status"]=="UNKNOWN" and x["frame"] is None for x in events(rot))

def test_observed_derivative_templates_and_hidden_joint():
    p=view(np.tile([1,0,0,0,1,0],(96,22)),4,True)
    assert not p["valid"].all() and np.count_nonzero(p["features"][0,:,6:])==0
    mask=~(p["valid"][1:]&p["valid"][:-1])
    assert not np.count_nonzero(p["features"][1:,:,6:][mask])

def test_relative_clock_bias_follows_known_legal_map_without_endpoint_cheat():
    u=torch.arange(96,dtype=torch.float32)[None]
    w=torch.tensor(np.interp(u.numpy()[0],[0,32,64,95],[0,24,64,95]))[None]
    logits=relative_time_bias(w,96,4)[:,:,1:]
    assert (logits.argmax(-1)-w.round()).abs().max()<=1
    assert abs(int(logits.argmax(-1)[0,32])-32)==8
    with pytest.raises(ValueError):relative_time_bias(w.flip(1),96,4)
    with pytest.raises(ValueError):relative_time_bias(w+100,96,4)

def test_geometry_gradient_translation_invariance_and_zero_difference():
    class Skeleton:
        parents=[-1]+[0]*21
        J=torch.randn(1,22,3)
    raw=torch.zeros(1,8,139);raw[:,:,7:]=torch.tensor([1,0,0,0,1,0]*22)
    raw.requires_grad_();points=fk22(raw,Skeleton)
    moved=raw.detach().clone();moved[:,:,4:7]+=12
    assert torch.allclose(relations(points),relations(fk22(moved,Skeleton)),atol=3e-6)
    truth=relations(points).detach();valid=torch.ones(truth.shape[:-1],dtype=torch.bool)
    p,v=supported_losses(relations(points),truth,valid);assert p==v==0
    loss,_=supported_losses(relations(points),truth+.01,valid)
    loss.backward();assert torch.isfinite(raw.grad).all() and raw.grad[:,:,7:].norm()>0
    false=torch.zeros_like(valid);p,v=supported_losses(relations(points.detach()),truth+1,false);assert p==v==0

def test_common_ledger_has_full_occluded_and_dropped_presentations():
    cfg=dict(seed=26041,steps=1200);rows=ledger(cfg,list(range(96)))
    assert rows==json.loads(json.dumps(rows))
    assert len(rows)==1200 and set(x for r in rows for x in r["cases"])==set(range(96))
    assert sum(not v for r in rows for v in r["occluded"])==1200
    assert sum(r["reference_drop"][1] for r in rows)==240
