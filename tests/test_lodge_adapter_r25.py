import numpy as np
import pytest
import torch
from dunhuang.research.lodge_adapter_r25.model import ReferenceAdapter
from dunhuang.research.lodge_adapter_r25.reference import view
from dunhuang.research.lodge_adapter_r25.common import ledger

def fixture():
    rot=np.tile([1,0,0,0,1,0],(96,22)).astype(np.float32)
    p=view(rot);return {k:torch.tensor(v[None]) for k,v in p.items()}

def test_zero_initialization_and_missing_reference_are_exact():
    torch.manual_seed(1);a=ReferenceAdapter(True,16,4,[1],32);p=fixture();a.set_reference(**p);h=torch.randn(1,256,32)
    assert torch.equal(a.inject(0,h),h)
    with torch.no_grad():a.out[0].weight.fill_(.1)
    p["valid"][:]=False;p["features"][:]=float("nan");p["ref_phase"][:]=float("nan");p["query_phase"][:]=float("nan")
    a.set_reference(**p);assert torch.equal(a.inject(0,h),h)
    a.clear();assert torch.equal(a.inject(0,h),h)

def test_zero_head_then_encoder_receives_gradient_through_frozen_layers():
    torch.manual_seed(2);a=ReferenceAdapter(True,16,4,[1],32);p=fixture()
    frozen=torch.nn.Linear(32,4)
    for v in frozen.parameters():v.requires_grad_(False)
    h=torch.randn(1,256,32);opt=torch.optim.SGD(a.parameters(),lr=.1)
    a.set_reference(**p);loss=frozen(a.inject(0,h)).square().mean();loss.backward()
    assert a.out[0].weight.grad.norm()>0 and a.joint[0].weight.grad.norm()==0
    opt.step();opt.zero_grad();a.set_reference(**p);frozen(a.inject(0,h)).square().mean().backward()
    assert a.joint[0].weight.grad.norm()>0 and all(v.grad is None for v in frozen.parameters())

def test_event_phase_parameter_counts_and_initial_state_match():
    torch.manual_seed(3);a=ReferenceAdapter(False,16,4,[1],32)
    torch.manual_seed(3);b=ReferenceAdapter(True,16,4,[1],32)
    assert sum(x.numel() for x in a.parameters())==sum(x.numel() for x in b.parameters())
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())

def test_phase_is_a_real_neural_input_and_invalid_pose_is_masked():
    torch.manual_seed(4);a=ReferenceAdapter(True,16,4,[1],32);p=fixture()
    with torch.no_grad():a.out[0].weight.normal_(0,.1)
    h=torch.randn(1,256,32);a.set_reference(**p);before=a.inject(0,h)
    p["ref_phase"]=p["ref_phase"].flip(1);a.set_reference(**p);after=a.inject(0,h)
    assert (before-after).abs().max()>1e-5
    p["valid"][:,15:30]=False;p["features"][:,15:30]=float("nan");a.set_reference(**p)
    assert torch.isfinite(a.inject(0,h)).all()

def test_reference_view_changes_visibility_without_fake_targets():
    rot=np.tile([1,0,0,0,1,0],(96,22)).astype(np.float32)
    a=view(rot,7,True);b=view(rot,7,True)
    assert all(np.array_equal(a[k],b[k]) for k in a) and not a["valid"].all()
    assert (a["valid"].any(1)==False).sum()==24
    assert a["features"].shape==(96,22,9)

def test_training_ledger_has_matching_population_and_bounded_budget():
    c=dict(seed=25041,steps=600,reference_dropout_second_row_every=5,music_dropout=.2)
    rows=ledger(c,list(range(24)))
    from collections import Counter
    assert len(rows)==600 and set(Counter(x for row in rows for x in row["cases"]).values())=={50}
    assert sum(row["reference_drop"][1] for row in rows)==120


def test_jerk_domains_partition_and_wrists_are_included():
    from dunhuang.research.lodge_adapter_r25.metrics import regional_jerk,source_metrics
    points=np.zeros((256,22,3));cue=np.zeros(256,bool);cue[[64,90,130,159]]=True
    domains=regional_jerk(points,cue);assert sum(r["templates"] for r in domains.values())==253
    source=np.zeros((256,139),np.float32);source[:,7:]=np.tile([1,0,0,0,1,0],22)
    changed=source.copy()
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler("z",.5).as_matrix();changed[:,7+6*20:13+6*20]=R[:2,:].reshape(6)
    m=source_metrics(changed,source,points,points,np.ones(256,bool))
    assert m["upper8_rad"]<1e-6 and m["wrist_rotation_rad"]>.2 and m["upper10_rad"]>.04

def test_occlusion_does_not_leak_velocity_at_hidden_predecessor():
    rot=np.tile([1,0,0,0,1,0],(96,22)).astype(np.float32)
    p=view(rot,9,True);valid=p["valid"]
    affected=~(valid[1:]&valid[:-1]);assert np.count_nonzero(p["features"][1:,:,6:][affected])==0


def test_timing_proxy_is_zero_for_identical_motion_regardless_of_type_order():
    from dunhuang.research.lodge_adapter_r25.metrics import timing_proxy
    from scipy.spatial.transform import Rotation
    x=np.zeros((256,139),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],22)
    angles=.5*np.sin(np.linspace(0,3*np.pi,96));R=Rotation.from_euler("z",angles).as_matrix()
    x[64:160,7+6*20:13+6*20]=R[:,:2,:].reshape(96,6)
    score=timing_proxy(x,x,np.zeros((256,35)))
    assert score["phase_timing_proxy_mae_frames"]==0 and score["phase_order_proxy_inversions"]==0
