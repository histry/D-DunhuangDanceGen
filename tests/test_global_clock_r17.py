import numpy as np
import pytest
import torch
from dunhuang.research.global_clock_r17.contract import conditions,metadata,views as nv,feature_distance,factor_conditions
from dunhuang.research.global_clock_r17.model import SharedState,views,merge,fk

def target():return np.random.default_rng(17).normal(size=(225,151)).astype(np.float32)

def test_task_masks_cannot_borrow_unknown_future_root():
    y=target()
    v,k,h=conditions(y,"free");assert not k[:,:13].any() and not h.any()
    poisoned=y.copy();poisoned[~k]=np.nan
    assert np.array_equal(conditions(poisoned,"free")[0],v)
    with pytest.raises(ValueError):conditions(y,"progressive")
    prefix=np.ones((75,151),np.float32)
    v,k,h=conditions(y,"progressive",prefix=prefix)
    assert np.array_equal(v[:75,4:],prefix[:,4:]) and h[:75,4:].all()
    assert not k[75:,:13].any() and not h[75:].any()
    v,k,h=conditions(y,"inbetween",hard=True);assert h[:8,4:].all() and h[-8:,4:].all() and not h[8:-8].any()

def test_global_views_merge_and_gradient():
    x=torch.randn(1,225,151,requires_grad=True);v=views(x);assert torch.equal(v[0,75:],v[1,:75])
    assert torch.equal(merge(v),x)
    merge(v).sum().backward();assert torch.equal(x.grad,torch.ones_like(x))
    assert np.array_equal(nv(metadata())[0,75:],nv(metadata())[1,:75])

def test_state_initial_noop_then_real_bidirectional_dependency():
    torch.manual_seed(17);a=SharedState("cross",8);h=torch.randn(2,150,512)
    state=torch.randn(2,150,298,requires_grad=True);clock=torch.stack((torch.arange(150),torch.arange(75,225))).float()
    a.context=(state,clock);assert torch.equal(a(h),h)
    with torch.no_grad():a.out.weight.fill_(.01)
    a(h)[0].sum().backward()
    assert state.grad[1].abs().sum()>0 and state.grad[0].abs().sum()==0
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in SharedState("self",8).parameters())
    a.context=None
    with pytest.raises(RuntimeError):a(h)

def test_factor_hardness_is_independent_of_source_keys():
    y=target()[:150];event=np.zeros_like(y,bool);event[25:28,79:127]=True
    for pos in ("mid","ends"):
        a,ka,ha=factor_conditions(y,pos,True,False,event);b,kb,hb=factor_conditions(y,pos,True,True,event)
        assert np.array_equal(ha,hb) and kb[event].all() and not ka[event].any()
        assert not a[~ka].any()
    assert not factor_conditions(y,"ends",False,True,event)[2].any()

def test_feature_distance_detects_context_change_beyond_max_abs():
    x=np.ones((75,4800),np.float32);y=x*2
    d=feature_distance(x,y);assert d["relative_l2"]==1 and abs(d["cosine_distance"])<1e-12
    assert feature_distance(x,-x)["cosine_distance"]==pytest.approx(2.)

def test_variable_horizon_fk_root_gradient():
    # Simple chain exercises225 directly, rather than reshaping through150.
    x=torch.zeros(1,225,151);x[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(24)
    x.requires_grad_(True);scale=torch.ones(151);bias=torch.zeros(151);origin=torch.zeros(1,3)
    parents=[-1]+list(range(23));offsets=torch.zeros(24,3);offsets[1:,1]=.1
    p=fk(x,scale,bias,origin,parents,offsets);assert p.shape==(1,225,24,3)
    p[:,:,:,0].sum().backward();assert torch.isfinite(x.grad).all() and x.grad[:,:,4].abs().sum()>0


def test_free_training_pairs_do_not_require_an_external_true_halo():
    from dunhuang.research.global_clock_r17.contract import pair_candidates
    rows=[dict(recording_uid="one",start_frame=t) for t in (0,75,150)]
    assert len(pair_candidates(rows))==2
    assert len(pair_candidates(rows,minimum_start=8))==1
    assert pair_candidates(rows)[0][0]["start_frame"]==0


def test_global_training_loss_has_no_contact_truth_and_finite_root_gradients():
    from dunhuang.research.global_clock_r17.objective import losses
    y=torch.zeros(1,225,151);y[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(24)
    p=y.clone();p[:,:,0:4]=.8;p[:,100,4]=.01;p.requires_grad_(True)
    v,k,h=conditions(y[0].numpy(),"free")
    args=(torch.tensor(v[None]),torch.tensor(k[None]),torch.tensor(h[None]),torch.ones(151),torch.zeros(151),
        torch.zeros(1,3),[-1]+list(range(23)),torch.zeros(24,3))
    a=losses(p,y,*args)
    assert a["world_fk"]>0 and a["reconstruction"]>0
    sum(a.values()).backward()
    assert torch.isfinite(p.grad).all() and p.grad[:,:,4:7].abs().sum()>0
    assert not p.grad[:,:,:4].any()
