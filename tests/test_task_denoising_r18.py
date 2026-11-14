import numpy as np
import pytest
import torch
from dunhuang.research.global_clock_r17.model import SharedState
from dunhuang.research.task_denoising_r18.model import State,memory_support,intervene,observation_mismatch
from dunhuang.research.task_denoising_r18.engine import supervision,history_contract,amplitude
from dunhuang.research.task_denoising_r18.gate import assess

def context():
    return torch.randn(2,150,298),torch.stack((torch.arange(150),torch.arange(75,225))).float()

def test_exclusive_keys_and_bidirectional_gradient_have_no_overlap():
    torch.manual_seed(18);a=State("unique",8);h=torch.randn(2,150,512)
    x,clock=context();x.requires_grad_();a.context=(x,clock)
    with torch.no_grad():a.out.weight.fill_(.01)
    a(h)[0].sum().backward()
    assert not x.grad[0].any() and not x.grad[1,:75].any() and x.grad[1,75:,:147].abs().sum()>0
    assert a.last["overlap_mass"]==0 and a.last["unique_mass"]==pytest.approx(1,abs=1e-6)
    assert a.last["admitted_keys"]==[75,75]

def test_content_permutation_cannot_import_disallowed_overlap_values():
    torch.manual_seed(18);a=State("unique",8);h=torch.randn(2,150,512);x,clock=context()
    with torch.no_grad():a.out.weight.fill_(.01)
    a.context=(x,clock);a.control="content";p=a(h)[0]
    y=x.clone();y[1,:75,:147]+=10000;a.context=(y,clock)
    assert torch.equal(a(h)[0],p)

def test_disentangled_controls_preserve_other_metadata_and_masks():
    x,p=context()
    content,_=intervene(x,p,"content")
    assert torch.equal(content[:,:,147:],x[:,:,147:]) and torch.equal(content[:,:,:147],x[:,:,:147].flip(1))
    for control,a,b in [("clock",294,296),("phase",296,298)]:
        v,_=intervene(x,p,control)
        assert torch.equal(v[:,:,:a],x[:,:,:a]) and torch.equal(v[:,:,b:],x[:,:,b:])
        assert torch.equal(v[:,:,a:b],x[:,:,a:b].flip(1))

def test_key_order_is_not_a_semantic_counterfactual():
    torch.manual_seed(18);a=State("unique",8);h=torch.randn(2,150,512);a.context=context()
    with torch.no_grad():a.out.weight.normal_(std=.01)
    p=a(h);a.control="key_order";q=a(h)
    assert torch.allclose(p,q,atol=1e-6,rtol=1e-5)

def test_full_memory_reproduces_parent_and_parameters_match():
    torch.manual_seed(18);old=SharedState("cross",8);a=State("full",8);a.load_state_dict(old.state_dict())
    h=torch.randn(2,150,512);x=context();a.context=x;old.context=x
    with torch.no_grad():old.out.weight.fill_(.01);a.out.weight.copy_(old.out.weight)
    assert torch.equal(a(h),old(h))
    assert len({sum(p.numel() for p in State(m,8).parameters()) for m in ("self","full","unique")})==1

def test_task_rank_supervision_covers_all_tasks():
    rows=[supervision(i) for i in range(120)]
    for task in ("free","inbetween","progressive"):
        assert sum(t==task for t,r in rows)==40 and sum(t==task and r for t,r in rows)==20

def test_observation_mismatch_preserves_complete_rotation_support():
    v=torch.arange(225*151).reshape(1,225,151).float();k=torch.zeros_like(v,dtype=torch.bool)
    k[:,[4,7,90],7:13]=True;k[:,[4,7],4:7]=True
    x=observation_mismatch(v,k)
    assert torch.equal(x[~k],v[~k]) and torch.equal(x[0,4,7:13],v[0,90,7:13])
    assert torch.equal(x[0,4,4:7],v[0,7,4:7])

def test_history_edit_contract_is_explicit_and_not_r17_relaxation():
    c=history_contract();assert c["committed"]==[0,59] and c["editable_buffer"]==[59,75] and c["future"]==[75,225]
    assert not c["generated_history_training"] and not c["qualified"]
    with pytest.raises(ValueError):history_contract(committed=75,buffer_end=59)

def test_amplitude_objective_cannot_reward_zero_motion_against_moving_target():
    y=torch.zeros(1,225,151);y[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(24)
    y[:,:,4]=torch.arange(225)/100
    p=y.clone();p[:,:,4]=0;p.requires_grad_()
    args=(torch.ones(151),torch.zeros(151),torch.zeros(1,3),[-1]+list(range(23)),torch.zeros(24,3))
    objective=amplitude(p,y,*args)
    assert objective>0
    objective.backward();assert torch.isfinite(p.grad).all()
    assert amplitude(y,y,*args)==0

def test_tiny_gate_rejects_reconstruction_gain_without_condition_use():
    cfg=dict(tiny_cases=[0,1],tasks=["free","inbetween","progressive"],tiny_gate=dict(unknown_ratio=.9,minimum_tasks=4,minimum_content_wins=4))
    def scores(correct):
        return [dict(case=i,task=t,control=c,unknown_mse=correct if c=="correct" else .4)
            for i in [0,1] for t in cfg["tasks"] for c in ("correct","content","observation")]
    result=dict(initial_scores=scores(1.),scores=scores(.5),
        task_counts={t:dict(updates=40,ranking_updates=20,motion_nonzero_gradients=40) for t in cfg["tasks"]})
    assert not assess(result,cfg)["main_admitted"]
