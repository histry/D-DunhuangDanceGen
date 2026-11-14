import torch
import pytest
import numpy as np
from torch import nn
from dunhuang.research.masked_denoiser.model import MaskConditionedDenoiser
from dunhuang.research.global_clock_r17.model import Predictor as SplitPredictor
from dunhuang.research.global_clock_r17.contract import conditions,metadata
from dunhuang.research.temporal_r20.joint import JointDenoiser,JointState,JointPredictor
from dunhuang.research.temporal_r20.conditions import endpoint
from dunhuang.research.temporal_r20.trace import blend,TracedPredictor
from dunhuang.research.temporal_r20.common import schedule,grid
from dunhuang.research.temporal_r20.objective import temporal,GROUPS,stage_metrics

class Dummy(nn.Module):
    def __init__(self):
        super().__init__();self.rotary=nn.Identity();self.abs_pos_encoding=nn.Identity()
        self.input_projection=nn.Linear(151,512);self.final_layer=nn.Linear(512,151)
        self.null_cond_embed=nn.Parameter(torch.randn(1,150,512))
        self.seqTransDecoder=nn.Module();self.seqTransDecoder.stack=nn.ModuleList([nn.Identity(),nn.Identity()])
    def forward(self,x,music,t,cond_drop_prob):
        p=self.input_projection(x);cond=self.null_cond_embed if cond_drop_prob else music[:,:,:512]
        p=p+p.mean(1,keepdim=True)+cond
        for layer in self.seqTransDecoder.stack:p=layer(p)
        return self.final_layer(p)

def fixture():
    y=torch.zeros(1,225,151);y[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(24)
    v,k,h=conditions(y[0].numpy(),"inbetween")
    return y,torch.tensor(v[None]),torch.tensor(k[None]),torch.tensor(h[None])

def test_joint_length_null_policy_no_new_selected_weights_and_poison():
    torch.manual_seed(20);old=MaskConditionedDenoiser(Dummy(),["input_projection.","final_layer."])
    before=old.state();joint=JointDenoiser(old)
    assert joint.net.null_cond_embed.shape==(1,225,512) and not joint.net.null_cond_embed.requires_grad
    assert all(torch.equal(v,joint.state()[n]) for n,v in before.items())
    y,v,k,h=fixture();music=torch.zeros(1,225,4800);t=torch.tensor([19])
    a=joint(y,music,t,v,k);poison=v.clone();poison[~k]=float("nan")
    assert torch.equal(a,joint(y,music,t,poison,k))
    with pytest.raises(ValueError):joint(y,music[:,:150],t,v,k)

def test_direct_right_visibility_differs_from_split_in_one_forward():
    torch.manual_seed(20)
    split=MaskConditionedDenoiser(Dummy(),["input_projection.","final_layer."]);joint=JointDenoiser(
        MaskConditionedDenoiser(Dummy(),["input_projection.","final_layer."]))
    y,v,k,h=fixture();music=torch.zeros(2,150,4800);meta=torch.tensor(metadata()[None]);t=torch.tensor([19])
    sp=SplitPredictor(split);jp=JointPredictor(joint,JointState())
    vv=v.clone();vv[:,217:,4:7]=.4
    a=sp(y,music,v,k,meta,t)[0];b=sp(y,music,vv,k,meta,t)[0]
    assert torch.equal(a[:,8:75],b[:,8:75])
    a=jp(y,music,v,k,meta,t)[0];b=jp(y,music,vv,k,meta,t)[0]
    assert not torch.equal(a[:,8:75],b[:,8:75])
    jp.close()

def test_endpoint_controls_do_not_change_common_middle_denominator():
    y,v,k,h=fixture();donor=y+.2
    for c in ("left_wrong","right_wrong","left_absent","right_absent","both_absent"):
        value,known,hard,original=endpoint(y,donor,c)
        assert torch.equal(original,k) and torch.equal(known[:,8:217],k[:,8:217])
        assert not hard[:,:,:4].any()
    assert not endpoint(y,donor,"left_absent")[1][:,:8].any()

def test_joint_requires_same_real_music_ownership():
    joint=JointDenoiser(MaskConditionedDenoiser(Dummy(),["input_projection.","final_layer."]))
    jp=JointPredictor(joint,JointState());y,v,k,h=fixture();m=torch.zeros(2,150,4800);m[1,0,0]=1
    with pytest.raises(ValueError):jp(y,m,v,k,torch.tensor(metadata()[None]),torch.tensor([19]))
    jp.close()

def test_feature_blend_is_an_identity_for_identical_views_and_preserves_edges():
    x=torch.randn(1,225,151);views=torch.cat((x[:,:150],x[:,75:]),0)
    assert torch.equal(blend(views,"mean"),x)
    assert torch.allclose(blend(views,"quintic"),x,atol=3e-7)
    y=torch.randn(2,150,151);z=blend(y,"quintic")
    assert torch.equal(z[0,:75],y[0,:75]) and torch.equal(z[0,150:],y[1,75:])
    assert torch.equal(z[0,75],y[0,75]) and torch.equal(z[0,149],y[1,74])
    with pytest.raises(ValueError):blend(x,"quintic")

def test_stratified_schedule_support_and_matched_cache_clock():
    cfg=dict(seed=20041,steps=240,train_cases=[0,1])
    a=schedule(cfg,"A1");b=schedule(cfg,"A2");c=schedule(cfg,"A3")
    assert a==b
    assert [{k:v for k,v in r.items() if k!="use_cache"} for r in a]==[{k:v for k,v in r.items() if k!="use_cache"} for r in c]
    assert sum(r["use_cache"] for r in c)==120
    assert {r["t"]//100 for r in a if r["slot"]=="uniform"}==set(range(10))
    assert {r["t"] for r in a if r["slot"]=="grid"}==set(grid())
    assert set(r["t"] for r in schedule(cfg,"A0"))=={999,519,119,19}

def test_paired_whole_body_dynamics_keep_30hz_and_do_not_reward_stillness():
    y,v,k,h=fixture();cache=dict(scale=torch.ones(151),bias=torch.zeros(151),parents=[-1]+list(range(23)),offsets=torch.zeros(24,3))
    origin=torch.zeros(1,3);p=y.clone();eps=1e-5;p[:,100,4]=eps;p.requires_grad_()
    terms,counts=temporal(p,y,v,k,h,cache,origin,dict(body_velocity=1.,body_acceleration=1.))
    expected=[]
    for group,joints in GROUPS.items():
        n=len([j for j in joints if j not in (7,8,10,11)])
        expected.append((eps*30)**2*n/counts["body_velocity_"+group])
    assert float(terms["body_velocity"].detach())==pytest.approx(np.mean(expected),rel=1e-5)
    assert counts["body_velocity_lower"]==210*4*3 and counts["body_acceleration_lower"]==211*8*3
    sum(terms.values()).backward();assert torch.isfinite(p.grad).all() and not p.grad[:,:,:4].any()
    moving=y.clone();moving[:,:,4]=torch.arange(225)/100
    still=y.clone()
    good,_=temporal(moving,moving,v,k,torch.zeros_like(h),cache,origin,dict(body_velocity=3.,body_acceleration=30.))
    bad,_=temporal(still,moving,v,k,torch.zeros_like(h),cache,origin,dict(body_velocity=3.,body_acceleration=30.))
    assert good["body_velocity"]==0 and bad["body_velocity"]>0

def test_temporal_domains_are_disjoint_and_complete():
    y,v,k,h=fixture();cache=dict(scale=torch.ones(151),bias=torch.zeros(151),parents=[-1]+list(range(23)),offsets=torch.zeros(24,3))
    r=stage_metrics(y,cache,torch.zeros(1,3));b=r["fk_jerk"]["buckets"]
    assert sum(v["templates"] for v in b.values())==222
    assert b["interior"]["templates"]==207 and r["valid_rotation"]


def test_joint_telemetry_records_real225_cfg_and_backward_without_150_reshape():
    torch.manual_seed(21)
    joint=JointDenoiser(MaskConditionedDenoiser(Dummy(),["input_projection.","final_layer."]))
    traced=TracedPredictor(JointPredictor(joint,JointState()))
    traced.telemetry.enabled=True;traced.telemetry.grad_enabled=True
    y,v,k,h=fixture()
    out,windows=traced(y,torch.zeros(2,150,4800),v,k,torch.tensor(metadata()[None]),torch.tensor([19]))
    assert windows.shape==(1,225,151) and len(traced.last["saturation"]["per_joint"])==24
    assert torch.equal(traced.last["preclip"].clamp(-1,1),windows.detach())
    out.square().mean().backward()
    assert "decoder_last" in traced.telemetry.backward and "combined" in traced.telemetry.backward
    traced.close()
