import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from dunhuang.research.context_r21.contract import regions,conditions,support_proxy
from dunhuang.research.context_r21.objective import losses,organized,scopes
from dunhuang.research.context_r21.engine import schedule,Predictor,branch_readout
from dunhuang.research.temporal_r20.objective import temporal as old_temporal
from dunhuang.research.objective_r19.objective import physical as old_physical
from dunhuang.research.masked_denoiser.model import MaskConditionedDenoiser
from dunhuang.research.temporal_r20.joint import JointDenoiser,JointState,JointPredictor
from dunhuang.research.temporal_r20.trace import TracedPredictor
from dunhuang.research.global_clock_r17.contract import metadata

def cfg():return json.loads(Path("src/dunhuang/research/context_r21/config.json").read_text())
def target():
    y=torch.zeros(1,225,151);y[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(24);return y
def cache():
    return dict(scale=torch.ones(151),bias=torch.zeros(151),parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
        offsets=torch.tensor([[0.,0,0]]+[[.1,.02,.03]]*23),proxy_foot_ids=[10,11],support_scales=cfg()["support_scales"])

def test_hard_regions_partition_complete_and_are_not_legacy_interior():
    y=target();_,k,h,_=conditions(y,cfg());m=regions(h[:,:,4:].all(-1),3)
    assert [int(m[n].sum()) for n in ("fixed","crossing","editable")]==[10,6,206]
    assert torch.equal(sum(v.int() for v in m.values()),torch.ones(1,222,dtype=torch.int))
    for variant in range(4):
        _,_,hard,_=conditions(y,cfg(),variant)
        z=regions(hard[:,:,4:].all(-1),2);assert sum(int(v.sum()) for v in z.values())==223
    with pytest.raises(ValueError):regions(torch.ones(1,2),1)

def test_pooled_objective_is_actual_r20_a2_on_canonical_contract():
    y=target();v,k,h,_=conditions(y,cfg());p=y.clone();p[:,8,4:7]=.003;p[:,120,7:13]+=.001
    c=cache();origin=torch.zeros(1,3)
    terms,counts=losses(p,y,v,k,h,c,origin,False)
    old,oldcounts=old_temporal(p,y,v,k,h,c,origin,dict(body_velocity=3.,body_acceleration=30.))
    oldph,_=old_physical(p,y,v,k,h,c["scale"],c["bias"],origin,c["parents"],c["offsets"],
        dict(seam_velocity=3.,seam_acceleration=30.,foot_position=.1,foot_velocity=1.))
    for name,value in dict(old,**oldph).items():assert torch.equal(terms[name],value),name

def test_crossing_normalization_raises_mass_without_double_counting():
    y=target();_,k,h,_=conditions(y,cfg());observed=k[:,:,4:7].all(-1)
    _,eligible,_,_,_=scopes(k);m=regions(observed,1)
    error=torch.zeros(1,224,2,3);error[m["crossing"]]=2
    pooled,counts=organized(error,eligible,observed,1,False);separate,new=organized(error,eligible,observed,1,True)
    assert counts==new and counts==dict(crossing=12,editable=1248)
    assert float(pooled)==pytest.approx(4*2/210) and float(separate)==pytest.approx(2.)
    flat=torch.ones_like(error)
    assert torch.equal(organized(flat,eligible,observed,1,False)[0],organized(flat,eligible,observed,1,True)[0])

def test_full_recording_proxy_does_not_reset_run_anchor_at_crop():
    p=np.zeros((50,24,3),np.float32);p[:,:,0]=np.arange(50)[:,None]*.001
    a=support_proxy(p,cfg()["proxy"]);crop=support_proxy(p[20:],cfg()["proxy"])
    assert a["proxy_static"].all()
    assert np.array_equal(a["proxy_anchor"][20],p[0,[10,11]])
    assert not np.array_equal(a["proxy_anchor"][20],crop["proxy_anchor"][0])
    assert (a["proxy_segment"][1:]==a["proxy_segment"][:-1]).all()

def test_support_loss_uses_fixed_population_and_joint_root_chain_gradients():
    y=target();v,k,h,_=conditions(y,cfg());p=y.clone();p[:,80:120,4:7]=.02;p.requires_grad_()
    c=cache();o=torch.zeros(1,3)
    from dunhuang.research.global_clock_r17.model import fk
    truth=fk(y,c["scale"],c["bias"],o,c["parents"],c["offsets"])[:,:,[10,11]].detach()
    proxy=dict(proxy_static=torch.ones(1,225,2,dtype=torch.bool),proxy_segment=torch.zeros(1,225,2,dtype=torch.long),
        proxy_anchor=truth,proxy_floor=torch.zeros(1),proxy_swing=torch.zeros(1,225,2,dtype=torch.bool))
    t,n=losses(p,y,v,k,h,c,o,True,proxy)
    assert n["support_anchor"]==209*2*3 and n["swing_clearance"]==0
    (t["support_anchor"]+t["support_velocity"]).backward()
    assert torch.isfinite(p.grad).all() and p.grad[:,:,4:7].abs().sum()>0 and p.grad[:,:,13:19].abs().sum()>0
    assert not p.grad[:,:,:4].any()

def test_same_training_population_and_t_ledger_across_factors():
    c=cfg();pop=dict(train_cases=list(range(24)),follow_cases=[0,2,4,6])
    a=[schedule(c,pop,n) for n in ("B0","B1","B2","B3")];assert all(x==a[0] for x in a)
    assert all(sum(r["case"]==case for r in a[0])==50 for case in range(24))
    q,m,f=[schedule(c,pop,n) for n in ("Q","M","F")]
    neutral=lambda rows:[{k:v for k,v in row.items() if k!="use_cache"} for row in rows]
    assert neutral(q)==neutral(m)==neutral(f) and sum(r["use_cache"] for r in m)==300
    assert {r["variant"] for r in a[0]}==set(range(4)) and {r["pattern"] for r in a[0]}==set(range(3))

class Dummy(nn.Module):
    def __init__(self):
        super().__init__();self.rotary=nn.Identity();self.abs_pos_encoding=nn.Identity()
        self.input_projection=nn.Linear(151,512);self.final_layer=nn.Linear(512,151)
        self.null_cond_embed=nn.Parameter(torch.randn(1,150,512))
        self.seqTransDecoder=nn.Module();self.seqTransDecoder.stack=nn.ModuleList([nn.Identity(),nn.Identity()])
    def forward(self,x,m,t,cond_drop_prob):
        z=self.input_projection(x);z=z+z.mean(1,keepdim=True)+(self.null_cond_embed if cond_drop_prob else m[:,:,:512])
        for layer in self.seqTransDecoder.stack:z=layer(z)
        return self.final_layer(z)

def test_branch_capture_does_not_change_cfg_and_supports_gradient_readouts():
    torch.manual_seed(21);w=JointDenoiser(MaskConditionedDenoiser(Dummy(),["input_projection.","final_layer."]))
    h=JointState();old=TracedPredictor(JointPredictor(w,h));y=target();v,k,hard,_=conditions(y,cfg())
    args=(y,torch.zeros(2,150,4800),v,k,torch.tensor(metadata()[None]),torch.tensor([19]))
    before=old(*args)[0].detach();p=Predictor(old);p.capture_branches=True;after=p(*args)[0]
    assert torch.equal(before,after.detach()) and len(p.branches)==2
    branches=branch_readout(p,y,k,w,h);assert all(np.isfinite(x["unknown"]) for x in branches.values())
    poison=v.clone();poison[~k]=float("nan");assert torch.equal(after.detach(),p(y,args[1],poison,k,args[4],args[5])[0].detach())
    p.close()
