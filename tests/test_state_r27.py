import numpy as np
import pytest
import torch
from dunhuang.research.state_r27.trace import row_noise,record_sampler
from dunhuang.research.state_r27.model import ReferenceModel
from dunhuang.research.state_r27.common import metrics
from dunhuang.research.state_r27.qualification import fitting_gate,sampling_gate

class FakeModel:
    def forward(self,x,c,g,t,cond_drop_prob=0):
        return x*.3+c*(1-cond_drop_prob)*.1
    def guided_forward(self,x,c,g,t,w):
        u=self.forward(x,c,g,t,cond_drop_prob=1);v=self.forward(x,c,g,t,cond_drop_prob=0)
        return u+(v-u)*w

class FakeDiffusion:
    guidance_weight=2.
    def __init__(self):self.model=FakeModel()
    def model_predictions(self,x,c,g,t,weight=None,clip_x_start=False):
        raw=self.model.guided_forward(x,c,g,t,self.guidance_weight if weight is None else weight)
        return x-raw,raw.clamp(-1,1) if clip_x_start else raw

def test_noise_rows_are_independent_or_shared_only_by_registered_seed():
    shape=(4,256,139);original=torch.randn
    with row_noise([1,1,2,2],device="cpu") as r:
        a=torch.randn(shape);b=torch.randn_like(a)
    assert torch.randn is original and r["draws"]==2
    assert torch.equal(a[0],a[1]) and not torch.equal(a[0],a[2])
    assert torch.equal(b[2],b[3]) and not torch.equal(a,b)
    with row_noise([1,1,2,2],device="cpu") as q:
        assert torch.equal(a,torch.randn(shape));assert torch.equal(b,torch.randn_like(a))
    assert r==q

def test_actual_state_recorder_observes_cfg_and_restores_on_exception():
    d=FakeDiffusion();x=torch.randn(2,256,139);c=torch.ones_like(x);t=torch.full((2,),999);target=torch.zeros_like(x)
    expected=d.model_predictions(x,c,None,t,weight=1.3,clip_x_start=True)
    with record_sampler(d,[999],target) as r:
        actual=d.model_predictions(x,c,None,t,weight=1.3,clip_x_start=True)
    assert torch.equal(actual[1],expected[1]);assert np.array_equal(r["states"][0],x.numpy())
    assert r["errors"][0].shape==(2,4,3,2) and r["saturation"][0].shape==(2,3,2)
    with pytest.raises(RuntimeError):
        with record_sampler(d,[999]):raise RuntimeError()
    assert torch.equal(d.model_predictions(x,c,None,t,weight=1.3,clip_x_start=True)[1],expected[1])

def test_lowrank_output_is_reference_gated_and_starts_exactly_zero():
    torch.manual_seed(3);a=ReferenceModel(True);h=torch.randn(2,256,512);base=torch.randn(2,256,139)
    a.context=(None,None,torch.tensor([True,False]),None)
    assert torch.equal(a.output_control(None,(h,),base),base)
    with torch.no_grad():a.lowrank["up"].weight.normal_(0,.1)
    out=a.output_control(None,(h,),base)
    assert torch.equal(out[1],base[1]) and torch.equal(out[0,:64],base[0,:64]) and torch.equal(out[0,160:],base[0,160:])
    assert (out[0,64:160]-base[0,64:160]).norm()>0
    a.clear();assert torch.equal(a.output_control(None,(h,),base),base)

def test_pelvis_local_geometry_distinguishes_heading_from_arm_structure():
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(1);points=rng.normal(0,.1,(256,22,3))
    raw=np.zeros((256,139));raw[:,7:]=np.tile([1,0,0,0,1,0],22)
    R=Rotation.from_euler("y",.7).as_matrix();moved=points@R.T+np.array([2,3,4])
    pred=raw.copy();pred[:,7:13]=R[:2].reshape(6)
    result=metrics(pred,moved,raw,points)
    assert result["world_wrist_rmse_m"]>.01 and result["pelvis_wrist_rmse_m"]<1e-12
    assert result["layers"]["pelvis"]>.6 and result["layers"]["wrist"]<1e-10

def fitting_rows(scale):
    rows=[]
    for t in [999,519,119,19]:
        for mode in ["correct","wrong","none"]:
            for i in range(4):
                good=mode=="correct"
                rows.append(dict(teacher=i,t=t,mode=mode,clipped_mse=scale if good else 1.,metrics=dict(body22_rad=.1 if good else .4,pelvis_wrist_rmse_m=.01 if good else .2)))
    return rows

def test_fitting_gate_requires_individual_reference_benefits():
    initial=fitting_rows(1.);now=fitting_rows(.4)
    assert fitting_gate(now,initial)["passed"]
    for r in now:
        if r["teacher"] in (0,1) and r["mode"]=="correct":r["clipped_mse"]=1.
    assert not fitting_gate(now,initial)["passed"]

def test_sampling_gate_does_not_turn_unknown_amplitude_into_success():
    rows=[]
    for mode in ["correct","wrong","none"]:
        for teacher in range(4):
            for seed in range(2):
                rows.append(dict(arm="P",mode=mode,teacher=teacher,seed_index=seed,
                    metrics=dict(body22_rad=.1 if mode=="correct" else .4,pelvis_wrist_rmse_m=.01 if mode=="correct" else .2,body_activity_ratio=1.,arm_range_ratio=1.)))
    assert sampling_gate(rows,"P")["passed"]
    for r in rows:
        if r["teacher"]==0 and r["mode"]=="correct":r["metrics"]["arm_range_ratio"]=None
    assert not sampling_gate(rows,"P")["passed"]
