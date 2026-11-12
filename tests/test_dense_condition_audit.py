import copy,json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from dunhuang.generation.event_music_adapter import EventMusicAdapter
from dunhuang.generation.event_visibility import clean_geometry,intervene
from dunhuang.alignment.dense_coverage import real_starts,sample_tokens,fast_candidates
from dunhuang.alignment.conditional_candidates import build_candidates
from dunhuang.planning.primitive_contract import build,verify,identity
from dunhuang.io import sha
torch.set_num_threads(2)

def fixture_geometry():
    torch.manual_seed(29);g=torch.randn(2,150,153);g[...,144:152]=1
    g[:,25:40,144:152]=0;g[1,...,144:152]=0
    return g,torch.randn(2,150,512)

def nonzero_adapter():
    torch.manual_seed(13);model=EventMusicAdapter().eval()
    torch.nn.init.normal_(model.output.weight,std=.01);return model

def test_default_is_exact_legacy_forward_and_state_keys_unchanged():
    a=nonzero_adapter();g,m=fixture_geometry()
    keys=list(a.state_dict())
    with torch.no_grad():
        event=a.encoder(a.input(g));musical=a.music(m)
        attended=a.attention(musical,event,event,need_weights=False)[0]
        expected=a.output(a.norm(event+musical+attended))
        actual=a(g,m)
    assert torch.equal(actual,expected)
    assert list(a.state_dict())==keys
    with torch.no_grad():assert torch.equal(actual,a(g,m,visibility_policy="legacy"))

def test_observed_visibility_safe_all_missing_and_unknown_storage_invariance():
    a=nonzero_adapter();g,m=fixture_geometry()
    damaged=intervene(g,"unknown_corrupted",12)
    damaged[:,:,:144]=torch.where((damaged[...,144:152]>0).repeat_interleave(6,-1).repeat(1,1,3),damaged[:,:,:144],float("nan"))
    y=a(g,m,visibility_policy="observed");z=a(damaged,m,visibility_policy="observed")
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    assert torch.isfinite(y).all() and not y[1].any()
    assert not a(intervene(g,"removed"),m,visibility_policy="observed").any()
    with pytest.raises(ValueError,match="Binary"):
        bad=g.clone();bad[...,144]=.5;a(bad,m,visibility_policy="observed")

def test_masked_storage_jacobian_zero_and_partial_observed_gradient_finite():
    a=nonzero_adapter();a.requires_grad_(False);g,m=fixture_geometry();g.requires_grad_()
    a(g,m,visibility_policy="observed").square().sum().backward()
    known=(g[...,144:152]>0).repeat_interleave(6,-1).repeat(1,1,3)
    assert torch.isfinite(g.grad).all()
    assert not g.grad[...,:144][~known].any()
    assert g.grad[0,:,:144][known[0]].abs().sum()>0

def test_real_dense_windows_require_actual_audio_and_calibrated_support():
    assert real_starts(360,360,0,11.989375,15)==list(range(0,211,15))
    assert real_starts(360,300,0,11.989375,15)==list(range(0,151,15))
    assert real_starts(149,900,0,30,15)==[]
    assert real_starts(360,900,-30,12,15)[0]==30
    with pytest.raises(ValueError,match="integer"):real_starts(360,360,.5,12,15)
    with pytest.raises(ValueError,match="stride"):real_starts(360,360,0,12,0)
    with pytest.raises(ValueError,match="extrapolation"):sample_tokens(np.zeros((60,128)),[119/30],15)

def test_fast_candidate_solver_preserves_original_strict_rules():
    rows=[dict(role="train",source_group="g",recording_uid=str(i),music_uid="m",choreography_uid="c",start_frame=i%2*150) for i in range(4)]
    bank=dict(mv=np.ones((4,3),bool),dv=np.ones((4,3),bool))
    stats=dict(counts=np.ones((4,2),int)*3,quantiles=np.ones((4,2,16)),hist=np.ones((4,2,4))*.25,energy=np.ones((4,2)),onset_density=np.ones(4))
    cfg=json.loads(Path("configs/local_identifiability.json").read_text())["strict"]
    a,e=build_candidates(rows,bank,stats,cfg);b,f=fast_candidates(rows,bank,stats,cfg)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert e["populations"]==f["populations"] and not f["populations"]["train"]["population_sufficient"]
    stats["energy"][1::2]*=100
    b,f=fast_candidates(rows,bank,stats,cfg);assert not b["eligible"].any() and not f["rules_relaxed"]

def test_contract_binds_source_clock_observations_and_shadow(tmp_path):
    data=np.zeros((150,151),np.float32);mask=np.zeros_like(data,bool);mask[:,79:127]=True
    p=tmp_path/"motion.npy";q=tmp_path/"mask.npy";np.save(p,data);np.save(q,mask)
    ref=lambda p:dict(path=str(p),sha256=sha(p))
    e=dict(event_uid="e",source_uid="source",reference=ref(p),source_interval=[0,150],music_interval=[0,150],time_warp=[[0,0],[150,150]],
        primitive=dict(source_frame=75,music_frame=75,joints=list(range(12,20)),radius=12),feasibility=dict(native="UNKNOWN"))
    plan=dict(schema="event_plan_v1",fps=30,frames=150,audio_sha256="x",time_scale_limits=[.75,1.33],events=[e])
    c=build(plan,{"source":ref(q)},beat_frames=[31,71,111]);assert verify(c,plan) is c
    assert c["can_route"] is False and len(c["events"][0]["key_motion"])==3
    assert not c["events"][0]["root_contact"]["root_observed"]
    for mutate in ("source","time","joint","boundary","expression"):
        bad=copy.deepcopy(c);v=bad["events"][0]
        if mutate=="source":v["source_uid"]="other"
        elif mutate=="time":v["key_motion"][0]["music_frame"]=-1
        elif mutate=="joint":v["key_motion"][0]["observed_joints"]=[0]
        elif mutate=="boundary":v["boundary"]["left_source_interval"]=[0,50]
        else:v["expression"]["learned"]=True
        bad["semantic_digest"]=identity(bad)
        with pytest.raises(ValueError):verify(bad,plan)
