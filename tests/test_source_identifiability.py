import copy,ast
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd,MotionAdapter
from dunhuang.alignment.audio_clock import music_seconds,music_sample,same_music_time

def native():
    x=np.zeros((150,151),np.float32)
    r=Rotation.from_euler("xyz",[.4,.7,-.3]).as_matrix()
    x[:,7:]=np.tile(sixd(r),(150,24))
    return x

def test_audible_clock_unifies_distinct_motion_origins_and_detects_false_alias():
    a=dict(start_frame=30,fps=30,audio_offset_seconds=0,music_uid="m",audio={"sha256":"a"})
    b=dict(a,start_frame=0,audio_offset_seconds=1)
    assert same_music_time(a,b) and music_sample(a)==44100
    assert not same_music_time(a,dict(a,audio_offset_seconds=.001))
    with pytest.raises(ValueError):music_seconds(dict(a,audio_offset_seconds=float("nan")))
    with pytest.raises(ValueError):music_seconds(dict(a,start_frame=-1))

def test_generation_package_does_not_import_legacy_trainers():
    for p in Path("src/dunhuang/generation").glob("*.py"):
        s=p.read_text(encoding="utf-8")
        assert "research.legacy" not in s
    assert not Path("src/dunhuang/generation/train.py").exists()
    assert not Path("scripts/event_causal_audit.py").exists()

def test_so3_following_respects_edge_rows_native_columns_and_gradients():
    torch=pytest.importorskip("torch")
    from dunhuang.generation.source_following import following_loss
    x=native();target=torch.tensor(MotionAdapter.to_edge(x)[None],dtype=torch.float64,requires_grad=True)
    ref=torch.tensor(x[None],dtype=torch.float64);available=torch.ones_like(ref,dtype=torch.bool)
    keys=torch.zeros(1,150,8,dtype=torch.bool);keys[:,75]=True
    scale=torch.ones(151,dtype=torch.float64);bias=torch.zeros(151,dtype=torch.float64)
    exact=following_loss(target,ref,available,keys,scale,bias,"same_recording_pose_exact")
    assert exact.item()<1e-10
    wrong=ref.clone();wrong[:,:,79:127]=torch.tensor(np.tile(sixd(Rotation.from_euler("z",1.2).as_matrix()),(150,8))[None])
    loss=following_loss(target,wrong,available,keys,scale,bias,"same_recording_pose_exact");loss.backward()
    assert loss>.1 and target.grad.isfinite().all() and target.grad[:,:,79:127].abs().sum()>0
    assert not target.grad[:,:,:79].any() and not target.grad[:,:,127:].any()
    with pytest.raises(ValueError,match="pose-exact"):following_loss(target,ref,available,keys,scale,bias,"cross_recording_weak")

def test_unknown_reference_channels_cannot_enter_following_loss():
    torch=pytest.importorskip("torch")
    from dunhuang.generation.source_following import following_loss
    x=native();pred=torch.tensor(MotionAdapter.to_edge(x)[None])
    ref=torch.tensor(x[None]);mask=torch.zeros_like(ref,dtype=torch.bool);mask[:,:,79:127]=True
    keys=torch.zeros(1,150,8,dtype=torch.bool);keys[:,75]=True
    ref[:,:,:79]=float("nan");ref[:,:,127:]=float("nan")
    value=following_loss(pred,ref,mask,keys,torch.ones(151),torch.zeros(151),"same_recording_pose_exact")
    assert value.isfinite() and value<1e-9
    with pytest.raises(ValueError,match="Missing"):following_loss(pred,ref,mask,keys&False,torch.ones(151),torch.zeros(151),"same_recording_pose_exact")

def test_interpolation_never_reads_hidden_source_truth():
    pytest.importorskip("torch")
    from dunhuang.generation.source_following import interpolate_observed
    x=native();known=np.zeros_like(x,bool);known[[0,149],79:127]=True
    end=Rotation.from_euler("xyz",[1.,-.2,.4]).as_matrix();x[149,79:127]=np.tile(sixd(end),8)
    base=native();base[:,4]=np.arange(150)/30
    a=interpolate_observed(base,x,known);x[1:149,79:127]=np.nan;b=interpolate_observed(base,x,known)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,:79],base[:,:79]);np.testing.assert_array_equal(a[:,127:],base[:,127:])
    np.testing.assert_allclose(a[[0,149],79:127],x[[0,149],79:127],atol=1e-6)

def test_canonical_contract_is_consumed_at_the_actual_warp(tmp_path):
    pytest.importorskip("torch")
    from dunhuang.generation.source_following import consume
    from dunhuang.planning.primitive_contract import build
    from dunhuang.io import sha
    from dunhuang.planning.time_map import source_positions
    x=native();mask=np.zeros_like(x,bool);mask[:,79:127]=True
    p=tmp_path/"native.npy";q=tmp_path/"known.npy";np.save(p,x);np.save(q,mask)
    ref=lambda p:dict(path=str(p),sha256=sha(p))
    e=dict(event_uid="e",source_uid="s",reference=ref(p),source_interval=[0,150],music_interval=[0,150],time_warp=[[0,0],[75,65],[150,150]],
        primitive=dict(source_frame=75,music_frame=65,joints=[12,19],radius=12),feasibility=dict(native="UNKNOWN"))
    plan=dict(schema="event_plan_v1",fps=30,frames=150,audio_sha256="a",time_scale_limits=[.75,1.33],events=[e])
    c=build(plan,{"s":ref(q)},beat_frames=[30,120]);value,keys=consume(plan,c,"dense_upper8")
    assert keys[0,65,0] and keys[0,65,7] and not value.known[:,:,:79].any()
    np.testing.assert_allclose(value.phase.numpy()[0,:,0],source_positions(e,np.arange(150))/150,atol=1e-7)
    with pytest.raises(ValueError):consume(plan,dict(c,can_route=True),"dense_upper8")

def test_v2_candidates_are_new_data_associations_with_inherited_roles():
    pytest.importorskip("torch")
    from dunhuang.research.source_identifiability.a2 import candidates
    cfg=dict(count_tolerance=0,time_w1_seconds=.15,histogram_l1=.25,energy_log_ratio=np.log(1.25),onset_density_per_second=.25,min_queries=30,min_recordings=15,min_train_components=3,min_val_components=2)
    rows=[dict(role="train",source_group=str(i),recording_uid=str(i),music_uid=str(i),choreography_uid=str(i),start_frame=0,audio_offset_seconds=0,fps=30,audio={"sha256":str(i)}) for i in range(4)]
    b=dict(mv=np.ones((4,2),bool),dv=np.ones((4,2),bool))
    s=dict(counts=np.ones((4,2),int)*2,quantiles=np.ones((4,2,16)),hist=np.ones((4,2,4))*.25,energy=np.ones((4,2)),onset_density=np.ones(4))
    c,e=candidates(rows,b,s,cfg)
    __import__("json").dumps(e,allow_nan=False)
    assert c["eligible"].all() and (c["positive"].sum(-1)==1).all() and not e["populations"]["train"]["population_sufficient"]
    rows[1]["audio_offset_seconds"]=1
    c,e=candidates(rows,b,s,cfg)
    assert not c["eligible"][1]
    rows[0]["role"]="test"
    with pytest.raises(ValueError):candidates(rows,b,s,cfg)

def test_hard_control_only_observes_supported_upper8_key_values():
    torch=pytest.importorskip("torch")
    from dunhuang.generation.source_following import hard_keyframe_inputs
    x=torch.full((1,150,151),float("nan"));x[:,75,79:85]=1
    keys=torch.zeros(1,150,8,dtype=torch.bool);keys[:,75,0]=True
    value,mask=hard_keyframe_inputs(x,keys)
    assert mask.sum()==6 and value.isfinite().all()
    assert value[:,75,79:85].eq(1).all() and not value[~mask].any()
    assert not mask[:,:,:79].any() and not mask[:,:,127:].any()

def test_a2_small_source_audit_completes_without_starting_ineligible_training(tmp_path):
    pytest.importorskip("torch")
    import json
    from dunhuang.io import write,sha
    from dunhuang.research.source_identifiability.a2 import run
    source=tmp_path/"study/a1";data=source/"data";data.mkdir(parents=True)
    event=tmp_path/"events/data/sidecars";event.mkdir(parents=True)
    cache=tmp_path/"clock/data/audio";cache.mkdir(parents=True)
    rows=[];seals={}
    for i,role in enumerate(("train","val")):
        p=data/(str(i)+".npy");x=native();x[:,4]=i;np.save(p,x)
        sidecar=event/(str(i)+".json");write(sidecar,{"events":[]});seals["data/sidecars/"+sidecar.name]=sha(sidecar)
        np.savez(cache/(str(i)+".npz"),values=np.ones((150,50),np.float32))
        rows.append(dict(role=role,window_uid=str(i),source_group=str(i),recording_uid=str(i),music_uid=str(i),choreography_uid=str(i),
            start_frame=0,audio_offset_seconds=0.,fps=30,audio={"sha256":str(i)},raw_motion={"sha256":str(i)},source_native={"path":str(p),"sha256":sha(p)}))
    write(event.parent.parent/"preflight.json",{"files":seals})
    write(data/"manifest.json",dict(rows=rows,source_population={"path":str(event.parent/"manifest.json")},source_clock_manifest={"path":str(cache.parent/"manifest.json")}))
    np.savez(data/"nodes.npz",music=np.ones((2,12,144),np.float32),motion=np.ones((2,20,144),np.float32),mv=np.ones((2,12),bool),dv=np.ones((2,20),bool),
        mt=np.tile(np.arange(12)/3,(2,1)),dt=np.tile(np.arange(20)/5,(2,1)))
    np.savez(data/"conditional.npz",eligible=np.zeros(2,bool))
    write(data/"candidate_evidence.json",{"evidence":[]})
    write(source.parent/"public_summary.json",{"evidence":{str(p.relative_to(source.parent)):sha(p) for p in (data/"manifest.json",data/"nodes.npz",data/"conditional.npz")}})
    cfg=json.loads(Path("configs/content_identifiability_v2.json").read_text())
    cfg["event_source"]=str(event.parent.parent)
    out=tmp_path/"audit"
    run(source,out,cfg)
    result=json.loads((out/"result.json").read_text())
    assert result["training"]["optimizer_updates"]==0
    assert result["training"]["status"]=="NOT_STARTED_INSUFFICIENT_V2_POPULATION"
    assert result["clock"]["old_positive_actual_clock_disagreements"]==0
