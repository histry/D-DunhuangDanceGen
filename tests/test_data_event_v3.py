import copy,json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.data.split_registry import components,partition,attest,leave_one_recording_out,checkpoint_binding,training_population,catalog_subset
from dunhuang.data.single_person_admission import admit,require_qualified_sources
from dunhuang.data.change_event_phases import motion_profiles,detect,phases_for_core,choose_anchor
from dunhuang.data.change_event_audit import audit_event,slot_eligibility
from dunhuang.research.data_event_v3.registry import official_finedance

PHASES=dict(minimum_peak_distance=6,speed_prominence_floor_radps=.05,acceleration_prominence_floor_radps2=1.,hold_quantile=.2,hold_max_speed_radps=.2,hold_min_frames=6)

def rows(n=8):
    return [dict(source_uid="s"+str(i),recording_uid="r"+str(i),source_sha256="h"+str(i),audio_sha256="a"+str(i),
        music_work_uid="w"+str(i),performer_uid="same_performer",choreography_uid="c"+str(i),genre="g") for i in range(n)]

def native(n=120):
    x=np.zeros((n,151),np.float32);x[:,7:]=np.tile(np.array([1,0,0,0,1,0],np.float32),(n,24));return x

def add_step(x,frame,joint,angle):
    x[frame:,7+6*joint:13+6*joint]=sixd(Rotation.from_euler("x",angle).as_matrix());return x

def test_task_graph_does_not_merge_all_music_via_shared_performer():
    r=rows()
    assert len(components(r,"strict_unseen_song"))==8
    assert len(components(r,"unseen_performer"))==1
    r[1]["music_work_aliases"]=["w0"]
    assert len(components(r,"strict_unseen_song"))==7

def test_roles_are_grouped_reproducible_and_missing_id_is_never_random_split():
    r=rows(30);a,p=partition(r,"strict_unseen_song",17);b,q=partition(r,"strict_unseen_song",17)
    assert a==b and p==q and p["forbidden_identity_overlaps"]==0
    assert p["role_music"]==dict(train=24,val=3,test=3)
    r[0]["music_work_uid"]=None
    with pytest.raises(ValueError,match="UNKNOWN"):partition(r,"strict_unseen_song",17)

def test_official_crossmodal_allows_only_declared_val_test_music_sharing():
    r=rows(3);roles={"s0":"train","s1":"val","s2":"test"}
    r[2]["music_work_uid"]=r[1]["music_work_uid"];r[2]["audio_sha256"]=r[1]["audio_sha256"]
    assert attest(r,roles,"official_crossmodal")["forbidden_identity_overlaps"]==0
    assert attest(r,roles,"strict_unseen_song")["forbidden_identity_overlaps"]>0
    r[2]["recording_uid"]=r[1]["recording_uid"]
    assert attest(r,roles,"official_crossmodal")["forbidden_identity_overlaps"]>0

def test_recording_and_raw_aliases_cannot_cross_song_split():
    r=rows(10);r[1]["recording_uid"]=r[0]["recording_uid"]
    a,p=partition(r,"strict_unseen_song",13)
    assert a["s0"]==a["s1"]
    r[2]["source_sha256"]=r[3]["source_sha256"]
    a,p=partition(r,"strict_unseen_song",13)
    assert a["s2"]==a["s3"] and not p["forbidden_identity_overlaps"]

def test_leave_one_source_out_requires_fresh_index_and_statistics():
    r=rows();folds=leave_one_recording_out(r,19)
    assert len(folds)==8
    for f in folds:
        assert sorted(f["roles"].values())==["test"]+["train"]*6+["val"]
        assert f["index_rebuild_required"] and f["train_only_statistics_required"]
        assert not f["existing_full_catalog_checkpoint_qualified"]
        catalog={"entries":[{"source_uid":x["source_uid"],"event_uid":str(i)} for i,x in enumerate(r)]}
        assert len(catalog_subset(catalog,f["roles"])["entries"])==6

def test_single_track_array_is_not_evidence_of_solo_capture():
    s=dict(dataset_id="finedance",performer_count=None,single_track_array=True,solo_verified=False,observed_joints=list(range(22)),pairing_status="UNKNOWN")
    assert not admit(s)["solo_admitted"]
    with pytest.raises(ValueError):require_qualified_sources([s],paired=True)
    s.update(dataset_id="change",performer_count=2,solo_verified=True,solo_evidence={"metadata":"x"})
    assert admit(s)["solo_status"]=="EXCLUDED_MULTI_PERFORMER"
    s["performer_count"]=1
    assert admit(s)["source_only_catalog_eligible"] and not admit(s)["full151_target_ready"]

def test_new_generation_split_requires_explicit_recording_roles():
    from dunhuang.generation.split import split_training_rows
    w=[("a",0),("a",75),("z",0)]
    with pytest.raises(ValueError,match="explicit"):split_training_rows(w,"motion",{})
    train,val,groups=split_training_rows(w,"motion",{"a":"val","z":"train"})
    assert train==[("z",0)] and groups==["a"] and len(val)==2

def test_checkpoint_population_is_bound_and_audit_registry_cannot_start_training():
    registry=dict(identity_digest="d",sources=rows(),protocols={"p":dict(roles={"s0":"train","s1":"val"},identity_scope="song",ready_for_new_training=False)})
    with pytest.raises(ValueError,match="NOT_EVALUABLE"):training_population(registry,"p")
    assert checkpoint_binding(registry,"p",["s0"])
    with pytest.raises(ValueError):checkpoint_binding(registry,"p",["s0","s1"])

def test_controlled_peak_ignores_wrist_spike_and_never_clamps_to_midpoint():
    x=add_step(add_step(native(),20,20,1.),70,12,.3);profiles=motion_profiles(x,list(range(22)))
    phases=detect(profiles,PHASES)
    old=dict(event_uid="e",source_uid="s",recording_uid="r",source_sha256="raw",source_interval=[0,100],
        center=35,reference={"path":"unused","sha256":"x"},feasibility={"temporal8":"PASS"})
    e,d=audit_event(old,x,profiles,phases,list(range(22)),np.ones(151),np.zeros(151),"catalog",{"discovery_dependency":{"native_start_frame":0,"native_end_frame_exclusive":len(x)}})
    assert d["original_peak"]==20 and d["controlled_peak"]==70
    assert d["old_peak_clamped"] and d["joint_set_peak_changed"]
    assert e["safe_anchor"]["frame"]==70 and e["event_uid"]=="e" and e["source_interval"]==[0,100]
    assert not d["e0_captures_controlled_peak"] and d["e1_captures_controlled_peak"]
    assert e["discovery_read_support"]==[0,len(x)] and not e["can_route"]

def test_peak_without_real_context_is_rejected_not_relocated():
    p=[dict(frame=1,type="controlled_core_maximum",observed_joints=list(range(12,20)),score=None)]
    assert choose_anchor(p,1,0,45,list(range(22)),np.ones(45)) is None

def test_normalizer_does_not_choose_or_move_the_anchor():
    x=add_step(native(),70,12,.3);p=motion_profiles(x,list(range(22)));ph=detect(p,PHASES)
    old=dict(event_uid="e",source_uid="s",recording_uid="r",source_sha256="raw",source_interval=[0,100],center=65,
        reference={"path":"unused","sha256":"x"},feasibility={"temporal8":"FAIL"})
    e,d=audit_event(old,x,p,ph,list(range(22)),np.ones(151)*2,np.zeros(151),"c",{"discovery_dependency":{}})
    assert e["safe_anchor"]["frame"]==70 and not e["normalization_support"]
    entry=dict(source_interval=[0,100],center=70,normalization_support=False)
    assert slot_eligibility(entry,dict(start=0,end=100),[],"temporal")[1]=="NORMALIZER_SUPPORT"
    assert slot_eligibility(entry,dict(start=0,end=100),[],"adapter")[0]

def test_unknown_rotation_and_root_storage_cannot_create_features():
    x=native();x[:,139:]=np.nan;x[:,4:7]=np.nan
    p=motion_profiles(x,list(range(22)))
    assert np.isfinite(p["upper8"]).all() and "root_horizontal_speed_mps" not in p
    with pytest.raises(ValueError,match="Root"):motion_profiles(x,list(range(22)),np.zeros((len(x),24,3)))

def test_whole_record_derivative_is_not_a_new_core_origin():
    x=add_step(native(),50,12,.5)
    full=motion_profiles(x,list(range(22)));cut=motion_profiles(x[50:80],list(range(22)))
    assert full["upper8"][50]>0 and cut["upper8"][0]==0

def test_constant_fast_motion_is_not_a_pose_hold_and_keys_are_not_forced():
    p=dict(upper8=np.ones(100)*5,upper_direction_change=np.zeros(100),root_yaw_radps=np.zeros(100))
    assert not any(e["type"]=="upper_pose_hold" for e in detect(p,PHASES))

def test_hold_proposal_records_actual_support_and_no_semantic_truth():
    p=dict(upper8=np.zeros(100),upper_direction_change=np.zeros(100),root_yaw_radps=np.zeros(100))
    events=detect(p,PHASES)
    assert len(events)==1 and events[0]["evidence_span"]==[0,100] and events[0]["semantic_status"]=="UNKNOWN"
    clipped=phases_for_core(events,0,45,1)
    assert clipped[0]["frame"]==1
    hold=next(e for e in clipped if e["type"]=="upper_pose_hold")
    assert hold["frame"]==22 and hold["evidence_truncated_by_core"]

def test_slots_keep_duration_and_local_rate_limits_and_actual_time_map():
    e=dict(source_interval=[100,190],center=145,normalization_support=True)
    ok,reason,warp=slot_eligibility(e,dict(start=0,end=90),[]);assert ok
    from dunhuang.planning.time_map import source_positions
    assert source_positions(dict(source_interval=e["source_interval"],music_interval=[0,90],time_warp=warp),np.array([45]))[0]==145
    assert slot_eligibility(dict(e,source_interval=[0,45],center=22),dict(start=0,end=90),[])[1]=="DURATION_RATIO"
    assert slot_eligibility(dict(e,source_interval=[0,90],center=12),dict(start=0,end=90),[])[1]=="ZERO_OR_NONMONOTONE_WARP"

def test_official_split_parser_never_executes_imports_and_applies_ignore(tmp_path):
    text='raise RuntimeError("must not execute")\n'+'def get_train_test_list(datasplit):\n if datasplit=="cross_genre":\n  test_list=["120","001"]\n  ignor_list=["120"]+["130"]\n elif datasplit=="cross_dancer":\n  test_list=["002"]\n  ignor_list=["003"]\n'
    p=tmp_path/"official.py";p.write_text(text)
    result=official_finedance(p)
    assert result["cross_genre"]["test"]=={"001"} and "120" not in result["cross_genre"]["train"]
    assert result["cross_dancer"]["test"]=={"002"}
