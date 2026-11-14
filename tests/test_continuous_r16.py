import copy,json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd,matrices,MotionAdapter
from dunhuang.research.continuous_r16.contract import views,plan_grid,continuous_pair,observations,patch_control,align_donor
from dunhuang.research.continuous_r16.gate import qualification

def cfg():return json.loads((Path(__file__).parents[1]/"src/dunhuang/research/continuous_r16/config.json").read_text())

def native():
    x=np.zeros((225,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],(225,24))
    x[:,4]=np.arange(225)*.01;x[:,5]=1
    return x

def test_one_global_clock_across_two_compute_windows():
    p=dict(frames=225,events=[
        dict(music_interval=[0,80],source_interval=[10,90],time_warp=[[10,0],[90,80]]),
        dict(music_interval=[80,225],source_interval=[100,245],time_warp=[[100,80],[245,225]])])
    g=plan_grid(p);v=views(np.c_[g["event_ids"],g["source_positions"],g["phase"]])
    assert np.array_equal(v[0,75:],v[1,:75])
    assert v[1,5,0]==1 and v[1,5,1]==100
    p["events"][1]["music_interval"][0]=81
    with pytest.raises(ValueError):plan_grid(p)

def test_pair_requires_same_capture_source_role_and_clock():
    l=dict(recording_uid="capture",source_group="music",role="train",start_frame=75,
        **{k:{"sha256":k} for k in ("raw_motion","source_native","audio","clock")})
    r=copy.deepcopy(l);r["start_frame"]=150;assert continuous_pair(l,r)
    for key in ("recording_uid","role","source_group"):
        bad=copy.deepcopy(r);bad[key]="other"
        with pytest.raises(ValueError):continuous_pair(l,bad)
    for key in ("raw_motion","source_native","audio","clock"):
        bad=copy.deepcopy(r);bad[key]["sha256"]="different"
        with pytest.raises(ValueError):continuous_pair(l,bad)
    bad=copy.deepcopy(r);bad["start_frame"]=151
    with pytest.raises(ValueError):continuous_pair(l,bad)

def test_hidden_target_poison_never_enters_source_observations():
    x=native()[:150];v,k,edge,event,_=observations(x,75,list(range(12,20)))
    poison=x.copy();poison[~k]=np.nan
    vp,kp,_,_,_=observations(poison,75,list(range(12,20)))
    assert np.array_equal(v,vp) and np.array_equal(k,kp)
    assert not k[:,:4].any() and not event[:,:79].any() and not event[:,127:].any()
    assert not np.logical_and(edge,event).any()

def test_source_controls_keep_true_context_and_do_not_expose_padding():
    v,k,edge,event,_=observations(native()[:150],75,list(range(12,20)))
    wrong=np.full_like(v,3.)
    for arm in ("C1_correct","C1_removed","C1_wrong_event","C1_wrong_context","C1_wrong_music"):
        out,known=patch_control(v,k,edge,event,wrong,arm)
        assert not out[~known].any()
        if arm!="C1_wrong_context":assert np.array_equal(out[edge],v[edge])
        if arm=="C1_removed":assert not known[event].any()
        if arm=="C1_wrong_event":assert np.array_equal(out[event],wrong[event])

def test_shared_motion_root_is_not_independently_recentered():
    x=native();o=x[0,4:7].copy();o[1]=0
    z=MotionAdapter.to_edge(x,o);v=views(z)
    assert np.array_equal(v[0,75:],v[1,:75])
    assert v[1,0,4]!=0
    assert np.allclose(MotionAdapter.to_native(z,o)[:,4:7],x[:,4:7])

def test_donor_root_uses_observed_recipient_gauge():
    x=native()[:150];d=x.copy();R=Rotation.from_euler("y",.7).as_matrix()
    d[:,4:7]=x[:,4:7]@R.T+[10,0,-4];d[:,7:13]=sixd(R@matrices(x[:,7:13]))
    a=align_donor(d,x)
    assert np.allclose(a[:,4:7],x[:,4:7],atol=2e-6)
    assert np.allclose(matrices(a[:,7:13]),matrices(x[:,7:13]),atol=2e-6)

def test_new_outer_seam_failure_cannot_be_hidden_by_old_center_improvement():
    c=cfg();records=[]
    for i in range(c["cases"]):
        for arm in c["arms"]:
            records.append(dict(case=i,arm=arm,event_pose_error_rad=.1 if arm=="C1_correct" else .3,
                edge_root_rmse_m=.01 if arm=="C1_correct" else .1,center_jerk_max_mps3=500 if arm=="C1_correct" else 2000,
                common_generated_span_pose_rad=.2,activity_ratio=1.,
                external_joins=[dict(root_step_m=.01,foot_step_max_m=.02,crossing_jerk_max_mps3=500)]*2))
    assert qualification(records,c,1620)["cross_window_training_admitted"]
    for r in records:
        if r["arm"]=="C1_correct":r["external_joins"]=[dict(root_step_m=.4,foot_step_max_m=.02,crossing_jerk_max_mps3=500)]*2
    assert not qualification(records,c,1620)["cross_window_training_admitted"]
    with pytest.raises(ValueError):qualification(records[:-1],c,1620)
