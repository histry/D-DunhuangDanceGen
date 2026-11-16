import copy
import numpy as np
import pytest
from dunhuang.research.transition_r35.graph import zones,pool,descriptor,edges,search,path_cost

def event(uid,source="s",score=0.):
    return dict(event_uid=uid,candidate_uid=uid,source_uid=source,scores=dict(heuristic=score),music_interval=[0,80],primitive=dict(music_frame=40))

def test_protected_core_and_editable_buffers_never_overlap():
    z=zones(event("x"));assert z["protected_core"]==[28,53]
    assert z["entry_edit"]==[0,12] and z["exit_edit"]==[68,80]
    e=event("y");e["primitive"]["music_frame"]=12
    assert zones(e)["entry_edit"]==[0,0]
    e["primitive"]["music_frame"]=0
    with pytest.raises(ValueError):zones(e)

def test_multiple_events_of_one_source_survive_music_pool():
    a=event("a");b=event("b",score=-.01);c=event("c",score=-.04)
    cfg=dict(music_slot_regret_max=.05,per_source_candidates=3)
    assert len(pool([a,b,c],a,cfg))==3

def test_boundary_descriptor_uses_velocity_and_wrists():
    R=np.tile(np.eye(3),(12,22,1,1));p=np.zeros((12,22,3));a=descriptor(R,p)
    q=p.copy();q[:,20,0]=np.arange(12)/30;b=descriptor(R,q)
    m=edges([a],[b],[event("a")],[event("b")])
    assert m["velocity_gap_m_s"][0,0]>0 and m["wrist_gap_m"][0,0]==0
    assert m["editable_frames"][0,0]==24

def test_search_keeps_baseline_and_rejects_music_budget_violation():
    a,b=event("a"),event("b","t");c,d=event("c",score=-.05),event("d","t",score=-.05)
    banks=[[a,c],[b,d]];cfg=dict(beam=8,music_mean_regret_max=.02,transition_weight=.25)
    mats=[dict(cost=np.array([[10,10],[10,0.]]))]
    cost,path,regret=search(banks,mats,[a,b],cfg)
    assert path==[0,0] and regret==0
    assert cost==path_cost(path,banks,mats,cfg)

def test_mismatched_or_nonfinite_body_is_rejected():
    with pytest.raises(ValueError):descriptor(np.zeros((2,22,3,3)),np.zeros((2,22,3)))
