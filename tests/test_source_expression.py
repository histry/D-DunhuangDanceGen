import copy
import numpy as np
import pytest
from dunhuang.research.source_expression.review_contract import summarize,validate,VERSION,time_valid
from dunhuang.research.source_expression.losses import scales

def binding(task="EVENT"):
    return dict(blind_item_id="opaque",task_type=task,source_uid="private",raw_motion_sha256="b"*64,event_uid="private_event",start_frame=10,end_frame=100,fps=30,media_sha256="a"*64)
def record(who="r1",decision="VALID",task="EVENT"):
    return dict(review_id=who,blind_item_id="opaque",reviewer_id=who,reviewer_roster_ref="roster",task_type=task,media_sha256="a"*64,reviewed_at="2026-11-13T12:00:00+08:00",decision=decision,confidence="LOW",reason_code="VISIBILITY_INSUFFICIENT" if decision=="UNSURE" else "JUDGMENT_SUPPORTED",review_protocol_version=VERSION,model_predictions_shown=False,label_origin="independent_human_review")
def roster():
    return {r:dict(identity_verified=True,independent_attested=True,private_identity_ref="private/"+r) for r in ("r1","r2","r3")}
def test_review_empty_and_partial_never_truth():
    s,d,a=summarize([binding()],[],{},"roster");assert s["states"]=={"UNREVIEWED":1} and not s["data_review_verified"]
    s,d,a=summarize([binding()],[record()],roster(),"roster");assert s["states"]=={"PARTIAL_REVIEW":1}
def test_review_independence_and_unsure_preserved():
    rs=[record("r1","UNSURE"),record("r2","UNSURE")]
    s,d,a=summarize([binding()],rs,roster(),"roster")
    assert len(a)==2 and s["states"]=={"UNSURE":1} and s["unsure_reasons"]=={"VISIBILITY_INSUFFICIENT":2}
    with pytest.raises(ValueError):summarize([binding()],[record(),dict(record(),review_id="other")],roster(),"roster")
@pytest.mark.parametrize("field,value",[("media_sha256","c"*64),("reviewed_at","2026-11-13T12:00:00"),("label_origin","automatic"),("model_predictions_shown",True),("reviewer_roster_ref","wrong"),("source_uid","wrong"),("decision","COMPATIBLE")])
def test_review_rejects_broken_identity_and_automatic_truth(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError):validate(r,binding(),roster(),"roster")
def test_review_disagreement_requires_third_original_vote():
    rs=[record(),record("r2","INVALID")]
    s,_,_=summarize([binding()],rs,roster(),"roster");assert s["states"]=={"PARTIAL_REVIEW":1}
    rs.append(record("r3","UNSURE"));adj=[dict(blind_item_id="opaque",original_review_ids=["r1","r2"],adjudicator_review_id="r3")]
    s,d,a=summarize([binding()],rs,roster(),"roster",adj)
    assert len(a)==3 and s["initial_disagreements"]==1 and d[0]["decision"]=="UNSURE"
    adj[0]["adjudicator_review_id"]="r1"
    with pytest.raises(ValueError):summarize([binding()],rs,roster(),"roster",adj)
def test_music_compatibility_never_crossmodal_event_truth():
    s,d,a=summarize([binding("MUSIC_MATCH")],[record("r1","COMPATIBLE","MUSIC_MATCH"),record("r2","COMPATIBLE","MUSIC_MATCH")],roster(),"roster")
    assert s["data_review_verified"] and s["crossmodal_event_correspondence"]=="NOT_EVALUABLE" and not s["operational_qualified"]
def test_train_scales_never_derivative_across_gaps():
    from scipy.spatial.transform import Rotation
    safe=np.tile([1.,0,0,0,1,0],(150,8,1));known=np.ones((150,8),bool);known[40:80]=False
    rr=Rotation.from_euler("z",1.).as_matrix();safe[80:]=np.r_[rr[:,0],rr[:,1]]
    cfg=dict(pose_floor_squared_rad=1e-4,velocity_floor_squared_radps=1e-4)
    a=scales([(safe,known,known,np.ones_like(known))],cfg)
    assert a["velocity"]==1e-4 and a["adjacent_samples"]==(39+69)*8
def test_source_loss_unknown_poison_and_equal_dose():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from dunhuang.research.source_expression.losses import reference_data,components,equalize
    safe=np.tile(np.array([1.,0,0,0,1,0],np.float32),(150,8,1));known=np.ones((150,8),bool);known[30:40]=False;stage=known.copy()
    first=reference_data((safe,known,stage,stage),123,device="cpu")
    raw=torch.tensor(np.tile([1.,0,0,0,1,0],(1,150,24)).reshape(1,150,144),dtype=torch.float32)
    pred=torch.cat([torch.zeros(1,150,7),raw],-1);rt=SimpleNamespace(bias=np.zeros(151,np.float32),scale=np.ones(151,np.float32))
    values=components(rt,pred,first)
    pred[:,30:40,79:127]=torch.randn(1,10,48)*10
    other=components(rt,pred,first)
    for key in values:assert torch.equal(values[key],other[key])
    ds={"a":torch.ones(100),"b":torch.ones(100)*.1};matched,norm=equalize(ds)
    assert all(float(d.double().norm())<=float(ds[k].double().norm())+1e-7 for k,d in matched.items())
    assert max(abs(float(d.double().norm())-norm) for d in matched.values())<=2*np.finfo(np.float32).eps*norm
def test_clip_mask_boundary_and_responsibility():
    torch=pytest.importorskip("torch")
    from dunhuang.research.source_expression.diagnose import clip_mask,responsibility
    x=torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True)
    assert torch.equal(torch.autograd.grad(x.clamp(-1,1).sum(),x)[0].bool(),clip_mask(x))
    assert responsibility(dict(own_gain_rad=.01,wrong_gain_rad=.0098,directional_gain_rad=.0002,protection={"protected":True}))==["COMMON_MODE_GAIN"]

def test_blind_media_fits_entire_skeleton_and_audio_clock():
    from dunhuang.research.source_expression.review import page
    points=np.zeros((150,22,3));points[:,10,1]=-1.2;points[:,20,1]=1.4
    html=page(points,"MUSIC_MATCH","opaque.wav")
    assert "210/radius" in html and "250-scale*p[1]" in html and "audio.currentTime*30" in html
    assert "source_uid" not in html and "event_uid" not in html
    # The spherical fit guarantees the full clip stays within canvas for every yaw.
    radius=np.linalg.norm(points,axis=-1).max();extent=points*210/radius
    assert np.abs(extent).max()<=210
