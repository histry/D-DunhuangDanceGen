import itertools,copy
import numpy as np
import torch
import pytest
from dunhuang.alignment.partial_path import exact_path,soft_path_cost,StrictPartialPath
from dunhuang.alignment.sampling import group_schedule,coverage
from dunhuang.alignment.event_targets import review_packet,validate_anchors,crop_events
from dunhuang.alignment.event_metrics import source_clock_masks,multi_positive_loss,independent_metrics,proxy_retrieval

def enumerated(cost,a,b,i=0,j=0):
    n,k=cost.shape
    if i==n and j==k: return [0.]
    values=[]
    if i<n:values += [a[i]+x for x in enumerated(cost,a,b,i+1,j)]
    if j<k:values += [b[j]+x for x in enumerated(cost,a,b,i,j+1)]
    if i<n and j<k: values += [cost[i,j]+x for x in enumerated(cost,a,b,i+1,j+1)]
    return values

def test_path_matches_exhaustive_and_soft_partition_gradient():
    torch.set_num_threads(2)
    rng=np.random.default_rng(2)
    for n,k in ((1,1),(2,3),(3,2)):
        c=rng.uniform(0,2,(n,k));a=np.ones(n)*.5;b=np.ones(k)*.5
        paths=np.array(enumerated(c,a,b))
        assert exact_path(c,a,b)["cost"]==pytest.approx(paths.min())
        x=torch.tensor(c[None],dtype=torch.double,requires_grad=True)
        v=torch.ones(1,n,dtype=torch.bool);w=torch.ones(1,k,dtype=torch.bool)
        eps=.17
        result=soft_path_cost(x,torch.tensor(a[None]),torch.tensor(b[None]),v[:,:,None]&w[:,None,:],v,w,eps)
        expected=-eps*np.log(np.exp(-paths/eps).sum())
        assert float(result)==pytest.approx(expected,abs=1e-9)
        result.sum().backward();assert x.grad.isfinite().all() and x.grad.min()>=0

def test_hard_crossing_null_masks_and_padding_equivalence():
    c=np.array([[1.,0.],[0.,1.]])
    r=exact_path(c,.3,.3)
    assert len(r["matches"])==1
    r=exact_path(np.ones((2,3))*2,.1,.1);assert not r["matches"] and r["cost"]==pytest.approx(.5)
    torch.manual_seed(8);head=StrictPartialPath(width=4)
    m=torch.randn(1,3,4);d=torch.randn(1,4,4);v=torch.tensor([[True,False,True]]);w=torch.tensor([[True,False,True,False]])
    t=torch.arange(3)/15;u=torch.arange(4)/15
    x=head(m,d,v,w,t,u);z=head(m[:,v[0]],d[:,w[0]],v[:,v[0]],w[:,w[0]],t[v[0]],u[w[0]])
    assert torch.allclose(x["score"],z["score"],atol=1e-6)
    bad=m.clone();bad[~v]=float("nan");assert torch.equal(x["score"],head(bad,d,v,w,t,u)["score"])
    assert head.decode(x)["crossings"]==0
    x["score"].sum().backward();assert head.music.weight.grad.isfinite().all()

def test_full_rotations_cover_more_groups_than_batch_and_train_only():
    rows=[dict(role="train",source_group=str(i)) for i in range(31)]+[dict(role="val",source_group="heldout")]
    s=group_schedule(rows,20,12,4);r=coverage(rows,s)
    assert r["omitted_groups"]==0 and max(r["per_group_counts"])-min(r["per_group_counts"])<=1
    assert np.array_equal(s,group_schedule(rows,20,12,4))
    with pytest.raises(ValueError):coverage(rows,np.array([[31]]))

def row():
    return dict(window_uid="w",role="train",source_group="g",recording_uid="r",start_frame=0,
        audio_offset_seconds=0.,audio={"sha256":"a"*64},raw_motion={"sha256":"b"*64},
        music_uid="song",choreography_uid="dance")

def test_multi_positive_and_ambiguous_captures_not_negatives():
    r=row();c=[r,dict(r,recording_uid="other"),dict(r,start_frame=120),dict(r,music_uid="other"),dict(r,music_uid="z",source_group="z",choreography_uid="z")]
    p,v=source_clock_masks(r,c)
    assert p.tolist()==[True,True,False,False,False] and v.tolist()==[True,True,True,False,True]
    s=torch.tensor([[1.,2.,-.5,999.,-1.]],requires_grad=True)
    loss=multi_positive_loss(s,torch.tensor(p[None]),torch.tensor(v[None]),temperature=1.);loss.backward()
    assert s.grad[0,3]==0 and s.grad[0,0]<0 and s.grad[0,1]<0
    metrics=proxy_retrieval(np.zeros((1,5)),p[None],v[None]);assert metrics["R1"]==0 and metrics["negative_pair_win_rate"]==.5

def test_unreviewed_labels_rejected_and_no_fake_event_metrics():
    r=row();packet=review_packet(r)
    with pytest.raises(ValueError,match="validated"):validate_anchors(packet,r)
    assert independent_metrics([],{},{})["EventMAE_seconds"] is None
    packet.update(validated=True,reviewers=["a","b"],label_origin="independent_source_review")
    packet["anchors"]=[dict(music_seconds=1.,motion_seconds=1.1,is_null=False,event_type="turn",body_part="root",ambiguous=False,agreement=True,reviewers=["a","b"])]
    validate_anchors(packet,r)
    q=copy.deepcopy(packet);q["model_predictions_shown"]=True
    with pytest.raises(ValueError):validate_anchors(q,r)
    q=copy.deepcopy(packet);q["audio_offset_seconds"]=.5
    with pytest.raises(ValueError):validate_anchors(q,r)
    score=independent_metrics([packet],{"w":r},{"w":[dict(is_null=False,motion_seconds=1.1,event_type="turn",body_part="root",confidence=.9)]})
    assert score["EventMAE_seconds"]==0 and score["TypedEventAccuracy"]==1

def test_synthetic_transform_has_exact_identity_and_no_length_shortcut():
    from dunhuang.alignment.event_perturb import construct,supervised_path_loss
    rng=np.random.default_rng(7);x=rng.normal(size=(60,8)).astype(np.float32);donor=rng.normal(size=x.shape).astype(np.float32)
    s=construct(x,donor,5)
    assert s["music"].shape==s["motion"].shape==(12,8)
    assert (s["target"]<0).sum()==len(s["inserted_indices"])==2
    for i,j in enumerate(s["target"]):
        if j>=0:np.testing.assert_array_equal(s["music"][i],s["motion"][j])
    slopes=np.diff(s["warp"][:,1])/np.diff(s["warp"][:,0]);assert slopes.min()>=.75 and slopes.max()<=1.33
    head=StrictPartialPath(width=8)
    sample={k:s[k][None] for k in ("music","motion","mt","dt","target")}
    losses,_=supervised_path_loss(head,sample,"cpu")
    sum(losses.values()).backward();assert head.null_logit.grad.isfinite()
    assert all(v.isfinite() for v in losses.values())

def test_weak_evidence_no_model_labels_and_unknown_not_forced_null():
    from dunhuang.alignment.event_weak_labels import weak_labels
    r=row();r["start_frame"]=0
    rows=[dict(r,recording_uid=str(i),window_uid=str(i)) for i in range(3)]
    m=dict(relative_seconds=1.,confidence=.8,detector_agreement=True,event_type="onset",body_part="music")
    d=dict(relative_seconds=1.,confidence=.8,detector_agreement=True,event_type="speed_peak",body_part="upper")
    events=[dict(music=[m],motion=[d]) for _ in rows]
    target,weight,e=weak_labels(rows,events)
    assert target[:,0,0].sum()==3 and weight.min()>0 and e["states"]["PSEUDO_MATCH"]==3
    events[0]["music"]=[dict(m,relative_seconds=3.)]
    target,weight,e=weak_labels(rows,events)
    assert weight[0,0]==0 and e["states"]["PSEUDO_NULL"]==0 and e["no_model_predictions_used"]
