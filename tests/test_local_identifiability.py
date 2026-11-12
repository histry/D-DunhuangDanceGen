import copy
import numpy as np
import torch
import pytest
from dunhuang.alignment.conditional_candidates import match_statistics,build_candidates,validate_roles
from dunhuang.alignment.content_discrimination import intervention,pair_result,template_paths,fixed_score,fit_normalization,query_auc
from dunhuang.alignment.local_information_audit import GRID,stages,reconstruct,ridge_fit,predict,binary_auc
from dunhuang.alignment.loss_gradient_audit import gradients,cosine_matrix,NAMES
from dunhuang.alignment.partial_path import StrictPartialPath
from dunhuang.alignment.token_geometry_audit import deduplicate,spectrum
torch.set_num_threads(2)

def config():
    return dict(count_tolerance=0,time_w1_seconds=.15,histogram_l1=.25,energy_log_ratio=np.log(1.25),
        onset_density_per_second=.25,min_queries=30,min_recordings=15,min_train_components=3,min_val_components=2)

def rows():
    return [dict(role="train",source_group="a",recording_uid="r"+str(i),music_uid="song",choreography_uid="dance",start_frame=i%2*150,
        audio={"sha256":"audio"},raw_motion={"sha256":"raw"+str(i)}) for i in range(4)]

def stats():
    return dict(counts=np.ones((4,2),int)*12,quantiles=np.ones((4,2,16)),hist=np.ones((4,2,4))*.25,
        energy=np.ones((4,2)),onset_density=np.ones(4))

def bank():
    rng=np.random.default_rng(10);x=rng.normal(size=(4,3,144)).astype(np.float32)
    return dict(music=x,motion=x.copy(),mv=np.ones((4,3),bool),dv=np.ones((4,3),bool),
        mt=np.tile(np.arange(3)/15,(4,1)).astype(np.float32),dt=np.tile(np.arange(3)/15,(4,1)).astype(np.float32))

def test_strict_candidates_do_not_relax_when_small_or_empty_and_keep_multi_positives():
    r=rows();b=bank();s=stats();out,e=build_candidates(r,b,s,config())
    assert e["populations"]["train"]["queries"]==4 and not e["populations"]["train"]["population_sufficient"]
    assert out["positive"][0,:2].all() and set(out["candidates"][0,:2])=={0,2}
    s["energy"][1::2]*=100
    out,e=build_candidates(r,b,s,config())
    assert not out["eligible"].any() and not e["relaxed"] and e["failed_rule_counts"]["energy"]>0
    s=stats();s["counts"][1,0]+=1
    ok,detail=match_statistics(s,0,1,config());assert not ok and not detail["rules"]["count"]
    r[1]["role"]="val"
    with pytest.raises(ValueError,match="crosses"):validate_roles(r)

def test_fixed_path_content_and_clock_decomposition_and_gradient():
    b=bank();head=StrictPartialPath(width=144);head.null_logit.requires_grad_(False)
    q=np.array([0,1]);c=np.array([0,2])
    r=pair_result(head,b,q,c);null=float(r["null_cost"])
    path=template_paths(b,q,c,.5,null)
    full=fixed_score(r,path,.5);z=pair_result(head,b,q,c,"clock");zero=fixed_score(z,path,.5)
    assert torch.equal(r["allowed"],z["allowed"])
    assert torch.equal(full["time"],zero["time"]) and torch.equal(full["constant"],zero["constant"])
    torch.testing.assert_close(full["score"]-zero["score"],full["content"]-zero["content"])
    loss=-full["mean_content"].sum();loss.backward()
    assert head.music.weight.grad.isfinite().all() and head.motion.weight.grad.isfinite().all()
    assert head.null_logit.grad is None

def test_intervention_preserves_valid_support_and_is_deterministic():
    x=torch.randn(1,5,144);v=torch.tensor([[True,True,False,True,False]]);x[~v]=float("nan")
    a=intervention(x,v,"token_shuffle",19);b=intervention(x,v,"token_shuffle",19)
    assert torch.equal(a,b) and a[~v].eq(0).all()
    torch.testing.assert_close(a[v][:,128:],x[v][:,128:])
    assert intervention(x,v,"descriptor",0)[...,:128].eq(0).all()
    assert intervention(x,v,"token",0)[...,128:].eq(0).all()

def test_common_real_grid_exposes_aliasing_and_refuses_padding():
    n=np.arange(120);tcn=np.tile(((-1.)**n)[None,:,None],(2,1,128)).astype(np.float32)
    raw=tcn[:,:,:74];transformer=tcn[:,::2]
    s=stages(raw,tcn,transformer)
    assert all(v.shape==(2,115,128) for v in s.values())
    assert np.mean((s["A1_tcn"]-s["A2_even"])**2)>1.
    assert np.abs(s["A3_antialias"]).max()==0
    with pytest.raises(ValueError,match="extrapolation"):reconstruct(tcn[:,::2],np.arange(0,120,2),np.array([119]))

def test_probe_and_normalization_fit_train_only():
    rng=np.random.default_rng(5);x=rng.normal(size=(5,10,128));y=x[...,:3]*2
    fit=ridge_fit(x,y,np.array([0,1,2]),.01)
    changed=x.copy();changed[3:]=1e6;changed_y=y.copy();changed_y[3:]=-1e6
    other=ridge_fit(changed,changed_y,np.array([0,1,2]),.01)
    assert all(np.array_equal(fit[k],other[k]) for k in fit)
    b=bank();r=rows();r[2]["role"]="val";r[3]["role"]="val"
    a=fit_normalization(b,r,"whiten");other=copy.deepcopy(b);other["music"][2:]=1e6
    z=fit_normalization(other,r,"whiten")
    np.testing.assert_array_equal(a["music"]["matrix"],z["music"]["matrix"])
    assert np.mean((predict(fit,x[:3])-y[:3])**2)<.01

def test_gradient_conflict_zero_gradient_and_no_parameter_mutation():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));losses=dict(zip(NAMES,[(p*p).sum(),-(p*p).sum(),p.sum()*0,p.sum(),p.sum()]))
    g=gradients(losses,[p]);flat=[x[0] for x in g];m=cosine_matrix(flat)
    assert m[0][0]==pytest.approx(1) and m[0][1]==pytest.approx(-1) and m[0][2] is None
    assert p.grad is None and torch.equal(p.detach(),torch.tensor([1.,2.]))

def test_hub_population_deduplication_and_auc_ties():
    b=bank();r=rows();events=[dict(music=[dict(frame=j,event_type="onset",body_part="music") for j in range(3)]) for _ in r]
    x,owners=deduplicate(b,r,events,"music",128,"train")
    assert len(x)==3 and len(owners)==3
    assert spectrum(np.ones((8,4)))["effective_rank"]==0
    assert binary_auc([1,0,1,0],[0,0,0,0])==.5
    p=np.array([[True,True,False,False]]);v=np.ones_like(p)
    assert query_auc(np.zeros((1,4)),p,v)[0]==.5

def test_equal_global_pool_does_not_identify_local_rotation():
    # A constructive representation ambiguity, not an assertion about this dataset.
    x=np.eye(4);q=np.roll(np.eye(4),1,axis=1)
    assert np.array_equal(x.mean(0), (x@q).mean(0))
    assert np.trace(x@(x@q).T)==0
    assert np.trace(x@((x@q)@q.T).T)==4

def test_repeat_consistency_materializes_lazy_banks_once():
    from dunhuang.alignment.token_geometry_audit import repeat_consistency
    r=rows()
    # Make all captures use the same original interval for repeated-capture analysis.
    for x in r:x["start_frame"]=0
    class Lazy:
        calls={}
        def __getitem__(self,key):
            self.calls[key]=self.calls.get(key,0)+1
            return np.ones((4,60,128),np.float32) if key=="mt" else np.ones((4,5,60,128),np.float32)
    source=Lazy();a=repeat_consistency(r,source,20)
    assert source.calls=={"mt":1,"dt":1} and a["train"]["capture_pairs"]==3
