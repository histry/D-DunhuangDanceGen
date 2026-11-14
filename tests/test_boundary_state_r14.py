import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.boundary_state_r14.states import extract,design,label,reference,FEATURE_DIM,FRAMES
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound

def native():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],(150,24))
    t=np.arange(150)/30;x[:,4:7]=np.stack((t,t*0+1,.1*t*t),-1)
    x[:,7:13]=sixd(Rotation.from_euler("y",t*.3).as_matrix())
    return x

def test_hidden_truth_cannot_change_boundary_conditions():
    x=native();k=np.zeros_like(x,dtype=bool);k[FRAMES,4:]=True;y=x.copy();y[~k]=np.nan
    a=extract(x,k);b=extract(y,k)
    assert all(np.array_equal(u,v) for u,v in zip(a,b))
    pp=label(x);pa,ra=reference(x,pp,k);pb,rb=reference(y,pp,k)
    assert np.array_equal(pa,pb) and np.array_equal(ra,rb)
    for family in ("A0","A1","A2","A3","A4"):
        z=design(*a,family,donor=(a[2],a[3]) if family=="A4" else None)
        assert len(z)==FEATURE_DIM and np.isfinite(z).all()

def test_boundary_derivatives_use_observed_frames_and_local_gauge():
    x=native();_,_,s,m=extract(x)
    L=Rotation.from_euler("y",59/30*.3).as_matrix()
    assert m.all()
    assert np.allclose(s[3:6],(x[59,4:7]-x[58,4:7])@L*30,atol=2e-5)
    assert np.allclose(s[12:15],(x[92,4:7]-2*x[91,4:7]+x[90,4:7])@L*900,atol=1e-4)

def test_missing_root_is_unknown_and_not_zero_truth():
    x=native();k=np.zeros_like(x,dtype=bool);k[FRAMES,7:]=True
    _,_,s,m=extract(x,k);assert not m[:15].any() and not s[:15].any()
    with pytest.raises(ValueError):reference(x,np.zeros(270),k)

def test_yaw_translation_invariance_and_exact_label_reconstruction():
    x=native();y=x.copy();T=Rotation.from_euler("y",.7).as_matrix()
    y[:,4:7]=x[:,4:7]@T.T+[3,0,-2];y[:,7:13]=sixd(T@Rotation.from_euler("y",np.arange(150)/30*.3).as_matrix())
    a=extract(x);b=extract(y)
    assert np.allclose(a[0],b[0],atol=3e-5) and np.allclose(a[2],b[2],atol=3e-3)
    p,r=reference(x,label(x));assert np.allclose(p,x[60:90,4:7],atol=1e-6)
    assert np.allclose(r,Rotation.from_euler("y",np.arange(60,90)/30*.3).as_matrix(),atol=1e-6)

def test_immutable_jerk_witness_is_necessary_not_sufficient():
    fk=np.zeros((150,24,3),np.float32);fk[10,0,0]=1
    b=immutable_jerk_bound(fk,1620);assert b["necessary_status"]=="EXCLUDED_IMMUTABLE_JERK"
    changed=fk.copy();changed[60:90]=np.random.default_rng(3).normal(size=(30,24,3))
    w=b["witnesses"][0];i=w["template"][0];j=w["joint"]
    v=np.linalg.norm(np.diff(changed,n=3,axis=0)*30.**3,axis=-1)
    assert v[i,j]==w["value"] and not b["sufficient_feasibility_claimed"]
    fk[:]=0;fk[70,0,0]=1
    assert immutable_jerk_bound(fk,1620)["necessary_status"]=="UNKNOWN"


def test_group_derangement_and_fail_closed_qualification():
    from dunhuang.research.boundary_state_r14.readout import donors,qualification
    rows=[dict(source_group=str(i//3),raw_motion={"sha256":str(i)},start_frame=i) for i in range(24)]
    d=donors(rows,41);assert sorted(d.tolist())==list(range(24))
    assert all(rows[i]["source_group"]!=rows[j]["source_group"] for i,j in enumerate(d))
    scores=[dict(group=str(i),family=f,root_rmse_m=.5 if f=="A3" else 1.,root_orientation_rad=.1) for i in range(8) for f in ("A0","A1","A2","A3","A4")]
    cfg=dict(head_holdout_groups=8,representation_gate=dict(minimum_group_wins=6,root_improvement=.02,shuffle_improvement=.02,orientation_nonregression_rad=.01))
    assert qualification(scores,cfg)["A3"]["qualified"]
    assert not qualification([r for r in scores if r["group"]!="7"],cfg)["A3"]["qualified"]
