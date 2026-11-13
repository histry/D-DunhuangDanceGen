import numpy as np
import pytest
from dunhuang.research.reference_guidance.core import reference,canonical,ridge_fit,ridge_predict,ap,protection

def test_reference_unknown_poison_and_key_visibility():
    x=np.zeros((150,151),np.float32);x[:,79:127]=np.tile([1,0,0,0,1,0],(150,8))
    m=np.zeros_like(x,bool);m[10:30,79:127]=True
    k=np.zeros((150,8),bool);k[20]=True;k[90]=True
    a=reference(x,m,k);x[~m]=np.nan;b=reference(x,m,k)
    assert all(np.array_equal(v,w) for v,w in zip(a,b))
    assert not b[2][90].any() and b[2][18:23].all()
    m[20,79]=False
    assert not reference(x,m,k)[1][20,0]

def test_unique_physical_frames_no_overlap_or_music_replicate_inflation():
    def row(uid,start,rec):
        return dict(role="train",audio={"sha256":"audio"},raw_motion={"sha256":rec},audio_offset_seconds=0.,start_frame=start,window_uid=uid)
    rows=[row("a",0,"one"),row("b",15,"one"),row("c",0,"two")]
    a=canonical(rows,"music",10000);b=canonical(rows,"motion",10000)
    assert len(a)==135 and len(b)==255
    assert len({(key,f) for key,f,_,_ in a})==len(a)
    rows[0]["role"]="test"
    with pytest.raises(ValueError):canonical(rows,"music",20)

def test_train_only_balanced_ridge_and_constant_score_ties():
    x=np.arange(40,dtype=float)[:,None];y=(x[:,0]>20).astype(int);g=np.repeat(["a","b"],20)
    f=ridge_fit(x,y,g,.01);prediction=ridge_predict(f,x)
    assert prediction[-1]>prediction[0] and np.isfinite(f["beta"]).all()
    assert ap(np.array([1,0,1,0]),np.ones(4))==.5
    assert ap(np.ones(4),np.zeros(4)) is None
    with pytest.raises(ValueError):ridge_fit(x,np.zeros(40),g,.01)

def test_protection_rejects_root_contact_and_non_target_drift():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],(150,24))
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import sixd
    x[:,79:85]=sixd(Rotation.from_euler("z",np.arange(150)*.01).as_matrix())
    cfg=dict(non_target_mean_rad=.01,root_rms_m=.01,contact_max=.05,activity_ratio=[.9,1.1],acceleration_ratio=1.1,jerk_ratio=1.1)
    assert protection(x,x,cfg)["protected"]
    y=x.copy();y[:,4]=.1
    assert not protection(x,y,cfg)["protected"]
    y=x.copy();y[:,0]=.2
    assert not protection(x,y,cfg)["protected"]

def test_qualification_cannot_use_val_or_own_loss_alone():
    pytest.importorskip("torch")
    from dunhuang.research.reference_guidance.control import qualification
    cfg=dict(min_direction_rad=.001)
    good=dict(scope="train",case=0,timestep=999,source=0,arm="B1",numeric_trusted=True,protection={"protected":True},own_gain_rad=.1,directional_gain_rad=-.1)
    random=dict(good,arm="B3",own_gain_rad=.01)
    assert not qualification([good,random],cfg)["train_all_cases_all_times_pass"]["B1"]
    assert not qualification([dict(good,scope="val")],cfg)["train_all_cases_all_times_pass"]["B1"]

def test_condition_norm_accepts_tensors_without_numpy_device_conversion():
    torch=pytest.importorskip("torch")
    from dunhuang.research.reference_guidance.control import control_norm
    x=torch.ones(1,150,512,requires_grad=True)
    assert abs(control_norm(x,.01)-.01*np.sqrt(150*512))<1e-12
