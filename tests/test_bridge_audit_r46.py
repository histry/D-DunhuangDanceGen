import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.bridge_audit_r46.audit import reconstruct,capacity
from dunhuang.research.bridge_audit_r46.diagnose import shifted,best_oracle

def test_observed_only_reconstruction_has_no_hidden_read_and_keeps_protected_body():
    t=np.arange(256)/80;R=Rotation.from_rotvec(np.c_[t*.2,np.sin(t)*.1,t*.05]).as_matrix();R=np.repeat(R[:,None],22,axis=1);raw=np.zeros((256,139));raw[:,7:]=sixd(R,row=True).reshape(256,132);a,_=reconstruct(raw,R,[104,152]);hidden=R.copy();hidden[104:152]=np.nan;b,_=reconstruct(raw,hidden,[104,152]);assert np.array_equal(a,b);assert np.array_equal(a[np.r_[:104,152:256]],raw[np.r_[:104,152:256]]);assert np.isfinite(a).all()

def test_shift_has_observation_mask_and_never_wraps():
    s=dict(onset=[1,2,3,4],beat_strength=[0,1,0,0],beats=[1],rms_accent_frames=[1],waveform_transient_frames=[2]);p=shifted(s,2);assert p['onset']==[0,0,1,2] and p['observed']==[False,False,True,True] and p['beats']==[3];assert shifted(s,-2)['onset']==[3,4,0,0]

def test_oracle_uses_eligible_joint_proxy_not_single_metric_or_unknown():
    bank=[dict(eligible=True,pattern_id=0),dict(eligible=False,pattern_id=1),dict(eligible=True,pattern_id=2)];err=[dict(rms_timing_seconds=.1,wrist_acceleration_timing_seconds=.2),dict(rms_timing_seconds=0,wrist_acceleration_timing_seconds=0),dict(rms_timing_seconds=.01,wrist_acceleration_timing_seconds=1)];assert best_oracle(bank,err)['pattern_id']==0;assert best_oracle(bank,[dict(rms_timing_seconds=None,wrist_acceleration_timing_seconds=.1)]*3) is None

def test_capacity_identity_reason_separates_quiet_and_score():
    p=dict(pattern_id=0,eligible=True,identity=True,ratios=dict(D1=1),source=[8,98],target=[8,98],score=0);m=dict(bank=[p],pattern_id=0,phases=[],interval=[8,99],evidence=dict(status='LOW_ACTIVITY_NO_FORCED_PHASE',maximum_speed=.01));assert capacity([m])[0]['reason']=='NO_DETECTED_STAGE';m['phases']=[dict(frame=50)];assert capacity([m])[0]['reason']=='NO_ELIGIBLE_NONIDENTITY'
