import numpy as np
import pytest
import torch
from dunhuang.research.lodge_reference_r24.compiler import compile_events,angular_errors,frequency_readout
from dunhuang.research.lodge_reference_r24.noise import common_noise

def event():
    return dict(source_interval=[0,90],music_interval=[0,90],time_warp=[[0,0],[90,90]],observed_joints=list(range(22)),
        root_translation_observed=False,contact_observed=False,full_phases=[dict(type=k,frame=t,score=.7,semantic_status="UNKNOWN")
            for k,t in zip(["upper_motion_onset","upper_speed_peak","upper_direction_turn"],[15,45,70])])
def native():
    x=np.zeros((90,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24);x[:,4:7]=50
    return x
def config():
    return dict(core_start=64,phase_types=["upper_motion_onset","upper_speed_peak","upper_direction_turn"],cue_radius=1)

def test_complete_core_mapping_masks_only_real_rotations():
    e=event();x=native();z,mask,phases=compile_events([e,e],[x,x],90,config())
    assert z.shape==(2,256,139) and np.array_equal(z[0],z[1])
    assert not mask[:,:7].any() and mask[:,7:].sum()>0 and not mask[:64].any() and not mask[154:].any()
    assert np.max(abs(z[:,:,:7]))==0 and len(phases[0])==3
    assert mask[:,7:].all(-1).sum()<45
    assert np.max(angular_errors(z[0],z[1]))<1e-6

def test_unverified_root_or_incomplete_body_cannot_be_prompts():
    e=event();e["root_translation_observed"]=True
    with pytest.raises(ValueError):compile_events([e,e],[native(),native()],90,config())
    e=event();e["observed_joints"]=list(range(8))
    with pytest.raises(ValueError):compile_events([e,e],[native(),native()],90,config())

def test_common_noise_matches_all_conditions_and_replays_exactly():
    original=torch.randn;receipts=[]
    for _ in range(2):
        with common_noise(17,batch=6,frames=8,dim=4) as r:
            a=torch.randn((6,8,4));b=torch.randn_like(a)
            assert torch.equal(a[:1].expand_as(a),a) and torch.equal(b[:1].expand_as(b),b)
            assert not torch.equal(a,b)
        assert torch.randn is original
        receipts.append(dict(r))
    assert receipts[0]==receipts[1] and receipts[0]["draws"]==2
    with pytest.raises(RuntimeError):
        with common_noise(1):raise RuntimeError("stop")
    assert torch.randn is original

def test_high_frequency_metric_separates_jitter_without_modifying_motion():
    t=np.arange(256)/30;points=np.zeros((256,22,3));points[:,1,0]=np.sin(2*np.pi*t)
    old=points.copy();smooth=frequency_readout(points)
    assert np.array_equal(points,old)
    points[:,1,0]+=.8*np.sin(2*np.pi*10*t);jitter=frequency_readout(points)
    assert jitter["high_frequency_velocity_energy_fraction"]>smooth["high_frequency_velocity_energy_fraction"]+.3
