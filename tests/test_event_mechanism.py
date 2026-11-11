import copy
import numpy as np
import pytest
from dunhuang.generation.source_pairs import source_pairs, validate_pair
from dunhuang.research.event_metrics import identity_probe, representation_stats, event_margin
from dunhuang.contracts import sixd
from scipy.spatial.transform import Rotation


def row(uid, recording, role='train'):
    return dict(window_uid=uid, role=role, recording_uid='aist_capture:gXX_sBM_'+recording,
                choreography_uid='routine1', music_uid='song', start_frame=0, fps=30,frames=150,
                audio_offset_seconds=0, audio={'sha256':'music'},raw_motion={'sha256':recording},dataset='aistpp')


def test_independent_source_task_never_crosses_roles_or_fabricates_exact_pose():
    a,b,c=row('a','r1'),row('b','r2'),row('c','r3','val')
    pairs=source_pairs({'rows':[a,b,c]})
    assert len(pairs)==2
    assert all(not p['pose_exact_correspondence'] for p in pairs)
    assert pairs[0]['target_uid']!=pairs[0]['source_uid']
    by_id={r['window_uid']:r for r in [a,b,c]}
    bad=dict(pairs[0],source_uid='c')
    with pytest.raises(ValueError,match='role'):
        validate_pair(bad,by_id)
    with pytest.raises(ValueError,match='Train/Val'):
        source_pairs({'rows':[a,b,c]},roles=('test',))
    duplicate=copy.deepcopy(b);duplicate['raw_motion']=a['raw_motion']
    assert source_pairs({'rows':[a,duplicate]})==[]


def test_correspondence_requires_exact_music_and_time():
    a,b=row('a','r1'),row('b','r2')
    p=source_pairs({'rows':[a,b]})[0]
    b['start_frame']=75
    with pytest.raises(ValueError,match='start_frame'):
        validate_pair(p,{'a':a,'b':b})
    assert not source_pairs({'rows':[a,b]})


def test_closed_set_probe_detects_identity_and_constant_feature_chance():
    labels=np.repeat(np.arange(4),3);views=np.tile(np.arange(3),4)
    features=np.eye(4)[labels]
    assert identity_probe(features,labels,views)['accuracy']==1
    assert identity_probe(np.zeros_like(features),labels,views)['accuracy']==.25
    assert representation_stats(np.ones((12,5)))['effective_rank']==0


def test_pose_preference_is_not_merely_large_output_change():
    a=np.zeros((150,151),np.float32)
    a[:,7:]=np.tile(sixd(np.eye(3)).reshape(1,6),(150,24))
    b=a.copy()
    r=Rotation.from_euler('x',.7).as_matrix()
    b[:,79:127]=np.tile(sixd(r),(150,8))
    assert event_margin(a,b,a,b)['preference_margin_radians']>.69
    assert abs(event_margin(a,a,a,b)['preference_margin_radians'])<1e-7
    assert event_margin(b,a,a,b)['preference_margin_radians']<-.69


def test_native_geometry_ignores_scaler_and_unknown_storage():
    import torch
    from dunhuang.generation.event_geometry import event_geometry
    x=torch.zeros(2,150,151)
    x[:,:,7:]=torch.tensor([1,0,0,0,1,0]).repeat(24)
    available=torch.zeros_like(x,dtype=torch.bool)
    available[:,:,79:127]=True
    phase=torch.arange(150).float()[None,:,None].expand(2,-1,-1)/150
    geom=event_geometry(x,available,phase)
    assert geom.shape==(2,150,153)
    changed=x.clone();changed[~available]=float('nan')
    torch.testing.assert_close(geom,event_geometry(changed,available,phase))
    # Arbitrarily unsupported target root values cannot block a valid reference.
    changed[:,:,4:7]=1e9
    torch.testing.assert_close(geom,event_geometry(changed,available,phase))
    missing=available.clone();missing[:,:5]=False
    changed[:,:5]=float('nan')
    got=event_geometry(changed,missing,phase)
    assert not got[:,:5,:152].any()
    assert torch.isfinite(got).all()
    with pytest.raises(ValueError,match='Degenerate'):
        event_geometry(torch.zeros_like(x),available,phase)


def test_joint_and_additive_have_identical_capacity_and_active_fusion():
    import torch
    from dunhuang.generation.event_music_adapter import EventMusicAdapter,authorize_event_adapter
    torch.set_num_threads(2)
    torch.manual_seed(73);a=EventMusicAdapter('cross').eval()
    torch.manual_seed(73);b=EventMusicAdapter('additive').eval()
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())
    x=torch.randn(1,150,153);m=torch.randn(1,150,512)
    assert not a(x,m).any() and not b(x,m).any()
    with torch.no_grad():
        a.output.weight.normal_(0,.01);b.load_state_dict(a.state_dict())
    assert not torch.equal(a(x,m),b(x,m))
    a(x,m).square().mean().backward()
    assert a.music[1].weight.grad.abs().sum()>0
    assert a.attention.in_proj_weight.grad.abs().sum()>0
    p=dict(base_sha256='base',condition_schema='native_event_music_v1',mode='cross')
    for flag in (False,None,1,'true'):
        with pytest.raises(ValueError,match='research-only'):
            authorize_event_adapter(p,'base',flag)
    assert authorize_event_adapter(p,'base',True)['research_unqualified']


def test_native_plan_condition_does_not_admit_targets_or_use_scaler(tmp_path):
    from dunhuang.generation.soft_reference import build_native_event_reference
    from dunhuang.io import sha
    x=np.zeros((150,151),np.float32)
    x[:,7:]=np.tile(sixd(np.eye(3)).reshape(1,6),(150,24))
    x[:,4:7]=1e6 # Unsupported target root is immaterial to upper8 source geometry.
    p=tmp_path/'source.npy';np.save(p,x)
    plan=dict(frames=150,events=[dict(music_interval=[0,150],source_interval=[0,150],
        time_warp=[[0,0],[150,150]],reference=dict(path=str(p),sha256=sha(p)))])
    value,known,phase,audit=build_native_event_reference(plan,150)
    assert known[:,:,79:127].all()
    assert not known[:,:,:79].any() and not known[:,:,127:].any()
    assert not value[~known].any()
    assert not audit['full_151d_target_supported'] and not audit['reference_scaler_clipping']
    assert abs(phase[0,-1,0]-149/150)<1e-6


def test_task_receipt_cannot_claim_a_different_capture_or_pose_exact_truth():
    a,b=row('a','r1'),row('b','r2')
    p=source_pairs({'rows':[a,b]})[0];index={'a':a,'b':b}
    with pytest.raises(ValueError,match='binding'):
        validate_pair(dict(p,source_recording='invented'),index)
    with pytest.raises(ValueError,match='pose-exact'):
        validate_pair(dict(p,pose_exact_correspondence=True),index)


def test_timing_proxy_preserves_undefined_stationary_case_and_known_lag():
    from dunhuang.research.event_metrics import event_timing
    x=np.zeros((150,151),np.float32)
    x[:,7:]=np.tile(sixd(np.eye(3)).reshape(1,6),(150,24))
    assert event_timing(x,x)['best_lag_frames'] is None
    y=x.copy()
    t=np.arange(150)
    x[:,79:127]=np.tile(sixd(Rotation.from_euler('x',.3*np.sin(2*np.pi*t/50)).as_matrix()),(1,8))
    y[:,79:127]=np.tile(sixd(Rotation.from_euler('x',.3*np.sin(2*np.pi*(t-6)/50)).as_matrix()),(1,8))
    timing=event_timing(y,x)
    assert timing['best_lag_frames']==6 and timing['best_lag_correlation']>.999


def test_native_mask_only_is_a_hint_not_degenerate_observed_rotation():
    import torch
    from dunhuang.research.mechanism import Runtime
    from dunhuang.generation.conditions import ReferenceCondition
    from dunhuang.generation.event_music_adapter import EventMusicAdapter
    rt=Runtime.__new__(Runtime);rt.native_geometry=True
    rt.adapter=EventMusicAdapter('cross').eval()
    value=torch.zeros(1,150,151);known=torch.zeros_like(value,dtype=torch.bool)
    known[:,:,79:127]=True;phase=torch.arange(150).float()[None,:,None]/150
    rt.state=dict(enabled=True,condition=ReferenceCondition(value,known,phase,'mask_hint'),
                  mask_only=True,capture=False)
    music=torch.randn(1,150,512)
    torch.testing.assert_close(rt.inject(None,(),music),music)
