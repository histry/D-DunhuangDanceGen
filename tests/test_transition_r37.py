import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.transition_r37.clock import source_positions,FRAME_CENTRES
from dunhuang.research.transition_r37.memory import union_length,reuse,incremental
from dunhuang.research.transition_r37.geometry import quintic_bridge,c2_envelope,necessary_jump
from dunhuang.research.transition_r37.smoothing import linear,rotation_filter
from dunhuang.research.transition_r37.content import relations,observations

def event(a,b,source='s',t=0):return dict(source_uid=source,source_interval=[a,b],music_interval=[t,t+b-a])

def test_uid_agnostic_memory_preserves_forward_continuation():
    path=[event(0,100),event(100,200,t=100)]
    assert reuse(path)['reuse_ratio']==0
    evidence=incremental(path,event(0,100,t=200),{})
    assert evidence['overlap_fraction']==1 and evidence['backtrack']
    assert reuse(path+[event(0,100,t=200)])['reuse_ratio']==pytest.approx(1/3)
    assert union_length([[0,10],[8,20],[20,30]])==30

def test_versioned_clock():
    e=dict(time_contract=FRAME_CENTRES,source_interval=[10,110],music_interval=[0,100],time_warp=[[10,0],[109,99]])
    assert np.array_equal(source_positions(e,np.arange(100)),np.arange(10,110))
    with pytest.raises(ValueError):source_positions(dict(e,time_contract='UNKNOWN'),[0])
    with pytest.raises(ValueError):source_positions(e,[100])
    legacy=dict(e);legacy.pop('time_contract');legacy['time_warp']=[[10,0],[110,100]]
    assert source_positions(legacy,[99])[0]==109

def test_geometric_transition_preserves_skeleton_rotations_and_readonly():
    r=np.broadcast_to(np.eye(3),(100,22,3,3)).copy();r[50:]=Rotation.from_rotvec([0,.8,0]).as_matrix()
    out=quintic_bridge(r,[30,70]);assert np.array_equal(out[:30],r[:30]);assert np.array_equal(out[70:],r[70:])
    assert np.max(abs(np.linalg.det(out)-1))<1e-10
    speed=Rotation.from_matrix((out[:-1].swapaxes(-1,-2)@out[1:]).reshape(-1,3,3)).magnitude()
    assert speed.max()<.08
    assert max(necessary_jump(r,np.zeros(100),50)['necessary_remaining_angle_rad'])==pytest.approx(.8)
    assert np.all(c2_envelope(100,[30,70])[:31]==0)

def test_filter_and_smoother_physical_time_noisy_constant_velocity():
    rng=np.random.default_rng(7);t=np.arange(120)/30;truth=np.c_[t,.3*t,-t];noisy=truth+rng.normal(0,.025,truth.shape)
    f=linear(noisy,.025,1.5,smooth=False);s=linear(noisy,.025,1.5)
    assert np.mean((s-truth)**2)<np.mean((noisy-truth)**2)
    assert np.mean((s-truth)**2)<=np.mean((f-truth)**2)
    r=Rotation.from_rotvec(np.c_[t*.3,t*0,t*0]+rng.normal(0,.015,(120,3))).as_matrix()[:,None]
    out=rotation_filter(r,.025,2.)
    assert np.max(abs(np.linalg.det(out)-1))<1e-10
    assert np.linalg.norm(np.diff(out,n=3,axis=0))<np.linalg.norm(np.diff(r,n=3,axis=0))

def test_true_clock_positive_unknown_and_false_negative_rules():
    rows=[dict(audio_sha256='a',audio_frame_interval=[0,100],paired_clock_verified=True),dict(audio_sha256='a',audio_frame_interval=[0,100],paired_clock_verified=True),
          dict(audio_sha256='a',audio_frame_interval=[50,150],paired_clock_verified=True),dict(audio_sha256='a',audio_frame_interval=[200,300],paired_clock_verified=True),
          dict(audio_sha256='b',audio_frame_interval=[0,100],paired_clock_verified=False)]
    p,n=relations(rows);assert p[0,1] and not p[0,2] and not n[0,2] and n[0,3] and not n[0,4]

def test_static_content_information_not_erased_by_dynamics():
    R=np.broadcast_to(np.eye(3),(16,22,3,3)).copy();B=R.copy();B[:,16]=Rotation.from_rotvec([.5,0,0]).as_matrix()
    J=np.c_[np.arange(22)*.01,np.arange(22)*.02,np.zeros(22)];parents=np.r_[-1,np.arange(21)]
    a,m=observations(R,J,parents);b,_=observations(B,J,parents)
    assert a.shape==(16,274) and not m[:,[266,267,269]].any()
    assert np.array_equal(a[:,198:272],b[:,198:272]) and not np.array_equal(a[:,:198],b[:,:198])

def test_content_reconstruction_uses_shared_tokens_and_finite_gradients():
    torch=pytest.importorskip('torch')
    from dunhuang.research.transition_r37.content import build_model,multi_positive_loss
    torch.manual_seed(3);model=build_model();x=torch.randn(3,12,274);mask=torch.ones_like(x,dtype=torch.bool)
    scores,rec=model(torch.randn(3,12,50),x,mask,torch.ones(3,12,dtype=torch.bool));positive=np.eye(3,dtype=bool)
    loss=multi_positive_loss(scores,positive,~positive)+(rec-x[:,:,:198]).square().mean();loss.backward()
    assert torch.isfinite(loss) and model.motion[0].weight.grad.abs().sum()>0 and model.music[0].weight.grad.abs().sum()>0
    with pytest.raises(ValueError):multi_positive_loss(scores,np.zeros((3,3),bool),~positive)
