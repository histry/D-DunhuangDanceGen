import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.bridge_dynamics_r48.geometry import path,optimize
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from dunhuang.research.event_feasibility_r48.study import windows,compile_clock
from dunhuang.research.phase_discrimination_r48.study import time_grid,content_scores

def signal():
    t=np.arange(256);rv=np.c_[.15*np.sin(t/25),.2*np.cos(t/40),.1*np.sin(t/30)]
    return np.broadcast_to(Rotation.from_rotvec(rv).as_matrix()[:,None],(256,22,3,3)).copy()

def test_fixed_internal_jet_origin_matches_frozen_segmented():
    R=signal();R[104:152]=np.nan;q,c=path(R,np.r_[np.ones(3),np.zeros(3)],certify=True);base,_=segmented(R,counts=(4,));assert np.allclose(q,base,atol=1e-12);assert c['physical_boundary_jet_residual']<1e-10

def test_hidden_gap_cannot_affect_optimized_bridge():
    from dunhuang.research.complete_r36.core import read
    cfg=read('src/dunhuang/research/event_feasibility_r48/config.json');cfg['maximum_solver_evaluations']=2;R=signal();J=np.zeros((22,3));J[:,1]=np.arange(22)/10;parents=np.r_[-1,np.arange(21)];ref=dict(D2=100.,D3=1000.);Q,c=optimize(R,J,parents,ref,cfg);poison=R.copy();poison[104:152]=np.nan;Z,d=optimize(poison,J,parents,ref,cfg);assert c==d;assert np.array_equal(Q,Z,equal_nan=True);assert np.array_equal(Q[:104],R[:104]);assert np.array_equal(Q[152:],R[152:])

def test_actual_long_event_window_coverage_and_clock():
    w=windows(590);mask=np.zeros(590,bool)
    for a,b in w:mask[a:b]=True;assert b-a<=256
    assert mask.all() and w[-1][1]==590
    u=dict(source_interval=[100,690],phases=[]);cfg=dict(window_frames=256,window_stride=128);r=compile_clock(u,[50,640],[],cfg);assert r['whole_event_preserved'];q={}
    for block in r['windows']:
        for t,x in zip(range(*block['target_interval']),block['source_positions']):assert t not in q or q[t]==x;q[t]=x
    assert q[50]==100 and q[639]==689
    assert compile_clock(u,[0,200],[],cfg) is None

def test_audio_grid_is_physical_not_assumed_thirty_hz():
    assert np.allclose(time_grid(100,22050,735),np.arange(100),atol=1e-12,rtol=0)
    assert abs(time_grid(100,22050,512)[-1]-99)>20

def test_frozen_token_shift_sign_and_zero_ties():
    rng=np.random.default_rng(2);d=rng.normal(size=(100,96));d/=np.linalg.norm(d,axis=1,keepdims=True);m=np.roll(d,8,axis=0);scores=content_scores(m,d,[30,40,50],[-15,-8,0,8,15]);assert np.argmax(scores)==3
    z=np.ones((100,96));scores=content_scores(z,z,[30,40,50],[-15,-8,0,8,15]);assert max(scores)-min(scores)<1e-12
