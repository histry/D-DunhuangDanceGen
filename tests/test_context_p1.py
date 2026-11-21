import numpy as np
from dunhuang.research.context_feasibility_p1.graph import sequential
from dunhuang.research.context_feasibility_p1.common import admissible,boundary_key,CFG

def test_registration_complete_and_unique_controls():
    cases=CFG['cases'];assert len(cases)==28
    assert sum(c['role']=='failure' for c in cases)==14
    assert len({(c['song'],c['slot'],c['event_uid']) for c in cases})==28

def test_unknown_never_supplies_a_path():
    layers=[[dict(uid='a',heuristic=0)],[dict(uid='b',heuristic=0)]]
    assert sequential(layers,{})[0] is None
    assert sequential(layers,{('a','b'):{}})[0]==['a','b']

def test_suffix_lookahead_beats_dead_greedy_prefix():
    layers=[[dict(uid='a',heuristic=2),dict(uid='x',heuristic=1)],[dict(uid='b',heuristic=0)],[dict(uid='c',heuristic=0)]]
    assert sequential(layers,{('x','b'):{},('b','c'):{}})[0]==['x','b','c']

def test_policy_is_not_geometry():
    a=dict(source_uid='same',target_interval=[0,233]);b=dict(source_uid='same',target_interval=[281,514]);assert not admissible(a,b)
    assert admissible(a,dict(b,source_uid='other'))

def test_boundary_cache_ignores_unread_interior_but_not_read_pose():
    L=np.broadcast_to(np.eye(3),(233,22,3,3)).copy();R=L.copy();J=np.zeros((22,3));p=np.r_[-1,np.zeros(21,dtype=int)];ref={'D3':1}
    k=boundary_key(L,R,J,p,ref);L[10,0,0,0]=2;assert boundary_key(L,R,J,p,ref)==k
    L[-1,0,0,0]=2;assert boundary_key(L,R,J,p,ref)!=k

def test_four_context_measure_matches_full_original_connection():
    from scipy.spatial.transform import Rotation
    from dunhuang.paper1.execute import connection
    from dunhuang.research.context_feasibility_p1.geometry import solve
    n=233;t=np.arange(n);v=np.zeros((n,22,3));v[:,:,0]=.1*np.sin(t[:,None]/30);L=Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(n,22,3,3);R=L[::-1].copy()
    J=np.c_[np.arange(22)*.01,np.arange(22)*.02,np.zeros(22)];p=np.r_[-1,np.arange(21)];ref=dict(D2=1e9,D3=1e9,shoulder_wrist_lower_m=0.)
    original=connection(L,R,J,p,ref);_,actual,Q=solve(('fixture',L,R,J,p,ref))
    assert original['bridge_sha256']==actual['bridge_sha256'] and original['passes_original_screen']==actual['passes_original_screen']
    for k in original['metrics']:assert original['metrics'][k]==actual['metrics'][k],k
