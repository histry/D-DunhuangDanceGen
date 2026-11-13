import numpy as np
import pytest
from dunhuang.research.protected_guidance.solver import project,constraints,ROOT,CONTACT,OTHER
from dunhuang.research.event_information.operators import event_labels,phase_context,moments,finalize,normalize,distance,neighbours,static_witness

def test_constrained_projection_has_an_actual_correction_beyond_dose_reduction():
    a=np.zeros((ROOT+CONTACT+3*OTHER,2));a[0]=[2,1]
    cfg=dict(linear_safety_fraction=.9,max_solver_iterations=200,solver_ftol=1e-10,linear_tolerance=1e-6)
    z,result=project(a,np.eye(2),np.array([1.,0.]),cfg)
    assert result["status"]=="SOLVED_FINITE_SUBSPACE" and np.allclose(z,[.56,-.22],atol=1e-6)
    assert z[0]>.45 and constraints(a,np.eye(2),z,.9)[0].min()>-1e-6
    assert not result["global_infeasibility_claimed"]

def test_projection_enforces_contact_infinity_and_mean_joint_tangent():
    a=np.zeros((ROOT+CONTACT+3*OTHER,1));a[ROOT,0]=2/np.sqrt(CONTACT)
    a[ROOT+CONTACT:,0]=np.tile([2/np.sqrt(OTHER),0,0],OTHER)
    cfg=dict(linear_safety_fraction=.9,max_solver_iterations=200,solver_ftol=1e-10,linear_tolerance=1e-6)
    z,r=project(a,np.eye(1),np.ones(1),cfg)
    assert r["status"]=="SOLVED_FINITE_SUBSPACE" and z[0]<=.45+1e-6
    with pytest.raises(ValueError):project(a,np.zeros((1,1)),np.ones(1),cfg)
    cfg["linear_safety_fraction"]=1.1
    with pytest.raises(ValueError):project(a,np.eye(1),np.ones(1),cfg)

def test_matrix_free_basis_matches_known_linear_operator():
    torch=pytest.importorskip("torch")
    from dunhuang.research.protected_guidance.solver import basis
    a=torch.tensor([[1.,2.,0.],[0.,1.,3.]],dtype=torch.float64);zero=torch.zeros(3,dtype=torch.float64);seed=torch.tensor([.4,.2,.1],dtype=torch.float64)
    vs,j,gram,target=basis(lambda z:a@z,zero,seed,torch.ones_like(zero,dtype=torch.bool),.5,3)
    v=torch.stack(vs,-1).numpy()
    assert np.allclose(j,a.numpy()@v*.5) and np.allclose(gram,v.T@v)
    assert np.allclose(target,v.T@seed.numpy()/.5)

def test_hold_interval_is_a_distinct_half_open_target_not_a_centre_expansion_claim():
    events=[dict(type="upper_pose_hold",frame=19,evidence_span=[10,30])]
    t=np.arange(40);centre=event_labels(events,t,"upper_pose_hold");span=event_labels(events,t,"upper_pose_hold",True)
    assert centre.sum()==5 and span.sum()==20 and not span[30] and not (centre&~span).any()
    assert not event_labels(events,t,"upper_speed_peak",True).any()

def test_phase_features_do_not_include_the_target_hold_or_speed_events():
    events=[dict(type="upper_motion_onset",frame=10),dict(type="upper_direction_turn",frame=20)]
    a=phase_context(events,np.arange(30));b=phase_context(events+[dict(type="upper_pose_hold",frame=3),dict(type="upper_speed_peak",frame=8)],np.arange(30))
    assert all(np.array_equal(x,y) for x,y in zip(a,b)) and not a[1][0].all()

def test_observation_mask_poison_does_not_change_statistics_or_distance():
    x=np.array([[1.,2.,3.],[4.,5.,6.]]);m=np.array([[1,0,1],[1,1,0]],bool);bad=x.copy();bad[~m]=np.nan
    a=finalize([moments(x,m)]);b=finalize([moments(bad,m)])
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert np.array_equal(distance(x,m,x,m),distance(bad,m,bad,m))
    z,known=normalize(bad,m,a);assert np.isfinite(z).all() and not z[~known].any()

def test_neighbour_bank_excludes_entire_query_music_group():
    x=np.arange(8)[:,None];m=np.ones_like(x,bool);g=np.array(["a"]*4+["b"]*4)
    n=neighbours([(x,m)],np.array([0,7]),np.arange(8),g,k=2)
    assert (n[0]>=4).all() and (n[1]<4).all()

def test_static_pose_information_witness_does_not_change_hold_state():
    w=static_witness()
    assert w["dynamic_features_identical"] and w["upper8_rotation_difference_rad"]>.5 and w["both_are_holds"]
    assert w["static_identity_is_not_hold_state"] and not w["semantic_claim"]

def test_shadow_search_consumes_actual_planner_without_changing_default(tmp_path):
    from dunhuang.io import sha
    from dunhuang.research.planner_integration.study import candidates,search,inspect_conditions
    from dunhuang.planning.planner import plan,_search
    from dunhuang.planning.event_plan import semantic_plan_id
    source=np.zeros((100,151),np.float32);source[:,7:]=np.tile([1,0,0,0,1,0],(100,24))
    file=tmp_path/"source.npy";np.save(file,source);ref=dict(path=str(file),sha256=sha(file))
    event=dict(event_uid="one",source_uid="source",reference=ref,source_interval=[0,100],center=50,activity_percentile=.5,feasibility={"temporal8":"PASS","native":"UNKNOWN"},entry=np.tile(np.eye(3),(8,1,1)).tolist(),exit=np.tile(np.eye(3),(8,1,1)).tolist())
    new=dict(event_uid="one",source_uid="source",reference=ref,source_interval=[0,100],safe_anchor={"frame":50},normalization_support=True,observed_joints=list(range(22)),kinematic_phases=[{"frame":50}])
    timing=dict(fps=30,frames=100,beats=[25,50,75],slots=[dict(start=0,end=100,density=.5)])
    options,_=candidates([event],[new],timing,"E0_legacy");cost,sequence=search(options,5,3)
    original=plan({"entries":[event]},timing,"a"*64)
    consumed=_search({"entries":[event]},timing,"a"*64,fixed_events={0:sequence[0]})[0]
    assert semantic_plan_id(consumed)==semantic_plan_id(original) and cost==original["planning_cost"]
    arrays,audit=inspect_conditions(consumed,[new],np.ones(151),np.zeros(151))
    assert audit["source_frames_verified"]==100 and not arrays["known"][:,:,:79].any()
    assert arrays["known"][0,:100,79:127].all() and not arrays["known"][0,100:].any()
