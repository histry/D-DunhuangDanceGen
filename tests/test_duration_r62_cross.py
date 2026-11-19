import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.duration_r62 import cross_source as cross


def motion(start=0.,end=.3,n=70):
    t=np.linspace(start,end,n)
    axis=np.tile([.02,.03,1.],(22,1))
    return Rotation.from_rotvec((t[:,None,None]*axis[None]).reshape(-1,3)).as_matrix().reshape(n,22,3,3)


@pytest.mark.parametrize('gap',[24,36,48,60,72])
def test_actual_gap_length_and_unmodified_context(gap):
    left,right=motion(),motion(.5,.8)
    observed,interval=cross.make_observed(left,right,gap)
    assert observed.shape==(60+gap,22,3,3)
    assert interval==(30,30+gap)
    assert np.array_equal(observed[:30],left[-30:])
    assert np.array_equal(observed[-30:],right[:30])
    assert np.isnan(observed[slice(*interval)]).all()
    curves,proof=cross.solve_boundary(observed,interval,controls=True)
    for Q in curves.values():
        assert Q is not None
        assert np.array_equal(Q[:30],left[-30:])
        assert np.array_equal(Q[-30:],right[:30])
    assert proof['hidden_read'] is False


def test_each_duration_keeps_identical_boundary_poses_and_jets():
    left,right=motion(),motion(.5,.8)
    jets=[];readsets=[]
    for duration in (24,36,48,60,72):
        R,(a,b)=cross.make_observed(left,right,duration)
        jets.append(cross.edge_jets(R,(a,b)))
        readsets.append(np.concatenate((R[a-4:a],R[b:b+4])))
    assert all(np.array_equal(jets[0],x) for x in jets)
    assert all(np.array_equal(readsets[0],x) for x in readsets)


def test_poisoned_unobserved_gap_does_not_enter_either_solver():
    R,gap=cross.make_observed(motion(),motion(.5,.8),36)
    poisoned=R.copy();poisoned[slice(*gap)]=1e9
    a,pa=cross.solve_boundary(R,gap,controls=True)
    b,pb=cross.solve_boundary(poisoned,gap,controls=True)
    assert pa==pb
    for name in ('G0','G1'):
        assert np.array_equal(a[name],b[name])


def test_g1_rejection_is_preserved_and_never_replaced_by_g0(monkeypatch):
    R,gap=cross.make_observed(motion(),motion(.5,.8),60)
    monkeypatch.setattr(cross,'segmented',lambda *args:(None,dict(status='REJECTED')))
    curves,proof=cross.solve_boundary(R,gap,controls=True)
    assert curves['G1'] is None and curves['G0'] is not None
    assert proof['rejection_fallback'] is False


def test_saved_core_slice_uses_explicit_nonzero_music_origin():
    R=motion(n=300)
    actual=cross.core_slice(R,[1050,1120],[1000,1300])
    assert np.array_equal(actual,R[50:120])
    with pytest.raises(ValueError):cross.core_slice(R,[50,120],[1000,1300])
    with pytest.raises(ValueError):cross.core_slice(R[:299],[1050,1120],[1000,1300])


def test_true_context_and_integer_duration_contract():
    left,right=motion(),motion(.5,.8)
    with pytest.raises(ValueError):cross.make_observed(left[:29],right,24)
    with pytest.raises(ValueError):cross.make_observed(left,right,24.5)
    with pytest.raises(ValueError):cross.make_observed(left,right,True)
    R,gap=cross.make_observed(left,right,24);R[gap[0]-2]=np.nan
    with pytest.raises(ValueError):cross.solve_boundary(R,gap)


def test_boundary_evidence_keeps_own_source_time_and_unknown_transition():
    source=dict(phase_evidence=[dict(evidence_span=[100,104])],protected_spans=[[98,108]])
    result=cross.boundary_evidence([101.2,102.3,103.4,104.5],source)
    assert result['interpolation_read_support']==[101,102,103,104,105]
    assert result['raw_observed_read_frames']==[101,102,103]
    assert result['raw_minimum_distance_source_frames']==0.
    assert result['verified_phase']=='UNKNOWN'
    assert result['transition_source_evidence']=='UNOBSERVED_DIFFERENT_SOURCE_GAP'


def test_descriptive_minimum_pass_is_not_adaptive_qualification():
    rows=[dict(pair_case=0,gap_frames=24,methods=dict(G1=dict(constructible=True,passes_original_screen=False))),
          dict(pair_case=0,gap_frames=48,methods=dict(G1=dict(constructible=True,passes_original_screen=True))),
          dict(pair_case=1,gap_frames=24,methods=dict(G1=dict(constructible=False,passes_original_screen=False)))]
    summary=cross.duration_summary(rows)
    assert summary['pair_responses'][0]['minimum_passing_duration']==48
    assert summary['pair_responses'][1]['minimum_passing_duration'] is None
    assert summary['adaptive_policy_qualified'] is False
    assert summary['by_duration']['24']['methods']['G1']['constructed']==1


def test_transported_acceleration_boundary_stencil_is_symmetric():
    R,gap=cross.make_observed(motion(),motion(.5,.8),24)
    curves,_=cross.solve_boundary(R,gap)
    J=np.zeros((22,3));J[:,1]=np.arange(22)*.01
    parents=np.r_[-1,np.arange(21)]
    reference=dict(D2=1e9,D3=1e9,shoulder_wrist_lower_m=0.)
    result=cross.measure(curves['G1'],R,J,parents,gap,reference)
    a,b=gap
    assert result['transported_acceleration_increment_indices']==[a-3,b-1]
    assert result['hidden_SO3_error_rad'] is None
    assert result['hidden_FK_error_m'] is None
    assert np.isfinite(result['transported_boundary_acceleration_increment_radps2'])
