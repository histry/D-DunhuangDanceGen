import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.phase_integrity_r60.common import CFG, crossings, proof_interval
from dunhuang.research.phase_integrity_r60.population import match_pair
from dunhuang.research.phase_integrity_r60.study import source_ci, error, fixed_window
from dunhuang.research.phase_integrity_r60.study import gap_task
from dunhuang.research.complete_r36.core import multi_warp

def test_equal_length_intervention_and_accepted_support():
    u=dict(source_interval=[40,140],support_component=[0,200])
    q=match_pair(u,np.ones(200),[[15,30],[145,160]])
    assert q['status']=='MATCHED' and q['length']==100
    assert q['A5'][1]-q['A5'][0]==100 and q['A5_crossings']
    assert 0<=q['A5'][0]<q['A5'][1]<=200 and q['relative_activity_difference']==0
    assert q['A4_crossings']==[]

def test_no_fabricated_match_across_quality_gap():
    q=match_pair(dict(source_interval=[40,140],support_component=[40,140]),np.ones(200),[[15,30],[145,160]])
    assert q['status']=='NOT_MATCHABLE'

def test_already_cut_positive_is_not_intact():
    q=match_pair(dict(source_interval=[40,140],support_component=[0,200]),np.ones(200),[[35,50]])
    assert q['reason']=='A4_ALREADY_CUTS_PROTECTED_SUPPORT'

def test_support_boundaries_are_half_open_not_false_crossings():
    assert crossings([30,145],[[15,30],[145,160]])==[]
    with pytest.raises(ValueError):proof_interval([29,80],[30,90])

def test_activity_constraint_prevents_unmatched_dynamic_confound():
    u=dict(source_interval=[40,140],support_component=[0,200]);v=np.ones(200);v[140:]=1000;v[:40]=1000
    assert match_pair(u,v,[[15,30],[145,160]])['status']=='NOT_MATCHABLE'

def test_missing_seam_support_is_explicit():
    cfg=copy.deepcopy(CFG);cfg['hidden_gap']=300
    q=match_pair(dict(source_interval=[40,140],support_component=[0,200]),np.ones(200),[[15,30],[145,160]],cfg)
    assert q['status']=='NOT_MATCHABLE'

def test_only_intervened_seam_with_equal_real_context_is_evaluated():
    q=match_pair(dict(source_interval=[40,140],support_component=[0,200]),np.ones(200),[[145,160]])
    assert q['status']=='MATCHED' and q['common_hidden_seams']==['right']
    assert all(q[k][0]>=30 and 200-q[k][1]>=30 for k in ('A4','A5'))
    assert match_pair(dict(source_interval=[0,100],support_component=[0,200]),np.ones(200),[[0,10],[101,120]])['reason']=='A4_LACKS_FULL_COMMON_REAL_CONTEXT'

def test_source_balancing_does_not_count_correlated_windows_as_sources():
    rows=[dict(source_uid='a',delta=1.)]*100+[dict(source_uid='b',delta=3.)]
    r=source_ci(rows,'delta');assert r['sources']==2 and r['mean']==2.
    assert source_ci(rows,'missing')['mean'] is None
    assert source_ci(rows,'delta')==r

def test_native_and_virtual_split_are_exact():
    R=Rotation.from_rotvec(np.random.default_rng(3).normal(size=(20*22,3))*.1).as_matrix().reshape(20,22,3,3)
    assert np.array_equal(np.concatenate((R[:9],R[9:])),R)
    assert error(R,R)<1e-12

def test_no_forced_96_to_240_warp():
    assert multi_warp([0,96],[0,240],[],[]) is None
    assert multi_warp([0,240],[0,240],[],[]) is not None

def test_equal_length_arms_have_same_endpoint_legal_domain():
    for target in CFG['target_lengths']:
        a=multi_warp([40,280],[0,target],[dict(type='upper_speed_peak',frame=100)],[60,120,180,240])
        b=multi_warp([50,290],[0,target],[dict(type='upper_speed_peak',frame=100)],[60,120,180,240])
        assert (a is None)==(b is None)

def test_fixed_length_risk_is_not_renamed_complete():
    u=dict(source_interval=[40,140],support_component=[0,200]);source=dict(protected_spans=[[0,200]])
    r=fixed_window(u,96,source)
    assert r['status']=='PROTECTION_NOT_SATISFIABLE'
    assert fixed_window(u,240,source)['status']=='NOT_CONSTRUCTIBLE_ON_ACCEPTED_COMPONENT'

def test_bridge_never_reads_hidden_motion_or_chooses_with_truth():
    truth=np.broadcast_to(np.eye(3),(56,22,3,3)).copy();changed=truth.copy()
    changed[4:52]=Rotation.from_rotvec(np.tile([.1,.05,0.],(48*22,1))).as_matrix().reshape(48,22,3,3)
    J=np.column_stack([np.arange(22)*.03,np.zeros((22,2))]);parents=np.zeros(22,int)
    reference=dict(D2=1e9,D3=1e9,shoulder_wrist_lower_m=0.)
    r1,z1=gap_task(('same',truth,J,parents,reference));r2,z2=gap_task(('same',changed,J,parents,reference))
    assert np.array_equal(z1['same_G1'],z2['same_G1'])
    assert r1['methods']['G1']['error_rad']<r2['methods']['G1']['error_rad']
    assert r1['solver_hidden_read'] is False and r1['virtual_split_array_exact']
