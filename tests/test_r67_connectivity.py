"""Synthetic G1 wiring/cache tests; mocked gates cannot qualify real motion."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.sdmr_warp_rank import connectivity as c


@pytest.fixture
def fixture(monkeypatch):
    baseline=np.broadcast_to(np.eye(3),(240,22,3,3)).copy()
    case=dict(key='coreA',parent_case_key='pairA',parent_record_key='recordA',parent_side='left',
        source_uid='sourceA',event_uid='eventA',source_interval=[0,240],target_interval=[100,340],
        time_warp=[[0,100],[239,339]],baseline_core_R_sha256='TEST_FIXTURE')
    ref=dict(D2=10.,D3=20.,shoulder_wrist_lower_m=.1)
    ctx=dict(registered={'coreA':copy.deepcopy(case)},pairs={'pairA':dict(left=baseline.copy(),right=baseline.copy(),
        case=dict(music_interval=[100,628]))},J=np.zeros((22,3)),parents=np.arange(-1,21),
        reference=ref,bindings=dict(TEST_FIXTURE=True),cache={},counters=dict(check_requests=0,
        actual_geometry_calls=0,cached_geometry_reuses=0,actual_constructed=0,actual_screen_passed=0))
    calls=[]
    def solve(observed,gap,controls):
        assert controls is False
        calls.append(observed.copy())
        Q=observed.copy();Q[gap[0]:gap[1]]=np.eye(3)
        return {'G1':Q},dict(TEST_FIXTURE=True,guards=dict(G1={'eligible':True}))
    def measure(Q,observed,J,parents,gap,reference):
        assert len(Q)==528 and gap==(240,288) and reference==ref
        assert np.isnan(observed[240:288]).all()
        return dict(constructible=True,passes_original_screen=True,metrics=dict(TEST_FIXTURE=True))
    monkeypatch.setattr(c,'solve_boundary',solve)
    monkeypatch.setattr(c,'measure',measure)
    return case,baseline,ctx,calls


def changed_core(baseline):
    result=baseline.copy()
    result[-3,0]=Rotation.from_rotvec([.01,0,0]).as_matrix()
    return result


def test_actual_retimed_edge_poses_drive_new_g1_and_full528_measure(fixture):
    case,baseline,ctx,calls=fixture
    candidate=changed_core(baseline)
    result,arrays=c.check(case,candidate,ctx)
    observed=calls[0]
    assert len(calls)==1 and np.array_equal(observed[:30],candidate[-30:])
    assert np.array_equal(observed[78:],baseline[:30]) and np.isnan(observed[30:78]).all()
    assert result['passes_original_screen'] and not result['previous_bridge_qualification_inherited']
    assert np.array_equal(arrays['full_R'][:240],candidate)
    assert np.array_equal(arrays['full_R'][288:],baseline)
    assert result['Native24']=='NOT_EVALUABLE' and result['artistic_quality']=='NOT_EVALUATED'


def test_identical_copy_is_cached_with_separate_actual_and_reuse_counts(fixture):
    case,baseline,ctx,calls=fixture
    first,arrays=c.check(case,baseline,ctx)
    arrays['full_R'][0,0]=0  # Returned arrays must never poison the saved cache.
    second,again=c.check(case,baseline.copy(),ctx)
    assert len(calls)==1 and second['cache_hit']
    assert second['actual_geometry_calls_this_request']==0 and second['cached_geometry_reuses_this_request']==1
    assert second['cumulative_counts']['actual_geometry_calls']==1
    assert second['cumulative_counts']['cached_geometry_reuses']==1
    assert np.array_equal(again['full_R'][0],baseline[0])


def test_changed_internal_warp_does_not_inherit_original_jets_or_bridge(fixture):
    case,baseline,ctx,calls=fixture
    a,_=c.check(case,baseline,ctx)
    b,_=c.check(case,changed_core(baseline),ctx)
    assert len(calls)==2 and not b['cache_hit']
    assert a['boundary_jets_sha256']!=b['boundary_jets_sha256']
    assert a['full_R_sha256']!=b['full_R_sha256']


def test_changed_endpoints_or_parent_identity_rejected_without_geometry(fixture):
    case,baseline,ctx,calls=fixture
    bad=baseline.copy();bad[-1,0]=Rotation.from_rotvec([.01,0,0]).as_matrix()
    with pytest.raises(ValueError,match='same source endpoints'):
        c.check(case,bad,ctx)
    changed=dict(case,parent_side='right')
    with pytest.raises(ValueError,match='frozen parent Core identity'):
        c.check(changed,baseline,ctx)
    assert not calls


def test_construction_rejection_has_no_saved_bridge_fallback(fixture,monkeypatch):
    case,baseline,ctx,_=fixture
    monkeypatch.setattr(c,'solve_boundary',lambda *a,**k:({'G1':None},{'guards':{'G1':{'reason':'TEST_FIXTURE_REJECT'}}}))
    result,arrays=c.check(case,baseline,ctx)
    assert not result['constructible'] and not result['passes_original_screen'] and not result['fallback_used']
    assert result['full_R_sha256'] is None and 'full_R' not in arrays
    assert result['actual_geometry_calls_this_request']==1


def test_prepare_recompiles_both_original_cores_exactly_without_solving(monkeypatch,tmp_path):
    baseline=np.broadcast_to(np.eye(3),(240,22,3,3)).copy()
    ref=dict(D2=10.,D3=20.,shoulder_wrist_lower_m=.1)
    metas={}
    for name,interval,uid in [('left',[0,240],'A'),('right',[288,528],'B')]:
        metas[name]=dict(status='LEGAL_CORE',event_uid='event'+uid,source_uid='source'+uid,
            source_interval=[0,240],target_interval=interval,
            time_warp=[[0,interval[0]],[239,interval[1]-1]],
            full_core_R_sha256=c.array_hash(baseline),source_R_sha256='TEST_FIXTURE_RAW_'+uid)
    parent=dict(key='pairA',music_interval=[0,528],beats=[],left=dict(side='left'),right=dict(side='right'))
    record=dict(key='recordA',gap_frames=48,timing_offset_frames=0,music_interval=[0,528],**metas)
    saved=dict(population={'original_numerical_gate':ref},cases=[parent],records=[record],
        selections=[dict(case_key='pairA',policy='Fixed48',record_key='recordA')])
    cases=[dict(key='case'+str(i),parent_case_key='pairA',parent_side='left',parent_record_key='recordA',
        source_uid='sourceA',source_interval=[0,240],target_interval=[0,240],
        time_warp=metas['left']['time_warp'],baseline_core_R_sha256=c.array_hash(baseline)) for i in range(12)]
    monkeypatch.setattr(c,'sha',lambda path:c.PARENT_SHA if c.Path(path).as_posix().endswith('main/result.json') else 'TEST_FIXTURE_CONFIG')
    monkeypatch.setattr(c,'read',lambda path:saved)
    monkeypatch.setattr(c,'asset_build',lambda:([],object(),np.zeros((22,3)),np.arange(-1,21),ref,
        dict(reference_sha256='TEST_FIXTURE_REFERENCE',catalog_sha256='TEST_FIXTURE_CATALOG')))
    calls=[]
    def compile_core(side,target,*args):
        calls.append((side['side'],target))
        return copy.deepcopy(metas[side['side']]),baseline.copy()
    monkeypatch.setattr(c,'compile_core',compile_core)
    monkeypatch.setattr(c,'solve_boundary',lambda *args,**kwargs:pytest.fail('prepare must not solve geometry'))
    ctx=c.prepare(tmp_path,population=dict(parent_sha=c.PARENT_SHA,cases=cases))
    assert calls==[('left',[0,240]),('right',[288,528])]
    assert len(ctx['pairs'])==1 and len(ctx['change_cases'])==12
    assert ctx['preparation_bridge_solves']==ctx['counters']['actual_geometry_calls']==0
