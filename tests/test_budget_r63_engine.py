"""R63 contracts on original synthetic sources, without dataset score tuning."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.budget_r63 import engine as e


@pytest.fixture
def fixture():
    n = 320
    t = np.arange(n)
    sources = {}
    for k, uid in enumerate(('left_source', 'right_source')):
        v = np.zeros((n, 22, 3))
        for j in range(22):
            v[:, j, 0] = .018*np.sin(t/43+j/8+k)
            v[:, j, 1] = .025*np.cos(t/61+j/10+k)
            v[:, j, 2] = .002*t/10+.016*np.sin(t/27+k)
        sources[uid] = Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    lib = SimpleNamespace(rotations=sources)
    J = np.zeros((22, 3))
    J[:, 1] = np.arange(22)*.06
    J[:, 0] = np.sin(np.arange(22))*.1
    parents = np.arange(-1, 21)
    case = dict(key='synthetic', music_interval=[1000, 1528],
                beats=list(range(1000, 1528, 15)), baseline_left_frames=240,
                baseline_right_frames=240)
    for side, start in (('left', 20), ('right', 30)):
        case[side] = dict(event_uid=side+'_event', source_uid=side+'_source',
                          source_interval=[start, start+240], support_component=[0, n],
                          phases=[dict(type='speed_peak', body_region='upper', frame=start+90),
                                  dict(type='direction_turn', body_region='upper', frame=start+150)])
    reference = dict(D2=1e12, D3=1e12, shoulder_wrist_lower_m=0.)
    return case, lib, J, parents, reference


def test_exact_budget_and_signed_debt():
    expected = {24: 252, 36: 246, 48: 240, 60: 234, 72: 228}
    for g, n in expected.items():
        a = e.allocation(g)
        assert a['left_frames'] == n == a['right_frames']
        assert 2*n+g == a['total_frames'] == 528
        assert a['left_core_debt_frames'] == (48-g)//2
    # An odd debt still sums exactly, rather than rounding both sides alike.
    a = e.allocation(25)
    assert a['left_frames']+a['right_frames']+25 == 528


@pytest.mark.parametrize('kwargs', [dict(g=0), dict(g=-1), dict(g=True), dict(g=24.),
    dict(g=48, total=529), dict(g=500), dict(g=48, left=0), dict(g=48, baseline_gap=0)])
def test_invalid_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        e.allocation(**kwargs)


def test_compile_uses_absolute_clock_full_interval_and_phase_namespace(fixture):
    case, lib, J, parents, _ = fixture
    m, R = e.compile_core(case['left'], [1000, 1252], case['beats'], lib, J, parents)
    assert m['status'] == 'LEGAL_CORE'
    w = np.asarray(m['time_warp'])
    assert np.array_equal(w[[0, -1]], [[20, 1000], [259, 1251]])
    assert m['phase_inputs'][0]['type'] == 'upper_speed_peak'
    assert m['warp']['anchor_source_frames'] == [110, 170]
    q = np.asarray(m['source_positions'])
    assert q[0] == 20 and q[-1] == 259 and (np.diff(q) > 0).all()
    assert min(m['interpolation_read_support']) >= 0
    assert max(m['interpolation_read_support']) < 320
    assert .75 <= m['rate_min'] <= m['rate_max'] <= 1.33
    assert R.shape == (252, 22, 3, 3)


def test_source_component_rejection_has_no_fallback(fixture, monkeypatch):
    case, lib, J, parents, ref = fixture
    case['left']['support_component'] = [30, 320]
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected source support must not reach the solver')
    monkeypatch.setattr(e, 'solve_boundary', forbidden)
    rec, arrays = e.evaluate_pair(case, 24, lib, J, parents, ref)
    assert rec['left']['status'] == 'SOURCE_SUPPORT_REJECTED'
    assert rec['status'] == 'CORE_RECOMPILE_REJECTED'
    assert not rec['methods']['G1']['constructible']
    assert 'synthetic_g24_G1' not in arrays


def test_illegal_e1w_is_explicit_not_clipped(fixture):
    case, lib, J, parents, _ = fixture
    side = copy.deepcopy(case['left'])
    side['source_interval'] = [20, 70]
    m, R = e.compile_core(side, [1000, 1252], case['beats'], lib, J, parents)
    assert m['status'] == 'E1W_ILLEGAL' and R is None
    assert 'source_positions' not in m


def test_invalid_warp_frame_centers_or_rates_are_not_accepted(fixture, monkeypatch):
    case, lib, J, parents, _ = fixture
    for invalid in ([[20, 1000], [260, 1251]], [[20, 1000], [21, 1100], [259, 1251]]):
        monkeypatch.setattr(e, 'multi_warp', lambda *args, w=invalid: dict(time_warp=w))
        m, R = e.compile_core(case['left'], [1000, 1252], case['beats'], lib, J, parents)
        assert m['status'] == 'E1W_INVALID' and R is None


def test_every_allocation_recompiles_full_sources_and_actual_jets(fixture, monkeypatch):
    case, lib, J, parents, ref = fixture
    calls = []
    original = e.multi_warp
    def spy(interval, target, phases, beats):
        calls.append((list(interval), list(target)))
        return original(interval, target, phases, beats)
    monkeypatch.setattr(e, 'multi_warp', spy)
    baseline, _ = e.evaluate_pair(case, 48, lib, J, parents, ref)
    changed, _ = e.evaluate_pair(case, 24, lib, J, parents, ref)
    assert len(calls) == 4
    assert calls == [([20, 260], [1000, 1240]), ([30, 270], [1288, 1528]),
                     ([20, 260], [1000, 1252]), ([30, 270], [1276, 1528])]
    assert baseline['left']['source_interval'] == changed['left']['source_interval']
    assert baseline['left']['source_positions'] != changed['left']['source_positions']
    e.attach_retention(changed, baseline)
    assert changed['boundary_readset_changed_vs_fixed48']
    assert changed['boundary_jets_changed_vs_fixed48']
    assert changed['left']['retention_vs_fixed48']['mean_joint_speed_radps_ratio'] != 1.


def test_solver_receives_no_hidden_truth_and_core_bytes_unchanged(fixture, monkeypatch):
    case, lib, J, parents, ref = fixture
    actual = e.solve_boundary
    seen = []
    def spy(observed, gap, controls=False):
        a, b = gap
        assert np.isnan(observed[a:b]).all()
        assert len(observed) == 60+(b-a)
        seen.append(observed.copy())
        return actual(observed, gap, controls=controls)
    monkeypatch.setattr(e, 'solve_boundary', spy)
    rec, arrays = e.evaluate_pair(case, 48, lib, J, parents, ref, controls=True)
    assert len(seen) == 1
    assert set(rec['methods']) == {'G0', 'G1'}
    for method, m in rec['methods'].items():
        assert m['constructible'], method
        Q = arrays[m['arrays_key']]
        assert Q.shape == (528, 22, 3, 3)
        assert np.array_equal(Q[:240], arrays['synthetic_g48_left'])
        assert np.array_equal(Q[288:], arrays['synthetic_g48_right'])
        assert m['hidden_SO3_error_rad'] is None and m['hidden_FK_error_m'] is None
        assert m['core_exact'] and not m['fallback_used']
    assert not rec['boundary_proof']['hidden_read']
    assert rec['boundary_available_pair_frame_indices'] == list(range(236, 240))+list(range(288, 292))


def test_hidden_poison_cannot_change_boundary_solution(fixture):
    case, lib, J, parents, _ = fixture
    _, L = e.compile_core(case['left'], [1000, 1240], case['beats'], lib, J, parents)
    _, R = e.compile_core(case['right'], [1288, 1528], case['beats'], lib, J, parents)
    observed, gap = e.make_observed(L, R, 48)
    first, _ = e.solve_boundary(observed, gap)
    observed[gap[0]:gap[1]] = 1e9
    poisoned, proof = e.solve_boundary(observed, gap)
    assert np.array_equal(first['G1'], poisoned['G1'])
    assert not proof['hidden_read']


def test_chart_failure_preserved_without_retiming_fallback(fixture, monkeypatch):
    case, lib, J, parents, ref = fixture
    def reject(observed, gap, controls=False):
        a, b = gap
        return {'G1': None}, dict(available_boundary_frame_indices=list(range(a-4, a))+list(range(b, b+4)),
            guards={'G1': {'chart_eligible': False}}, hidden_read=False)
    monkeypatch.setattr(e, 'solve_boundary', reject)
    rec, arrays = e.evaluate_pair(case, 72, lib, J, parents, ref)
    assert rec['left']['status'] == 'LEGAL_CORE'
    assert not rec['methods']['G1']['constructible']
    assert not rec['methods']['G1']['fallback_used']
    assert 'synthetic_g72_G1' not in arrays


def test_retention_requires_same_pair_fixed48(fixture):
    case, lib, J, parents, ref = fixture
    a, _ = e.evaluate_pair(case, 48, lib, J, parents, ref)
    b = copy.deepcopy(a)
    b['case_key'] = 'other'
    with pytest.raises(ValueError):
        e.attach_retention(a, b)
    b['case_key'] = a['case_key']
    b['gap_frames'] = 24
    with pytest.raises(ValueError):
        e.attach_retention(a, b)


@pytest.mark.parametrize('missing', [('left',), ('right',), ('left', 'right')])
def test_unmatchable_missing_unit_retained_without_compilation(fixture, monkeypatch, missing):
    case, lib, J, parents, ref = fixture
    case.update(status='NOT_MATCHABLE', missing_source_uids=[])
    for side in missing:
        case[side+'_source_uid'] = case[side]['source_uid']
        case['missing_source_uids'].append(case[side]['source_uid'])
        case[side] = None
    def forbidden(*args, **kwargs):
        raise AssertionError('Incomplete pair must not compile a substitute or half-pair')
    monkeypatch.setattr(e, 'compile_core', forbidden)
    baseline, arrays = e.evaluate_pair(case, 48, lib, J, parents, ref, controls=True)
    changed, other = e.evaluate_pair(case, 24, lib, J, parents, ref)
    assert arrays == other == {}
    assert baseline['status'] == 'CORE_RECOMPILE_REJECTED'
    assert baseline['case_status'] == 'NOT_MATCHABLE'
    assert set(baseline['methods']) == {'G0', 'G1'}
    assert all(not m['constructible'] and not m['fallback_used'] for m in baseline['methods'].values())
    for side in missing:
        assert baseline[side]['status'] == 'SOURCE_MISSING'
        assert baseline[side]['source_interval'] is None
        assert baseline[side]['source_uid'] == case[side+'_source_uid']
    e.attach_retention(changed, baseline)
    assert changed['left']['retention_vs_fixed48'] is None
    assert changed['right']['retention_vs_fixed48'] is None
