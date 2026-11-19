"""Asymmetric fixed-budget contracts; no training or source search."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.timing_r64 import engine as e


@pytest.fixture
def data():
    t = np.arange(320)
    rotations = {}
    for k, uid in enumerate(('left_source', 'right_source')):
        v = np.zeros((320, 22, 3))
        for j in range(22):
            v[:, j, 0] = .018*np.sin(t/43+j/8+k)
            v[:, j, 1] = .025*np.cos(t/61+j/10+k)
            v[:, j, 2] = .002*t/10+.016*np.sin(t/27+k)
        rotations[uid] = Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(320, 22, 3, 3)
    case = dict(key='test', music_interval=[1000, 1528], beats=list(range(1000, 1528, 15)),
                baseline_left_frames=240, baseline_right_frames=240,
                baseline_gap_frames=48, audio_sha256='fixed-audio-hash')
    for side, a in (('left', 20), ('right', 30)):
        case[side] = dict(event_uid=side+'_event', source_uid=side+'_source',
            source_interval=[a, a+240], support_component=[0, 320],
            phases=[dict(type='speed_peak', body_region='upper', frame=a+90),
                    dict(type='direction_turn', body_region='upper', frame=a+150)])
    J = np.zeros((22, 3)); J[:, 1] = np.arange(22)*.06
    J[:, 0] = np.sin(np.arange(22))*.1
    return case, SimpleNamespace(rotations=rotations), J, np.arange(-1, 21), dict(
        D2=1e12, D3=1e12, shoulder_wrist_lower_m=0.)


def test_all25_allocations_repay_actual_baseline():
    for g in e.DURATIONS:
        for offset in e.OFFSETS:
            a = e.allocation(g, offset)
            assert a['left_frames'] == 240+(48-g)//2+offset
            assert a['right_frames'] == 528-g-a['left_frames']
            assert a['left_core_debt_frames'] == a['left_frames']-240
            assert a['right_core_debt_frames'] == a['right_frames']-240
            assert a['left_core_debt_frames']+a['right_core_debt_frames'] == 48-g
            assert a['baseline_left_frames'] == a['baseline_right_frames'] == 240


@pytest.mark.parametrize('g,offset,total', [(0, 0, 528), (25, 0, 528), (48, 6, 528),
    (True, 0, 528), (48., 0, 528), (48, False, 528), (48, -24., 528), (48, 0, 529)])
def test_unregistered_or_invalid_grid_rejected(g, offset, total):
    with pytest.raises(ValueError):
        e.allocation(g, offset, total)


def test_zero_offset_fixed48_exact_frozen_r63_parity(data):
    case, lib, J, parents, reference = data
    rec, arrays = e.evaluate(case, 48, 0, lib, J, parents, reference)
    old, old_arrays = e.frozen.evaluate_pair(case, 48, lib, J, parents, reference)
    for kind in ('left', 'right', 'G1'):
        assert np.array_equal(arrays[rec['key']+'_'+kind], old_arrays[old['key']+'_'+kind])
    assert rec['methods']['G1']['metrics'] == old['methods']['G1']['metrics']
    assert rec['boundary_jets_sha256'] == old['boundary_jets_sha256']
    assert rec['boundary_readset_sha256'] == old['boundary_readset_sha256']
    assert rec['case_key'] == case['key']
    assert rec['key'] == 'test_o0_g48'
    assert set(rec['methods']) == {'G1'}


def test_asymmetry_rebuilds_warp_jets_and_complete_cores(data, monkeypatch):
    case, lib, J, parents, reference = data
    before = deepcopy(case)
    calls = []
    actual = e.frozen.compile_core
    def spy(side, target, beats, library, skeleton, parent):
        calls.append((list(side['source_interval']), list(target)))
        return actual(side, target, beats, library, skeleton, parent)
    monkeypatch.setattr(e.frozen, 'compile_core', spy)
    records = []
    for offset in (-24, 0, 24):
        rec, arrays = e.evaluate(case, 48, offset, lib, J, parents, reference)
        records.append(rec)
        nl, nr = rec['allocation']['left_frames'], rec['allocation']['right_frames']
        Q, L, R = [arrays[rec['key']+'_'+suffix] for suffix in ('G1', 'left', 'right')]
        assert len(Q) == nl+48+nr == 528
        assert np.array_equal(Q[:nl], L) and np.array_equal(Q[nl+48:], R)
        assert rec['left']['source_positions'][0] == 20
        assert rec['left']['source_positions'][-1] == 259
        assert rec['right']['source_positions'][0] == 30
        assert rec['right']['source_positions'][-1] == 269
    assert len(calls) == 6 and all(v[0] in ([20, 260], [30, 270]) for v in calls)
    assert calls[0][1] == [1000, 1216] and calls[-1][1] == [1312, 1528]
    assert len({r['boundary_jets_sha256'] for r in records}) == 3
    assert len({r['execution_cache_key'] for r in records}) == 3
    assert case == before


def test_cache_contains_actual_full_execution_not_just_pair_uid(data):
    case, lib, J, parents, reference = data
    rec, _ = e.evaluate(case, 36, 12, lib, J, parents, reference)
    payload = rec['execution_cache_payload']
    assert payload['sides']['left']['time_warp'] == rec['left']['time_warp']
    assert payload['sides']['right']['target_interval'] == rec['right']['target_interval']
    assert payload['boundary_jets_sha256'] == rec['boundary_jets_sha256']
    assert payload['boundary_readset_sha256'] == rec['boundary_readset_sha256']
    assert payload['audio_sha256'] == case['audio_sha256']
    assert not rec['execution_cache_lookup'] and not rec['prior_bridge_or_core_reused']


def test_illegal_retiming_is_failure_not_event_replacement(data):
    case, lib, J, parents, reference = data
    case['left']['source_interval'] = [20, 70]
    rec, arrays = e.evaluate(case, 24, 24, lib, J, parents, reference)
    assert rec['left']['status'] == 'E1W_ILLEGAL'
    assert rec['left']['source_interval'] == [20, 70]
    assert not rec['methods']['G1']['constructible']
    assert rec['key']+'_G1' not in arrays
    assert rec['key']+'_right' in arrays


def test_missing_unit_has_stable_full_candidate_identity(data):
    case, lib, J, parents, reference = data
    case['left'] = None
    case['left_source_uid'] = 'left_source'
    case['status'] = 'NOT_MATCHABLE'
    rec, arrays = e.evaluate(case, 72, -24, lib, J, parents, reference)
    assert not arrays and rec['left']['status'] == 'SOURCE_MISSING'
    assert rec['left']['source_uid'] == 'left_source'
    assert rec['execution_cache_key'] and not rec['independent_e1w_recompile']
    assert rec['case_key'] == 'test' and rec['allocation']['total_frames'] == 528


def test_case_cannot_redefine_declared_baseline(data):
    case, lib, J, parents, reference = data
    case['baseline_left_frames'] = 252
    with pytest.raises(ValueError, match='baseline'):
        e.evaluate(case, 48, 0, lib, J, parents, reference)


def _without_reuse_flags(record):
    return {k: v for k, v in record.items() if k not in (
        'prior_bridge_or_core_reused', 'execution_cache_lookup', 'independent_e1w_recompile')}


@pytest.mark.parametrize('controls', [False, True])
def test_sealed_zero_offset_import_equals_fresh_except_three_execution_flags(data, monkeypatch, controls):
    case, lib, J, parents, reference = data
    old, old_arrays = e.frozen.evaluate_pair(case, 36, lib, J, parents, reference, controls=controls)
    # Frozen study annotations must not become extra R64 engine observations.
    old['role'] = 'DEV'
    old['retention_reference_key'] = case['key']+'_g48'
    old['left']['retention_vs_fixed48'] = {'ratio': 1.}
    fresh, fresh_arrays = e.evaluate(case, 36, 0, lib, J, parents, reference)
    before = deepcopy(old)
    def forbidden(*args, **kwargs):
        raise AssertionError('Imported identical R63 execution must not rerun geometry')
    monkeypatch.setattr(e.frozen, 'evaluate_pair', forbidden)
    monkeypatch.setattr(e.frozen, 'compile_core', forbidden)
    monkeypatch.setattr(e.frozen, 'solve_boundary', forbidden)
    reused, arrays = e.from_frozen(case, 36, old, old_arrays)
    assert _without_reuse_flags(reused) == _without_reuse_flags(fresh)
    assert arrays.keys() == fresh_arrays.keys()
    assert all(np.array_equal(a, fresh_arrays[k]) for k, a in arrays.items())
    assert reused['prior_bridge_or_core_reused'] and reused['execution_cache_lookup']
    assert not reused['independent_e1w_recompile']
    assert old == before


@pytest.mark.parametrize('failure', ['illegal_warp', 'missing_unit', 'chart'])
def test_rejected_frozen_candidates_preserve_exact_failure_contract(data, monkeypatch, failure):
    case, lib, J, parents, reference = data
    if failure == 'illegal_warp':
        case['left']['source_interval'] = [20, 70]
    elif failure == 'missing_unit':
        case.update(left=None, left_source_uid='left_source', status='NOT_MATCHABLE')
    else:
        original_solver = e.frozen.solve_boundary
        def reject(observed, gap, controls=False):
            curves, proof = original_solver(observed, gap, controls=controls)
            curves['G1'] = None
            return curves, proof
        monkeypatch.setattr(e.frozen, 'solve_boundary', reject)
    old, old_arrays = e.frozen.evaluate_pair(case, 48, lib, J, parents, reference, controls=True)
    fresh, fresh_arrays = e.evaluate(case, 48, 0, lib, J, parents, reference)
    reused, arrays = e.from_frozen(case, 48, old, old_arrays)
    assert _without_reuse_flags(reused) == _without_reuse_flags(fresh)
    assert arrays.keys() == fresh_arrays.keys()
    assert all(np.array_equal(a, fresh_arrays[k]) for k, a in arrays.items())
    assert not reused['methods']['G1']['constructible']


@pytest.mark.parametrize('corruption', ['core', 'full', 'boundary', 'jets', 'target', 'source', 'gap', 'music'])
def test_frozen_import_refuses_identity_clock_or_array_corruption(data, corruption):
    case, lib, J, parents, reference = data
    old, arrays = e.frozen.evaluate_pair(case, 48, lib, J, parents, reference)
    if corruption == 'core':
        arrays[old['key']+'_left'] = arrays[old['key']+'_left'].copy()
        arrays[old['key']+'_left'][0, 0, 0, 0] += .01
    elif corruption == 'full':
        arrays[old['key']+'_G1'] = arrays[old['key']+'_G1'].copy()
        arrays[old['key']+'_G1'][250, 0, 0, 0] += .01
    elif corruption == 'boundary':
        old['boundary_readset_sha256'] = 'wrong'
    elif corruption == 'jets':
        old['boundary_jets_sha256'] = 'wrong'
    elif corruption == 'target':
        old['left']['target_interval'][0] += 1
    elif corruption == 'source':
        old['left']['event_uid'] = 'another-event'
    elif corruption == 'gap':
        old['gap_frames'] = 24
    elif corruption == 'music':
        old['music_interval'][0] += 1
    with pytest.raises(ValueError):
        e.from_frozen(case, 48, old, arrays)
