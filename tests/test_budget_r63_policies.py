import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.budget_r63.policies import (
    FEATURE_NAMES, boundary_features, digest, feature_row, fit, predict, select,
    shortest_screened)
from dunhuang.research.budget_r63.population import (
    evaluation_cases, native, role_hash_isolation, select_eval_units)


def row(key, value, label, role='DEV'):
    return dict(key=key, role=role, features=[float(value)]*len(FEATURE_NAMES),
                shortest_gap=label, left_source_uid='dev_left', right_source_uid='dev_right')


def candidate(gap, passed=False, constructible=True, status='GEOMETRY_EVALUATED'):
    return dict(gap_frames=gap, status=status, methods=dict(G1=dict(
        constructible=constructible, passes_original_screen=passed)))


def test_fit_is_dev_only_and_standardizes_only_usable_dev_rows():
    model = fit([row('b', 3, 48), row('a', 1, 24), row('missing', 100000, None)])
    assert model['normalization_mean'] == [2.]*5
    assert model['normalization_std'] == [1.]*5
    assert model['fitted_cases'] == 2 and model['input_development_cases'] == 3
    assert model['excluded_cases'] == [dict(key='missing', reason='NO_PASSING_FIXED_BUDGET_G1_LABEL')]
    assert model['development_case_keys'] == ['a', 'b', 'missing']
    assert model['fitted_source_uids'] == ['dev_left', 'dev_right']
    assert model['evaluation_labels_read'] is False and model['optimizer_calls'] == 0
    assert model['policy_sha256'] == digest(model)
    with pytest.raises(ValueError, match='DEV'):
        fit([row('eval', 999, 72, role='EVAL')])


def test_1nn_ties_are_by_key_and_input_order_does_not_change_fit():
    first = fit([row('z', 2, 72), row('a', 0, 24)])
    second = fit([row('a', 0, 24), row('z', 2, 72)])
    assert first == second
    decision = predict(first, [1.]*5)
    assert decision['nearest_development_key'] == 'a'
    assert decision['gap_frames'] == 24
    assert decision['status'] == 'PREDICTED_UNSCREENED'
    assert decision['candidate_outcomes_read'] is False and decision['fallback_used'] is False


def test_missing_labels_never_become_long_duration_positives():
    model = fit([row('failed', 0, None)])
    assert model['status'] == 'NOT_FIT' and model['fitted_cases'] == 0
    assert model['fitted_rows'] == []
    assert predict(model, [0.]*5)['gap_frames'] is None
    assert predict(model, [0.]*5)['status'] == 'MODEL_NOT_FIT'


def test_missing_features_and_no_prediction_fallback():
    bad = row('bad', 0, 24)
    bad['features'] = None
    model = fit([row('valid', 0, 24), bad])
    assert model['fitted_cases'] == 1
    assert model['excluded_cases'][0]['reason'] == 'MISSING_OR_NONFINITE_BASELINE48_FEATURES'
    assert predict(model, [np.nan]*5)['gap_frames'] is None
    assert select('BoundaryConditioned', model=model, features=None)['status'] == 'BASELINE48_FEATURES_UNAVAILABLE'
    with pytest.raises(ValueError, match='outcomes'):
        select('BoundaryConditioned', records=[candidate(48, True)], model=model, features=[0.]*5)


def test_policy_serialization_is_exact_and_tamper_rejected():
    model = fit([row('a', 0, 24)])
    restored = json.loads(json.dumps(model, allow_nan=False))
    assert predict(restored, [0.]*5) == predict(model, [0.]*5)
    changed = copy.deepcopy(model)
    changed['normalization_mean'][0] += 1
    with pytest.raises(ValueError, match='hash'):
        predict(changed, [0.]*5)


@pytest.mark.parametrize('label', [25, True, -1, 24.0])
def test_invalid_labels_rejected(label):
    with pytest.raises(ValueError, match='targets'):
        fit([row('a', 0, label)])


def test_shortest_screened_requires_actual_retimed_g1_pass():
    records = [candidate(72, True), candidate(24, True, constructible=False),
               candidate(36, True, status='CORE_RECOMPILE_REJECTED'), candidate(48, True)]
    assert shortest_screened(records) == 48
    assert shortest_screened([]) is None
    assert shortest_screened([candidate(48, True)]) is None  # unexecuted24/36 are unknown
    assert shortest_screened([candidate(24), candidate(48)]) is None
    assert select('ShortestScreened', records=[candidate(48)])['status'] == 'NO_PASSING_G1'
    assert select('Fixed48')['gap_frames'] == 48
    assert select('Fixed24')['gap_frames'] == 24
    with pytest.raises(ValueError, match='Duplicate'):
        shortest_screened([candidate(24), candidate(24)])


def rotations(rate, length=12):
    theta = np.arange(length)*rate/30
    return Rotation.from_rotvec(np.tile(theta[:, None, None]*np.array([0., 0., 1.]),
                                       (1, 22, 1)).reshape(-1, 3)).as_matrix().reshape(length, 22, 3, 3)


def test_boundary_features_only_read_observed_edges():
    L, R = rotations(.2), rotations(.4)
    before = boundary_features(L, R)
    L[:-4] = np.nan
    R[4:] = np.nan
    assert np.allclose(boundary_features(L, R), before)
    assert before[1] == pytest.approx(.2/np.sqrt(3))
    assert before[2] == pytest.approx(.4/np.sqrt(3))
    assert before[3] < 1e-12 and before[4] < 1e-12
    L[-1, 0] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        boundary_features(L, R)


def test_feature_row_rejects_other_duration_or_other_case():
    case = dict(key='a', role='DEV', left=dict(source_uid='x'), right=dict(source_uid='y'))
    base = dict(key='a_g48', case_key='a', gap_frames=48, status='GEOMETRY_EVALUATED')
    arrays = dict(a_g48_left=rotations(.2), a_g48_right=rotations(.4))
    result = feature_row(case, base, arrays)
    assert result['features'] == boundary_features(arrays['a_g48_left'], arrays['a_g48_right'])
    assert result['feature_source'] == 'RECOMPILED_FIXED48_VISIBLE_EDGE_POSES_ONLY'
    with pytest.raises(ValueError, match='fixed48'):
        feature_row(case, dict(base, gap_frames=24), arrays)


def unit(uid, start, length, name=None):
    return dict(unit_uid=name or uid+str(start), source_uid=uid,
                source_interval=[np.int64(start), np.int64(start+length)],
                support_component=[np.int64(0), np.int64(2000)],
                phases=[dict(frame=np.int64(start+5), type='speed_peak', body_region='upper')])


def test_eval_unit_choice_is_closest_then_chronology_without_substitution():
    units = [unit('x', 500, 240, 'later'), unit('x', 100, 240, 'first'),
             unit('x', 0, 200, 'farther'), unit('bad', 0, 350), unit('bad', 500, 360),
             unit('dev', 0, 240)]
    selected = select_eval_units(units, ['x', 'bad', 'missing', 'dev'], ['dev'])
    lookup = {r['source_uid']: r for r in selected}
    assert lookup['x']['unit']['event_uid'] == 'first'
    assert lookup['bad']['status'] == 'SELECTED'
    assert lookup['bad']['baseline_length_legal'] is False
    assert lookup['bad']['unit']['source_interval'] == [0, 350]
    assert lookup['missing']['status'] == 'NOT_MATCHABLE'
    assert lookup['missing']['unit'] is None
    assert all(r['replacement_used'] is False for r in selected)
    json.dumps(selected, allow_nan=False)


def test_all_twenty_eval_pairs_have_fixed_budget_deterministic_music_and_native_ids():
    selected = select_eval_units([unit('s%d' % i, 0, 240) for i in range(5)],
                                 ['s%d' % i for i in range(5)], [])
    tracks = [dict(path='song%d.wav' % i, audio_sha256='audio%d' % i,
                   structure_sha256='struct%d' % i,
                   structure=dict(frames=3600+i, beats=[0, 15, 30])) for i in range(3)]
    cases = evaluation_cases(selected, tracks)
    assert len(cases) == 20
    assert len({c['source_pair_uid'] for c in cases}) == 20
    for i, case in enumerate(cases):
        assert case['songidx'] == i % 3
        assert case['music_interval'][1]-case['music_interval'][0] == 528
        assert case['baseline_left_frames']+case['baseline_gap_frames']+case['baseline_right_frames'] == 528
        assert case['left']['source_uid'] != case['right']['source_uid']
        assert case['role'] == 'EVAL' and case['status'] == 'REGISTERED'
        assert case['audio_sha256'] == 'audio%d' % (i % 3)
        assert all(type(v) is int for v in case['left']['source_interval'])
    assert cases == evaluation_cases(list(reversed(selected)), tracks)
    json.dumps(cases, allow_nan=False)


def test_missing_eval_source_keeps_pair_population():
    selected = select_eval_units([unit('a', 0, 240)], ['a', 'b'], [])
    track = dict(path='x.wav', audio_sha256='x', structure_sha256='y',
                 structure=dict(frames=600, beats=[]))
    cases = evaluation_cases(selected, [track]*3)
    assert len(cases) == 2 and all(c['status'] == 'NOT_MATCHABLE' for c in cases)
    assert all(c['missing_source_uids'] == ['b'] for c in cases)
    assert native({'id': np.int64(1), 'frames': np.array([2, 3])}) == {'id': 1, 'frames': [2, 3]}


def test_aliased_native_sha_across_distinct_uids_rejected():
    sources = {'dev_uid': [dict(reference=dict(sha256='a'*64), source_sha256='b'*64)],
               'eval_alias': [dict(reference=dict(sha256='a'*64), source_sha256='c'*64)]}
    with pytest.raises(ValueError, match='native_sha256 alias'):
        role_hash_isolation(sources, ['dev_uid'], ['eval_alias'])


def test_aliased_raw_sha_across_distinct_native_versions_rejected():
    sources = {'dev_uid': [dict(reference=dict(sha256='a'*64), source_sha256='c'*64)],
               'eval_alias': [dict(reference=dict(sha256='b'*64), source_sha256='c'*64)]}
    with pytest.raises(ValueError, match='raw_source_sha256 alias'):
        role_hash_isolation(sources, ['dev_uid'], ['eval_alias'])


def test_hash_disjointness_never_claims_recording_independence_and_retains_missing_raw():
    sources = {'dev_uid': [dict(reference=dict(sha256='a'*64), source_sha256='c'*64)],
               'eval_uid': [dict(reference=dict(sha256='b'*64))]}
    result = role_hash_isolation(sources, ['dev_uid'], ['eval_uid'])
    assert result['native_hash_disjoint'] and result['available_raw_source_hash_disjoint']
    assert result['raw_source_hash_coverage_complete'] is False
    assert result['raw_source_hash_missing'] == dict(DEV=[], EVAL=['eval_uid'])
    assert result['recording_independence_verified'] is False
    assert result['overlaps'] == dict(native_sha256=[], raw_source_sha256=[])
    json.dumps(result, allow_nan=False)
