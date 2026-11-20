"""TEST_FIXTURE smooth-clock contracts; no motion generation or quality claims."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


from dunhuang.research.rhythm_r66 import warp as w


@pytest.fixture
def task():
    phases = [dict(type='upper_speed_peak' if i % 2 else 'speed_peak',
                   body_region='upper', frame=t, evidence_span=[t-1, t+2], semantic_truth='UNKNOWN')
              for i, t in enumerate((73, 139, 211))]
    phases += [dict(type='global_pose_hold', body_region='global', frame=50,
                   evidence_span=[35, 65], semantic_truth='UNKNOWN'),
               dict(type='root_turn_span', body_region='root', frame=230,
                   evidence_span=[225, 238], semantic_truth='UNKNOWN')]
    n = 1300
    peaks = [1061, 1127, 1199]
    strength = np.zeros(n); strength[peaks] = 1.
    clock = dict(frames=n, beat_frames=[990]+peaks+[1250], accent_frames=peaks,
                 accent_strength=strength.tolist(), clock_sha256='TEST_FIXTURE_CLOCK',
                 audio_sha256='TEST_FIXTURE_AUDIO')
    return [20, 260], [1000, 1240], phases, clock


def test_explicit_uniform_preserves_actual_centers_and_source_support(task):
    source, target, phases, clock = task
    result = w.uniform(source, target, phases, clock, source_support=[0, 300])
    assert result['status'] == 'LEGAL'
    assert np.array_equal(result['source_positions'], np.arange(20, 260))
    assert result['source_positions_authoritative']
    assert result['speed_penalty'] == result['curvature_penalty'] == 0.
    assert result['proof']['analytic_rate_min'] == result['proof']['analytic_rate_max'] == 1.
    assert result['endpoint_source_to_target_rates'] == [1., 1.]
    assert not result['endpoint_velocity_changed_vs_uniform']
    assert not result['fallback_used'] and not result['adjacent_bridge_revalidated']
    assert result['transition48'] == 'RESERVED_NOT_REGENERATED'


def test_pchip_is_not_certified_by_legal_secants_alone():
    x, y = np.array([0., 100., 200.]), np.array([0., 75., 208.])
    assert np.all((np.diff(y)/np.diff(x) >= .75)&(np.diff(y)/np.diff(x) <= 1.33))
    curve, proof = w._curve(x, y)
    assert curve is None
    assert proof['reason'] == 'ANALYTIC_PCHIP_RATE_OUTSIDE_FROZEN_LIMITS'
    assert proof['analytic_rate_min'] < .75 and proof['analytic_rate_max'] > 1.33


def test_search_uses_original_phase_positions_and_exact_R64_reward(task):
    source, target, phases, clock = task
    result = w.optimize(source, target, phases, clock, source_support=[0, 300])
    linear = w.uniform(source, target, phases, clock, source_support=[0, 300])
    assert result['status'] == 'LEGAL'
    assert result['objective'] >= linear['objective']-1e-12
    assert result['reward']['primary_music_score'] > linear['reward']['primary_music_score']
    q = np.asarray(result['source_positions'])
    assert len(q) == 240 and q[0] == 20 and q[-1] == 259 and np.all(np.diff(q) > 0)
    assert np.floor(q).min() >= 0 and np.ceil(q).max() < 300
    assert set(result['source_phase_anchors']) <= {p['frame'] for p in phases}
    assert len(result['source_phase_anchors']) <= 8
    assert not result['source_phase_frames_moved']
    assert result['proof']['analytic_rate_min'] >= .75-1e-10
    assert result['proof']['analytic_rate_max'] <= 1.33+1e-10
    assert result['proof']['dense_checks'] >= 16
    assert result['proof']['rate_units'] == 'TARGET_FRAMES_PER_SOURCE_FRAME'
    assert result['proof']['curvature_units'] == 'TARGET_FRAMES_PER_SOURCE_FRAME_SQUARED'
    assert result['candidate_counts']['complete_paths_screened'] <= w.CONFIG['beam_width']+1
    assert result['candidate_counts']['uniform_candidate_registered']
    assert result['candidate_counts']['legal_paths_scored'] >= 1
    b = w._reward(result['target_peak_frames'], clock['beat_frames'], 3)
    a = w._reward(result['target_peak_frames'], clock['accent_frames'], 3, [1., 1., 1.])
    assert result['reward']['primary_music_score'] == .5*b['score']+.5*a['score']
    assert not result['adjacent_bridge_revalidated']


def test_local_shift_control_is_repeatable_keeps_strengths_and_is_not_artistic_truth(task):
    source, target, phases, clock = task
    one = w.optimize(source, target, phases, clock, control=True)
    two = w.optimize(source, target, phases, clock, control=True)
    assert one == two
    assert one['status'] == 'LEGAL' and one['clock']['local_event_shift_frames'] == 120
    assert one['clock']['original_evidence_sha256'] != one['clock']['used_evidence_sha256']
    assert not one['clock']['audio_waveform_changed'] and not one['clock']['outside_core_events_changed']
    assert 'NOT_ARTISTIC_INCOMPATIBILITY' in one['clock']['control_scope']
    beats, accents, values, _ = w._clock(clock, target, True)
    assert 990 in beats and 1250 in beats
    assert np.array_equal(accents, [1007, 1079, 1181])
    assert np.array_equal(values, [1., 1., 1.])


def test_endpoint_duration_illegal_is_not_cropped_or_replaced(task):
    _, target, phases, clock = task
    result = w.optimize([0, 100], target, phases, clock)
    assert result['status'] == 'ILLEGAL'
    assert result['reason'] == 'FIXED_ENDPOINT_RATE_OUTSIDE_FROZEN_LIMITS'
    assert 'source_positions' not in result and not result['fallback_used']
    with pytest.raises(ValueError, match='240'):
        w.optimize([0, 240], [1000, 1239], phases, clock)
    with pytest.raises(ValueError, match='support'):
        w.optimize([20, 260], target, phases, clock, source_support=[21, 300])


def test_no_speed_peak_remains_unknown_not_a_zero_music_score(task):
    source, target, phases, clock = task
    hold = [p for p in phases if 'hold' in p['type']]
    result = w.optimize(source, target, hold, clock)
    assert result['status'] == 'NOT_EVALUABLE'
    assert 'source_positions' not in result
    baseline = w.uniform(source, target, hold, clock)
    assert baseline['status'] == 'LEGAL'
    assert baseline['reward']['primary_music_score'] is None and baseline['objective'] is None


def test_anchor_cap_is_metadata_only_and_phase_frames_never_move(task):
    source, _, phases, _ = task
    many = phases+[dict(type='lower_speed_peak', body_region='lower', frame=t) for t in range(30, 250, 10)]
    selected = w._anchors(w._phases(many, source), source)
    assert len(selected) == 8
    assert selected == sorted(set(selected))
    assert all(v in {p['frame'] for p in many} for v in selected)
    assert 50 in selected and 230 in selected
