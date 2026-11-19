import copy
import json

import numpy as np
import pytest

from dunhuang.research.quality_r65.diagnostics import CONTROLS, batch, diagnostic


def inputs():
    clock = dict(frames=300, fps=30, audio_sha256='a'*64, clock_sha256='c'*64,
                 beat_frames=[0, 100, 130, 160, 190, 220, 250],
                 accent_frames=[0, 100, 130, 160, 190, 220],
                 section_proxy_frames=[99, 160, 220], accent_strength=[1.]*300)
    reward = dict(phase_peak_observations=[dict(target_frame=115), dict(target_frame=145)],
                  junction_frames=[160], primary_music_score=.123,
                  phase_beat_score=.456, phase_accent_score=.789,
                  section_proxy_score=.333, clock_sha256='c'*64)
    record = dict(key='r1', case_key='c1', music_interval=[100, 220], audio_sha256='a'*64,
                  music_reward=reward, methods=dict(G1=dict(full_R_sha256='d'*64)))
    return record, clock


def test_absolute_clock_projection_count_and_same_delta_for_all_kinds():
    record, clock = inputs()
    d = diagnostic(record, clock, [100, 220], 'HalfBeat', 15)
    assert d['beat'] == pytest.approx(1.)
    assert d['accent'] == pytest.approx(1.)
    assert d['primary'] == pytest.approx(1.)
    assert d['shift_frames'] == 15 and d['shift_seconds'] == .5
    assert d['count_exact'] is True
    assert d['counts']['beat'] == dict(before=4, after=4, fixed_query_count=2, original_full_track_events=7)
    assert d['counts']['accent']['before'] == 4
    assert d['counts']['section']['before'] == 1  # 99 and220 are outside [100,220)
    assert all(v['circular_gap_multiset_preserved'] for v in d['IOI_change'].values())
    assert d['distance_change_fraction'] == 1.
    assert d['original_r64_scalar_scores']['primary_music_score'] == .123
    assert d['original_r64_score_replaced'] is False
    assert d['sigma_frames'] == 3.
    assert 'NOT_SEMANTIC_WRONGNESS' in d['control_scope']


def test_identity_retains_local_events_and_saved_r64_score_separately():
    r, c = inputs()
    d = diagnostic(r, c, [100, 220], 'Identity', 0)
    assert d['distance_change_fraction'] == 0.
    assert d['event_time_set_change_fraction'] == dict(beat=0., accent=0., section=0.)
    assert d['primary'] != r['music_reward']['primary_music_score']
    assert d['local_identity_may_differ_from_full_track_reward'] is True


def test_periodic_nonzero_shift_can_preserve_event_set_and_distances():
    r, c = inputs()
    c['accent_frames'] = []
    c['section_proxy_frames'] = []
    d = diagnostic(r, c, [100, 220], 'OneBeatPeriodicityProbe', 30)
    assert d['shift_frames'] == 30
    assert d['event_time_set_change_fraction']['beat'] == 0.
    assert d['distance_change_by_kind']['beat'] == 0.
    assert d['distance_change_fraction'] == 0.
    assert d['primary'] is None  # beat evidence is not an invented accent observation


def test_empty_queries_or_events_return_unknown_not_zero():
    r, c = inputs()
    r['music_reward']['phase_peak_observations'] = []
    r['music_reward']['junction_frames'] = []
    d = diagnostic(r, c, [100, 220], 'Identity', 0)
    assert d['beat'] is None and d['accent'] is None and d['section'] is None
    assert d['primary'] is None and d['distance_change_fraction'] is None
    r, c = inputs()
    c.update(beat_frames=[], accent_frames=[], section_proxy_frames=[])
    d = diagnostic(r, c, [100, 220], 'HalfWindow', 60)
    assert d['primary'] is None and d['beat'] is None
    assert d['count_exact'] is True
    assert d['distance_change_fraction'] is None


def test_no_input_mutation_or_source_phase_retiming():
    r, c = inputs()
    r0, c0 = copy.deepcopy(r), copy.deepcopy(c)
    first = diagnostic(r, c, [100, 220], 'QuarterBeat', 7.5)
    second = diagnostic(r, c, [100, 220], 'QuarterBeat', 7.5)
    assert r == r0 and c == c0 and first == second
    assert first['phase_and_junction_locations_changed'] is False
    assert first['trajectories_changed'] is False
    json.dumps(first, allow_nan=False)


def test_local_frame_mistake_and_outside_timepoints_rejected():
    r, c = inputs()
    r['music_reward']['phase_peak_observations'][0]['target_frame'] = 15
    with pytest.raises(ValueError, match='absolute phase'):
        diagnostic(r, c, [100, 220], 'Identity', 0)
    r, c = inputs()
    with pytest.raises(ValueError, match='interval differ'):
        diagnostic(r, c, [0, 120], 'Identity', 0)
    c['beat_frames'] = [100, 130, 130]
    with pytest.raises(ValueError, match='sorted, unique'):
        diagnostic(r, c, [100, 220], 'Identity', 0)


@pytest.mark.parametrize('shift', [True, np.nan, np.inf, None])
def test_nonfinite_or_unregistered_scalar_type_rejected(shift):
    r, c = inputs()
    with pytest.raises(ValueError, match='Finite scalar'):
        diagnostic(r, c, [100, 220], 'Identity', shift)


def test_local_wrap_preserves_circular_ioi_even_when_linear_ioi_changes():
    r, c = inputs()
    c['beat_frames'] = [105, 125, 170, 200]
    d = diagnostic(r, c, [100, 220], 'HalfWindow', 60)
    ioi = d['IOI_change']['beat']
    assert ioi['circular_gap_multiset_preserved'] is True
    assert ioi['linear_ranked_sequence_change_fraction'] > 0
    assert ioi['circular_median_before_frames'] == ioi['circular_median_after_frames']


def test_batch_registers_five_local_controls_and_retains_repeated_selected_trajectories():
    r, c = inputs()
    r2 = copy.deepcopy(r)
    r2['key'] = 'r2'
    cases = [dict(key='c1', songidx=0, music_interval=[100, 220])]
    result = batch([r, r2], cases, {'0': c}, ['r1', 'r2', 'r1'])
    assert result['selected_occurrences'] == 3
    assert result['scalar_diagnostic_records'] == 15
    assert result['unique_selected_record_keys'] == 2
    controls = result['registration']['case_controls']['c1']['controls']
    assert [a['name'] for a in controls] == list(CONTROLS)
    assert [a['delta_frames'] for a in controls] == [0., 7.5, 15., 22.5, 60.]
    assert result['original_r64_music_rewards']['r1'] == r['music_reward']
    repeated = result['repeated_trajectory_control_audit']
    assert repeated['unique_available_trajectory_hashes'] == 1
    assert repeated['repeated_trajectories'][0]['selection_indices'] == [0, 2, 1]
    assert repeated['coordinates_read'] is False and repeated['identical_outputs_retained'] is True
    assert result['new_method_selections'] == 0 and result['quality_winner_reported'] is False
    assert [a['selection_index'] for a in result['records']] == [0]*5+[1]*5+[2]*5
    json.dumps(result, allow_nan=False)


def test_batch_missing_local_ioi_has_pending_controls_not_fake_zero_shift():
    r, c = inputs()
    c['beat_frames'] = [100]
    cases = [dict(key='c1', songidx=0, music_interval=[100, 220])]
    result = batch([r], cases, {0: c}, ['r1', None])
    assert len(result['records']) == 10
    pending = [a for a in result['records'] if a['record_key'] == 'r1' and a['control'] in CONTROLS[1:4]]
    assert len(pending) == 3 and all(a['shift_frames'] is None and a['primary'] is None for a in pending)
    assert all(a['status'] == 'PENDING_NO_FROZEN_SELECTED_RECORD' for a in result['records'][5:])


def test_batch_does_not_infer_or_replace_unknown_selected_key():
    r, c = inputs()
    with pytest.raises(ValueError, match='absent'):
        batch([r], [dict(key='c1', songidx=0, music_interval=[100, 220])], {0: c}, ['not_saved'])


def test_frozen_116_selection_occurrences_produce_exactly_580_scalar_controls():
    r, c = inputs()
    cases = [dict(key='c1', songidx=0, music_interval=[100, 220])]
    result = batch([r], cases, {0: c}, ['r1']*116)
    assert result['selected_occurrences'] == 116
    assert result['scalar_diagnostic_records'] == len(result['records']) == 580
    assert result['repeated_trajectory_control_audit']['repeated_record_occurrences'][0]['selection_indices'] == list(range(116))
