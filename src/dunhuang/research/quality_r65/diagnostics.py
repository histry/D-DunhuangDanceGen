"""Local count-preserving audio-event controls on frozen R64 selections.

Only saved scalar clocks and saved phase/junction locations are read. These
controls neither generate motion nor select a method. Their Gaussian proxy
responses are not independent music quality or semantic incompatibility truth.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from dunhuang.research.timing_r64.music import _reward


FPS = 30
EPS = 1e-8
KINDS = dict(beat='beat_frames', accent='accent_frames', section='section_proxy_frames')
CONTROLS = ('Identity', 'QuarterBeat', 'HalfBeat', 'ThreeQuarterBeat', 'HalfWindow')
REGISTRATION = dict(version='R65_LOCAL_CIRCULAR_EVENT_COUNT_CONTROLS_V1',
    controls=list(CONTROLS), beat_fractions=[0., .25, .5, .75], half_window_fraction=.5,
    median_IOI='MEDIAN_ADJACENT_LINEAR_INTERVALS_BETWEEN_BEATS_INSIDE_THIS_WINDOW',
    all_kinds_shifted_by_same_delta=True, sigma_frames=3.,
    primary='0.5*beat+0.5*accent; UNKNOWN_IF_EITHER_MISSING; SECTION_SEPARATE',
    source='SAVED_R64_MUSIC_REWARD_PHASE_PEAKS_AND_JUNCTIONS_ONLY',
    local_evidence='ONLY_AUDIO_EVENTS_IN_REGISTERED_HALF_OPEN_MUSIC_WINDOW',
    distance_change='FRACTION_OF_FIXED_PHASE_OR_JUNCTION_QUERIES_WHOSE_NEAREST_EVENT_DISTANCE_CHANGES',
    event_set_change='FRACTION_OF_ORIGINAL_EVENT_TIMES_WITHOUT_A_MATCH_IN_SHIFTED_TIME_SET_WITHIN_1e-8_FRAMES',
    IOI_scope='SORTED_LINEAR_INTERVALS_MAY_CHANGE_AT_WRAP; CIRCULAR_GAP_MULTISET_MUST_BE_PRESERVED',
    missing_median_IOI='BEAT_FRACTION_CONTROLS_PENDING; NO_ZERO_OR_GLOBAL_TEMPO_SUBSTITUTION',
    original_r64_scores_replaced=False, new_geometry_solves=0, new_method_selections=0,
    quality_winner_reported=False,
    control_scope='LOCAL_AUDIO_EVENT_PROXY_TIME_SENSITIVITY_ONLY; CIRCULAR_PERIODICITY_MAY_SURVIVE; NOT_SEMANTIC_WRONGNESS_OR_INDEPENDENT_QUALITY')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def _vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional '+name+' required')
    return x


def _interval(interval):
    if len(interval) != 2 or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) for t in interval):
        raise ValueError('Integer half-open absolute music interval required')
    A, B = map(int, interval)
    if not 0 <= A < B:
        raise ValueError('Positive half-open absolute music support required')
    return A, B


def _local_events(clock, interval):
    A, B = _interval(interval)
    if clock.get('fps', FPS) != FPS or B > clock['frames']:
        raise ValueError('Registered 30FPS interval must fit the actual audio clock')
    result = {}
    for kind, key in KINDS.items():
        x = _vector(clock.get(key, []), key)
        if len(x) and (x[0] < 0 or x[-1] >= clock['frames'] or (np.diff(x) <= 0).any()):
            raise ValueError('Audio event clocks must be sorted, unique actual frame centers')
        result[kind] = x[(A <= x) & (x < B)]-A
    return result


def _query_points(record, interval):
    A, B = _interval(interval)
    if list(record.get('music_interval', interval)) != [A, B]:
        raise ValueError('Frozen record and diagnostic interval differ')
    reward = record.get('music_reward')
    if not isinstance(reward, dict):
        raise ValueError('Saved R64 music_reward is required; do not recompute phase locations')
    phases = _vector([p['target_frame'] for p in reward.get('phase_peak_observations', [])], 'saved phase frames')
    junctions = _vector(reward.get('junction_frames', []), 'saved junction frames')
    for name, x in (('phase', phases), ('junction', junctions)):
        if len(x) and ((x < A).any() or (x >= B).any()):
            raise ValueError('Saved absolute '+name+' locations leave this music window')
    return dict(beat=phases-A, accent=phases-A, section=junctions-A)


def _ioi(values, length):
    linear = np.diff(values)
    circular = np.diff(np.r_[values, values[0]+length]) if len(values) else np.array([])
    return linear, circular


def _ioi_change(original, shifted, length):
    a, ac = _ioi(original, length)
    b, bc = _ioi(shifted, length)
    return dict(linear_interval_count=len(a),
        linear_median_before_frames=float(np.median(a)) if len(a) else None,
        linear_median_after_frames=float(np.median(b)) if len(b) else None,
        linear_mean_before_frames=float(a.mean()) if len(a) else None,
        linear_mean_after_frames=float(b.mean()) if len(b) else None,
        linear_ranked_sequence_change_fraction=float(np.mean(abs(a-b) > EPS)) if len(a) else None,
        circular_gap_count=len(ac),
        circular_gap_multiset_preserved=bool(np.allclose(np.sort(ac), np.sort(bc), rtol=0, atol=EPS)),
        circular_median_before_frames=float(np.median(ac)) if len(ac) else None,
        circular_median_after_frames=float(np.median(bc)) if len(bc) else None)


def _fraction_changed(before, after):
    if not len(before) or not len(after):
        return None
    nearest = np.min(abs(before[:, None]-after[None]), axis=1)
    return float(np.mean(nearest > EPS))


def diagnostic(record, clock, interval, control_name, delta_frames):
    """Evaluate one registered coherent local shift; mutate none of its inputs."""
    if not isinstance(control_name, str) or not control_name:
        raise ValueError('An explicit control identity is required')
    if isinstance(delta_frames, (bool, np.bool_)) or not isinstance(delta_frames, (int, float, np.integer, np.floating)) or not np.isfinite(delta_frames):
        raise ValueError('Finite scalar registered shift required')
    A, B = _interval(interval)
    original = _local_events(clock, [A, B])
    queries = _query_points(record, [A, B])
    reward = record['music_reward']
    if record.get('audio_sha256', clock.get('audio_sha256')) != clock.get('audio_sha256'):
        raise ValueError('Frozen motion plan and clock audio identities differ')
    if reward.get('clock_sha256', clock.get('clock_sha256')) != clock.get('clock_sha256'):
        raise ValueError('Frozen reward and supplied music clock identities differ')
    L, delta = B-A, float(delta_frames)
    components, counts, digests, ioi = {}, {}, {}, {}
    distance_by_kind, set_changes, changed_queries, total_queries = {}, {}, 0, 0
    for kind, x in original.items():
        unranked = (x+delta) % L
        order = np.argsort(unranked, kind='stable')
        shifted = unranked[order]
        strengths = None
        if kind == 'accent' and clock.get('accent_strength') is not None:
            strength = _vector(clock['accent_strength'], 'accent_strength')
            if len(strength) != clock['frames'] or (strength < 0).any():
                raise ValueError('Actual finite nonnegative full-audio accent strengths required')
            strengths = np.interp(x+A, np.arange(clock['frames']), strength)
        before = _reward(queries[kind], x, 3., strengths)
        after = _reward(queries[kind], shifted, 3., strengths[order] if strengths is not None else None)
        components[kind] = dict(score=after['score'], local_identity_score=before['score'],
            mean_nearest_error_frames=after['mean_nearest_error_frames'],
            mean_nearest_error_seconds=after['mean_nearest_error_frames']/FPS if after['mean_nearest_error_frames'] is not None else None)
        distances0 = np.asarray(before['per_event_distance_frames'])
        distances1 = np.asarray(after['per_event_distance_frames'])
        change_count = int(np.sum(abs(distances0-distances1) > EPS))
        distance_by_kind[kind] = change_count/len(distances0) if len(distances0) else None
        changed_queries += change_count
        total_queries += len(distances0)
        set_changes[kind] = _fraction_changed(x, shifted)
        counts[kind] = dict(before=len(x), after=len(shifted), fixed_query_count=len(queries[kind]),
                            original_full_track_events=len(clock.get(KINDS[kind], [])))
        digests[kind] = _digest(dict(local_event_frames=shifted.tolist(),
            event_strengths=strengths[order].tolist() if strengths is not None else None))
        ioi[kind] = _ioi_change(x, shifted, L)
    beat, accent, section = (components[k]['score'] for k in ('beat', 'accent', 'section'))
    primary = .5*beat+.5*accent if beat is not None and accent is not None else None
    original_scalars = {name: reward.get(name) for name in (
        'phase_beat_score', 'phase_accent_score', 'section_proxy_score', 'primary_music_score')}
    return dict(record_key=record.get('key'), case_key=record.get('case_key'), control=control_name,
        status='PROXY_DIAGNOSTIC_ONLY' if primary is not None else 'NOT_EVALUABLE_MISSING_LOCAL_PROXY_EVIDENCE',
        beat=beat, accent=accent, section=section, primary=primary,
        components=components, counts=counts,
        count_exact=all(c['before'] == c['after'] for c in counts.values()),
        music_interval=[A, B], interval_frames=L, shift_frames=delta,
        shift_seconds=delta/FPS, circular_shift_frames=delta % L,
        shifted_event_digest=digests, IOI_change=ioi,
        distance_change_fraction=changed_queries/total_queries if total_queries else None,
        distance_change_by_kind=distance_by_kind, event_time_set_change_fraction=set_changes,
        original_r64_scalar_scores=original_scalars, original_r64_music_reward_sha256=_digest(reward),
        original_r64_score_replaced=False, local_identity_may_differ_from_full_track_reward=True,
        phase_and_junction_locations_changed=False, trajectories_changed=False,
        clock_sha256=clock.get('clock_sha256'), audio_sha256=clock.get('audio_sha256'),
        control_scope=REGISTRATION['control_scope'], sigma_frames=3.,
        independent_quality_metric=False, semantic_wrongness_truth='UNKNOWN',
        new_method_selections=0, geometry_solver_calls=0, optimizer_calls=0)


def _controls(clock, interval):
    events = _local_events(clock, interval)['beat']
    median = float(np.median(np.diff(events))) if len(events) >= 2 else None
    A, B = _interval(interval)
    return median, [('Identity', 0.), ('QuarterBeat', median*.25 if median is not None else None),
        ('HalfBeat', median*.5 if median is not None else None),
        ('ThreeQuarterBeat', median*.75 if median is not None else None), ('HalfWindow', (B-A)/2)]


def batch(records, cases, clocks, selected_keys):
    """Preserve every supplied frozen selection occurrence, including repeats."""
    records = list(records)
    index = {r['key']: r for r in records}
    if len(index) != len(records):
        raise ValueError('Frozen record identities must be unique')
    case_index = {c['key']: c for c in cases}
    if len(case_index) != len(cases):
        raise ValueError('Frozen case identities must be unique')
    if any(key is not None and not isinstance(key, str) for key in selected_keys):
        raise ValueError('Only the ordered list of frozen record keys may be passed')
    result, originals, registrations, occurrences = [], {}, {}, {}
    for i, key in enumerate(selected_keys):
        if key is None:
            result.extend(dict(selection_index=i, record_key=None, control=name,
                status='PENDING_NO_FROZEN_SELECTED_RECORD', beat=None, accent=None, section=None,
                primary=None, count_exact=None, control_scope=REGISTRATION['control_scope']) for name in CONTROLS)
            continue
        if key not in index:
            raise ValueError('Requested key is absent from the frozen R64 result: '+key)
        r = index[key]
        c = case_index[r['case_key']]
        song = c['songidx']
        clock = clocks[song] if song in clocks else clocks[str(song)]
        interval = c['music_interval']
        median, controls = _controls(clock, interval)
        registrations[c['key']] = dict(music_interval=list(interval), songidx=int(song),
            clock_sha256=clock.get('clock_sha256'), local_median_beat_IOI_frames=median,
            controls=[dict(name=name, delta_frames=delta, delta_seconds=delta/FPS if delta is not None else None)
                      for name, delta in controls])
        originals[key] = deepcopy(r['music_reward'])
        occurrences.setdefault(key, []).append(i)
        for name, delta in controls:
            if delta is None:
                result.append(dict(selection_index=i, record_key=key, case_key=r['case_key'], control=name,
                    status='PENDING_LOCAL_MEDIAN_BEAT_IOI_UNAVAILABLE', beat=None, accent=None, section=None,
                    primary=None, shift_frames=None, shift_seconds=None, count_exact=None,
                    control_scope=REGISTRATION['control_scope']))
            else:
                result.append(dict(selection_index=i, **diagnostic(r, clock, interval, name, delta)))
    hashes = {}
    for key, ids in occurrences.items():
        h = index[key].get('methods', {}).get('G1', {}).get('full_R_sha256')
        if h is not None:
            group = hashes.setdefault(h, dict(full_R_sha256=h, record_keys=[], selection_indices=[]))
            group['record_keys'].append(key)
            group['selection_indices'].extend(ids)
    registration = deepcopy(REGISTRATION)
    registration['case_controls'] = registrations
    return dict(registration=registration, registration_sha256=_digest(registration), records=result,
        selected_occurrences=len(selected_keys), unique_selected_record_keys=len(occurrences),
        controls_per_occurrence=len(CONTROLS), scalar_diagnostic_records=len(result),
        original_r64_music_rewards=originals,
        repeated_trajectory_control_audit=dict(
            ordered_selected_keys=list(selected_keys), repeated_record_occurrences=[
                dict(record_key=key, selection_indices=ids) for key, ids in sorted(occurrences.items()) if len(ids) > 1],
            unique_available_trajectory_hashes=len(hashes),
            repeated_trajectories=[v for h, v in sorted(hashes.items()) if len(v['selection_indices']) > 1],
            missing_trajectory_hash_record_keys=[k for k in occurrences
                if index[k].get('methods', {}).get('G1', {}).get('full_R_sha256') is None],
            identical_outputs_retained=True, coordinates_read=False),
        new_method_selections=0, geometry_solver_calls=0, optimizer_calls=0,
        independent_quality_metric=False, quality_winner_reported=False, human_status='PENDING')
