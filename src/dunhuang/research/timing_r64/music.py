"""Audio-only MusicClock and frozen proxy timing rewards for R64.

Detected beats, spectral-flux accents and novelty boundaries are observations or
automatic proxies, never independent phrase or semantic event ground truth.
The objective is registered before any duration candidate is evaluated.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from dunhuang.research.complete_r36.core import read, sha
from dunhuang.research.music_structure_r56.music import analyze
from dunhuang.research.music_structure_r56.common import CFG as R56_CFG
from dunhuang.research.warp_coupling_r53.common import array_hash


FPS = 30
SCORE_PROTOCOL = dict(version='R64_FIXED_AUDIO_PROXY_REWARD_V1',
    phase_tolerance_frames=3, boundary_tolerance_frames=15,
    kernel='EXP_MINUS_SQUARED_DISTANCE_OVER_TWO_SIGMA_SQUARED',
    primary_weights=dict(phase_beat_score=.5, phase_accent_score=.5),
    boundary_rewards_in_primary=False,
    boundary_reward_scope='SEPARATE_SECTION_PROXY_TERM; ROOT_POLICY_FREEZES_ITS_WEIGHT; PHRASE_DOWNBEAT_ZERO',
    outcomes_used_to_choose_weights=False,
    labels_used='R47_KINEMATIC_SPEED_PEAKS_AND_AUDIO_ONLY_AUTOMATIC_PROXIES',
    independent_quality_metric=False)


def _digest(value):
    data = {k: v for k, v in value.items() if k != 'clock_sha256'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def _finite(value, size=None, name='curve'):
    data = np.asarray(value, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all() or size is not None and len(data) != size:
        raise ValueError('Finite clock-aligned one-dimensional '+name+' required')
    return data


def _events(value, frames, name):
    data = _finite(value, name=name)
    if len(data) and (data[0] < 0 or data[-1] > frames-1 or np.any(np.diff(data) <= 0)):
        raise ValueError(name+' must be sorted, unique actual audio frame centers')
    return data


def validate_clock(clock):
    """Validate sample/second/frame agreement without granting label quality."""
    if clock.get('schema') != 'MusicClockR64V1' or clock.get('fps') != FPS:
        raise ValueError('Registered MusicClockR64V1 at 30FPS required')
    for name in ('audio_sha256', 'music_structure_sha256'):
        value = clock.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
            raise ValueError('Valid source '+name+' required')
    for name in ('sample_rate', 'audio_samples', 'frames'):
        if type(clock.get(name)) is not int or clock[name] <= 0:
            raise ValueError('Actual positive integer '+name+' required')
    frames, sr, samples = clock['frames'], clock['sample_rate'], clock['audio_samples']
    if frames != (samples*FPS+sr-1)//sr:
        raise ValueError('Music clock frame count does not cover the exact original audio samples')
    if clock['duration_seconds'] != samples/sr:
        raise ValueError('Audio duration must retain actual sample precision')
    times = _finite(clock['frame_times_seconds'], frames, 'frame seconds')
    if not np.array_equal(times, np.arange(frames)/FPS) or times[-1] >= samples/sr:
        raise ValueError('Frame timestamps must be actual 30FPS audio frame centers')
    beats = _events(clock['beat_frames'], frames, 'beats')
    if not np.array_equal(_finite(clock['beat_seconds'], len(beats), 'beat seconds'), beats/FPS):
        raise ValueError('Beat frame and second clocks disagree')
    for name in ('onset_strength', 'accent_strength', 'phrase_proxy_strength', 'section_proxy_strength'):
        if (_finite(clock[name], frames, name) < 0).any():
            raise ValueError('Automatic evidence strength may not be negative')
    for name in ('accent_frames', 'phrase_proxy_frames', 'section_proxy_frames'):
        _events(clock[name], frames, name)
    tempo = clock['tempo_curve_bpm']
    if len(tempo) != frames or any(v is not None and (not np.isfinite(v) or v <= 0) for v in tempo):
        raise ValueError('Positive tempo observations or explicit UNKNOWN values required')
    if clock.get('downbeat_frames') is not None or clock.get('independent_phrase_labels') is not None:
        raise ValueError('Automatic MusicClock cannot invent downbeat or independent phrase labels')
    if clock.get('human_reviewed') is not False or clock.get('operational_qualified') is not False:
        raise ValueError('MusicClock construction does not establish human or operational qualification')
    if clock.get('clock_sha256') is not None and clock['clock_sha256'] != _digest(clock):
        raise ValueError('Music clock hash mismatch')
    return True


def clock_from_features(songidx, track, arrays, metadata):
    """Use frozen R56 features and already recorded real-audio beat timestamps."""
    frames = int(metadata['frames'])
    samples, sr = int(metadata['audio_samples']), int(metadata['sample_rate'])
    if metadata['audio_sha256'] != track['audio_sha256']:
        raise ValueError('Audio feature hash differs from the sealed music source')
    hashes = {k: array_hash(v) for k, v in arrays.items()}
    if metadata.get('arrays') != hashes:
        raise ValueError('R56 feature arrays differ from their extraction receipt')
    if abs(frames-int(track['structure']['frames'])) > 1:
        raise ValueError('Legacy audio beat clock does not span this actual waveform')
    beats = _events(track['structure']['beats'], frames, 'beats')
    accent = _finite(arrays['accent_curve'], frames, 'R56 spectral-flux accent')
    phrase = _finite(arrays['held_boundary'], frames, 'R56 chroma novelty')
    section = _finite(arrays['proposal'], frames, 'R56 spectral novelty')
    accents = _events(arrays['proxy_accents'], frames, 'R56 accent events')
    phrase_events = _events(arrays['proxy_boundaries'], frames, 'R56 chroma boundary proxies')
    section_events = find_peaks(section,
        distance=int(R56_CFG['proxy_boundary_distance_frames']),
        prominence=float(R56_CFG['proxy_boundary_prominence']))[0]
    # This is a beat-interval estimate, not an independently measured tempo GT.
    if len(beats) >= 2:
        tempo_midpoints = (beats[:-1]+beats[1:])/2
        interval_bpm = 60*FPS/np.diff(beats)
        tempo = np.interp(np.arange(frames), tempo_midpoints, interval_bpm).tolist()
    else:
        tempo_midpoints, interval_bpm = np.array([]), np.array([])
        tempo = [None]*frames
    clock = dict(schema='MusicClockR64V1', songidx=int(songidx),
        audio_path=str(track['path']), audio_sha256=track['audio_sha256'],
        music_structure_sha256=track['structure_sha256'],
        audio_samples=samples, sample_rate=sr, frames=frames, fps=FPS,
        duration_seconds=samples/sr, frame_times_seconds=(np.arange(frames)/FPS).tolist(),
        beat_frames=beats.tolist(), beat_seconds=(beats/FPS).tolist(),
        beats=beats.tolist(), beat_scope='RECORDED_AUDIO_DETECTOR_OUTPUT_NOT_HUMAN_BEAT_TRUTH',
        beat_clock_source='SEALED_R36_AUDIO_STRUCTURE; MOTION_NOT_USED',
        tempo_curve_bpm=tempo, tempo_observation_seconds=(tempo_midpoints/FPS).tolist(),
        tempo_observation_bpm=interval_bpm.tolist(),
        tempo_scope='INTERPOLATED_ADJACENT_DETECTED_BEAT_INTERVAL_PROXY',
        onset_strength=accent.tolist(), accent_strength=accent.tolist(),
        onset_scope='R56_POSITIVE_LOG_SPECTRAL_DIFFERENCE_NOT_A_NEW_INDEPENDENT_DETECTOR',
        accent_frames=accents.tolist(), accent_seconds=(accents/FPS).tolist(),
        accent_scope='R56_AUDIO_SPECTRAL_FLUX_PEAK_PROXY',
        phrase_proxy_frames=phrase_events.tolist(), phrase_proxy_strength=phrase.tolist(),
        phrase_proxy_scope='R56_CHROMA_NOVELTY_BOUNDARY_CANDIDATES_NOT_ANNOTATED_PHRASES',
        section_proxy_frames=section_events.astype(int).tolist(), section_proxy_strength=section.tolist(),
        section_proxy_scope='R56_LOG_SPECTRAL_NOVELTY_CANDIDATES_NOT_ANNOTATED_SECTIONS',
        downbeat_frames=None, downbeat_status='UNKNOWN_NO_INDEPENDENT_EVIDENCE',
        independent_phrase_labels=None, phrase_truth='UNKNOWN_NO_INDEPENDENT_ANNOTATIONS',
        section_truth='UNKNOWN_NO_INDEPENDENT_ANNOTATIONS',
        human_annotations=dict(status='UNAVAILABLE', records=[], verified=False),
        human_reviewed=False, operational_qualified=False,
        exposure='HISTORICALLY_EXPOSED_DUNHUANG_DEVELOPMENT_TRACK',
        feature_version=metadata['feature_version'], feature_metadata=deepcopy(metadata),
        feature_array_hashes=hashes,
        score_protocol=deepcopy(SCORE_PROTOCOL), motion_used_to_extract_music=False,
        independent_music_truth=False)
    validate_clock(clock)
    clock['clock_sha256'] = _digest(clock)
    return clock


def build(tracks=None):
    """Return clocks by integer song index; default population is the three songs."""
    if tracks is None:
        from dunhuang.research.budget_r63.population import _tracks
        tracks = _tracks()
    clocks = {}
    for song, track in enumerate(tracks):
        if sha(track['path']) != track['audio_sha256']:
            raise ValueError('Original audio bytes differ from the sealed track')
        arrays, metadata = analyze(track['path'])
        clocks[song] = clock_from_features(song, track, arrays, metadata)
    return clocks


def import_human_labels(clock, path):
    """Attach optional audit records; schema validity alone never verifies truth."""
    validate_clock(clock)
    document = read(path)
    if document.get('audio_sha256') != clock['audio_sha256']:
        raise ValueError('Human annotations refer to another audio identity')
    records = document.get('records')
    if not isinstance(records, list):
        raise ValueError('Raw annotation records must be retained as a list')
    seen = set()
    kinds = {'BEAT', 'DOWNBEAT', 'PHRASE_BOUNDARY', 'SECTION_BOUNDARY', 'ACCENT', 'UNSURE'}
    for row in records:
        key = row.get('annotation_id')
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('Unique annotation IDs required')
        seen.add(key)
        if row.get('label_type') not in kinds or not row.get('reviewer_id') or not row.get('protocol_version'):
            raise ValueError('Annotation type, reviewer and protocol provenance required')
        t = row.get('time_seconds')
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t) or not 0 <= t < clock['duration_seconds']:
            raise ValueError('Actual in-audio annotation timestamp required')
        reviewed = datetime.fromisoformat(row.get('reviewed_at', '').replace('Z', '+00:00'))
        if reviewed.utcoffset() is None:
            raise ValueError('Human review time must include its time zone')
        if row['label_type'] == 'UNSURE' and not row.get('reason'):
            raise ValueError('UNSURE observations require their reason')
    attached = deepcopy(clock)
    attached['human_annotations'] = dict(status='IMPORTED_UNVERIFIED',
        records=deepcopy(records), import_path=str(Path(path)), import_sha256=sha(path),
        verified=False, raw_record_count=len(records),
        independent_reviewer_and_source_verification='NOT_PERFORMED')
    attached['clock_sha256'] = _digest(attached)
    validate_clock(attached)
    return attached


def _reward(events, evidence, sigma, strengths=None):
    if not len(events) or not len(evidence):
        return dict(score=None, mean_nearest_error_frames=None, per_event_distance_frames=[],
                    accepted_event_count=len(events), evidence_count=len(evidence))
    distances = np.abs(np.asarray(events)[:, None]-np.asarray(evidence)[None])
    nearest = distances.min(axis=1)
    values = np.exp(-distances**2/(2*sigma*sigma))
    if strengths is not None:
        values *= np.asarray(strengths)[None]
    return dict(score=float(values.max(axis=1).mean()),
                mean_nearest_error_frames=float(nearest.mean()),
                per_event_distance_frames=nearest.tolist(), accepted_event_count=len(events),
                evidence_count=len(evidence))


def score(record, clock, shuffle=False):
    """Score retimed source peak proxies, never generated-body quality or truth.

    Shuffle is a frozen half-song circular shift of audio evidence only; it
    keeps source motion, E1-W maps and candidate durations unchanged. It is an
    exposed negative control, not a claim about artistic incompatibility.
    """
    validate_clock(clock)
    if record.get('music_interval') is None:
        raise ValueError('Absolute music interval required for proxy timing')
    A, B = map(int, record['music_interval'])
    if not 0 <= A < B <= clock['frames']:
        raise ValueError('Candidate interval leaves the actual audio clock')
    if record.get('audio_sha256', clock['audio_sha256']) != clock['audio_sha256']:
        raise ValueError('Candidate and MusicClock audio identities differ')
    phases, targets = [], {}
    for name in ('left', 'right'):
        side = record.get(name) or {}
        warp = np.asarray(side.get('time_warp', []), dtype=float)
        target = side.get('target_interval')
        if warp.ndim != 2 or warp.shape[1:] != (2,) or len(warp) < 2:
            continue
        if not np.isfinite(warp).all() or not (np.diff(warp, axis=0) > 0).all():
            raise ValueError('Only actual legal monotone source-to-music maps may be scored')
        if not A <= warp[0, 1] <= warp[-1, 1] < B:
            raise ValueError('Source time map leaves the declared candidate music interval')
        if target is not None:
            if warp[0, 1] != target[0] or warp[-1, 1] != target[1]-1:
                raise ValueError('Time map must match actual half-open Core frame centers')
            targets[name] = [float(target[0]), float(target[1])]
        for phase in side.get('phase_inputs', []):
            if not phase['type'].endswith('speed_peak'):
                continue
            source_frame = float(phase['frame'])
            if warp[0, 0] <= source_frame <= warp[-1, 0]:
                phases.append(dict(side=name, source_frame=source_frame,
                    target_frame=float(np.interp(source_frame, warp[:, 0], warp[:, 1])),
                    body_region=phase.get('body_region', 'UNKNOWN'),
                    type=phase['type'], truth='KINEMATIC_PROXY_NOT_SEMANTIC_ACCENT_TRUTH'))
    peak_frames = [p['target_frame'] for p in phases]
    # A musical junction denotes the transition center, matching the frozen R56
    # allocation contract. Its location changes with asymmetric Core repayment.
    junctions = [(targets['left'][1]+targets['right'][0])/2] if len(targets) == 2 else []
    shift = clock['frames']//2 if shuffle else 0
    def shifted(name):
        x = np.asarray(clock[name], float)
        return np.sort((x+shift) % clock['frames']) if shift else x
    beats, accents = shifted('beat_frames'), shifted('accent_frames')
    phrase, section = shifted('phrase_proxy_frames'), shifted('section_proxy_frames')
    # Accent values travel with their original event under the clock shift.
    original_accents = np.asarray(clock['accent_frames'], float)
    accent_values = np.interp(original_accents, np.arange(clock['frames']), clock['accent_strength'])
    if shift:
        accent_values = accent_values[np.argsort((original_accents+shift) % clock['frames'])]
    sigma = SCORE_PROTOCOL['phase_tolerance_frames']
    b = _reward(peak_frames, beats, sigma)
    a = _reward(peak_frames, accents, sigma, accent_values)
    p = _reward(junctions, phrase, SCORE_PROTOCOL['boundary_tolerance_frames'])
    s = _reward(junctions, section, SCORE_PROTOCOL['boundary_tolerance_frames'])
    beat_accent = .5*b['score']+.5*a['score'] if b['score'] is not None and a['score'] is not None else None
    # The caller applies its frozen section-proxy weight separately, avoiding
    # double counting a boundary reward inside the primary beat/accent score.
    primary = beat_accent
    return dict(phase_beat_score=b['score'], phase_accent_score=a['score'],
        phrase_proxy_score=p['score'], section_proxy_score=s['score'], primary_music_score=primary,
        beat_accent_score=beat_accent, missing_section_proxy_reward=0. if s['score'] is None else None,
        objective_terms=dict(phase_beat=b, phase_accent=a, phrase_boundary_proxy=p, section_boundary_proxy=s),
        phase_peak_observations=phases, junction_frames=junctions,
        music_interval=[A, B], audio_sha256=clock['audio_sha256'], clock_sha256=clock['clock_sha256'],
        music_control='HALF_SONG_AUDIO_CLOCK_SHIFT' if shuffle else 'CORRECT_AUDIO_CLOCK',
        music_shift_frames=shift, motion_and_warp_changed=False,
        score_protocol=deepcopy(SCORE_PROTOCOL), independent_quality_metric=False,
        semantic_alignment_truth='UNKNOWN', human_reviewed=False, can_route=False)
