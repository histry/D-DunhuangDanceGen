"""R66 paired-clock audit using the frozen R47 peaks and R64 reward.

No fitting, detector tuning, window-local differentiation or source replacement.
Body peaks are kinematic proxies; source-clock recovery is not semantic truth.
"""
import importlib
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from dunhuang.io import checked, sha
from dunhuang.contracts import matrices
from dunhuang.research.complete_r36.core import read
from dunhuang.research.paired_event_r41 import events as event_api
from dunhuang.research.music_structure_r56 import music as features_api
from dunhuang.research.complete_r36 import music as legacy_api
from dunhuang.planning import planner as planner_api
from dunhuang.research.timing_r64 import music as reward_api
from dunhuang.research.warp_coupling_r53.common import array_hash


FPS = 30
OFFSETS = (-15, -8, 0, 8, 15)
R47_CONFIG = Path('src/dunhuang/research/manifold_bridge_r47/config.json')
AIST_SEALS = Path('outputs/transfer_audit_01/a2/source_seals.json')
AUDIO_RUNTIME = Path('inputs/clock_audio_runtime_v2/receipt.json')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def load_motion(source, skeleton=None):
    """Return full original common22 rotations and independently checked lineage.

    `skeleton` is deliberately ignored: no FK, Root or contact inference is
    needed to reproduce the R47 local-angular-speed detector.
    """
    raw = checked(source['raw_motion'])
    if sha(raw) != source['raw_motion_sha256']:
        raise ValueError('Raw motion identity differs from R59 source metadata')
    audio = checked(source['audio'])
    if sha(audio) != source['audio_sha256']:
        raise ValueError('Raw audio identity differs from the paired source')
    evidence = dict(source_uid=source['source_uid'], dataset=source['dataset'],
        raw_motion_sha256=source['raw_motion_sha256'], audio_sha256=source['audio_sha256'],
        raw_bytes_verified=True, skeleton_used=False, FK_used=False,
        root_used=False, contact_used=False, raw_origin_preserved=True,
        audio_offset_frames=float(source['audio_offset_frames']),
        clock_scope=source['clock_scope'])
    if source['dataset'] == 'AIST++':
        seals = read(AIST_SEALS)
        candidates = [p for p in seals if p['raw_motion']['sha256'] == source['raw_motion_sha256']]
        if not candidates:
            raise ValueError('Raw AIST motion absent from frozen source seals')
        proof = candidates[0]
        fields = ('raw_motion', 'native', 'source_clock', 'audio', 'role')
        if any(any(p.get(k) != proof.get(k) for k in fields) for p in candidates[1:]):
            raise ValueError('Conflicting raw AIST source seal identities')
        if proof['role'] != source['role']:
            raise ValueError('AIST role differs from sealed recording')
        checked(proof['raw_motion'])
        if proof.get('audio') is not None:
            checked(proof['audio'])
            if proof['audio']['sha256'] != source['audio_sha256']:
                raise ValueError('Optional source-seal audio differs from paired source')
        native_path = checked(proof['native'])
        clock_path = checked(proof['source_clock'])
        if source.get('clock_proof') is not None:
            checked(source['clock_proof'])
            if source['clock_proof']['sha256'] != proof['source_clock']['sha256']:
                raise ValueError('R59 AIST clock proof differs from its source seal')
        clock = read(clock_path)
        calibration = clock['calibration']
        if (calibration.get('verified') is not True
                or calibration.get('origin_anchor_used') is not True
                or not np.isfinite(calibration['maximum_residual_seconds'])
                or not 0 <= calibration['maximum_residual_seconds'] <= 1/FPS
                or clock['raw_motion']['sha256'] != source['raw_motion_sha256']
                or clock['raw_audio']['sha256'] != source['audio_sha256']
                or FPS*calibration['audio_offset_seconds'] != source['audio_offset_frames']):
            raise ValueError('AIST physical-clock calibration/identity not verified')
        native = np.load(native_path, allow_pickle=False)
        if native.ndim != 2 or native.shape[1] != 151 or not np.isfinite(native[:, 7:139]).all():
            raise ValueError('Full sealed Native151 AIST rotations required')
        R = matrices(native[:, 7:139].reshape(-1, 22, 6))
        evidence.update(representation='NATIVE151_COLUMNS_LOCAL_SO3_FIRST22_30HZ',
            source_seals_sha256=sha(AIST_SEALS), native_sha256=proof['native']['sha256'],
            clock_proof_sha256=proof['source_clock']['sha256'],
            calibration=calibration, source_clock_verified=True)
    elif source['dataset'] == 'FineDance':
        native = np.load(raw, allow_pickle=False)
        if native.ndim != 2 or native.shape[1] != 315 or not np.isfinite(native).all():
            raise ValueError('Full raw315D FineDance source required')
        if source['audio_offset_frames'] != 0:
            raise ValueError('R59 FineDance archive-common-start offset changed')
        R = matrices(native[:, 3:135].reshape(-1, 22, 6), row=True)
        evidence.update(representation='FINEDANCE315_ROWS6D_COMMON22_LOCAL_SO3_30HZ',
            source_clock_verified=False,
            source_clock_limitation='OFFICIAL_ARCHIVE_ID_COMMON_START_NOT_AIST_WAVEFORM_CALIBRATION')
    else:
        raise ValueError('Only frozen R59 AIST++ or FineDance paired sources are allowed')
    if len(R) < 4 or not np.isfinite(R).all() or np.max(abs(R.swapaxes(-1, -2)@R-np.eye(3))) >= 2e-4:
        raise ValueError('Finite full-record SO3 body22 observations required')
    prefix_path = Path(source['arrays'])
    if sha(prefix_path) != source['arrays_sha256']:
        raise ValueError('R59 shared-prefix arrays changed')
    with np.load(prefix_path, allow_pickle=False) as z:
        prefix = z['R']
        if len(prefix) > len(R) or not np.array_equal(prefix, R[:len(prefix)]):
            raise ValueError('Full raw motion does not reproduce frozen R59 shared prefix')
        shared_frames = len(prefix)
        prefix_hash = array_hash(prefix)
    if source.get('frames', shared_frames) != shared_frames:
        raise ValueError('R59 recorded shared-prefix frame count differs')
    start = source['shared_start']
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= start < shared_frames:
        raise ValueError('Actual shared-source interval required')
    evidence.update(full_record_frames=len(R), full_R_sha256=array_hash(R),
        shared_source_interval=[int(start), shared_frames],
        R59_shared_prefix_frames=shared_frames, R59_shared_prefix_R_sha256=prefix_hash,
        R59_source_arrays_sha256=source['arrays_sha256'], R59_prefix_exact=True,
        full_record_not_shared_prefix_used_for_peak_extraction=True)
    return R, evidence


def motion_events(R):
    """Exact R47 whole-record profiles+phases; no smoothing or new detector."""
    cfg = read(R47_CONFIG)
    profile = event_api.profiles(R)
    all_phases = event_api.phases(profile, cfg)
    peaks = [p for p in all_phases if p['type'] == 'speed_peak']
    proof = dict(full_R_sha256=array_hash(R), whole_recording=True,
        frame_origin=0, fps=FPS, frames=len(R), events=len(peaks),
        full_phase_count=len(all_phases), profile_hashes={k: array_hash(profile[k])
            for k in ('upper', 'lower', 'trunk', 'global', 'omega')},
        region_joints={k: list(v) for k, v in event_api.REGIONS.items()},
        detector='paired_event_r41.events.profiles + phases; R47_CONFIG',
        detector_code_sha256=sha(event_api.__file__), R47_config_sha256=sha(R47_CONFIG),
        peak_min_distance=cfg['peak_min_distance'],
        speed_prominence='MAX(1.4826*MAD_FULL_RECORD_REGION_SPEED,0.15_RADPS)',
        additional_smoothing=False, window_local_derivatives=False,
        score_scope='UNCALIBRATED_KINEMATIC_PROXY', semantic_truth='UNKNOWN')
    return peaks, proof


def _audio_runtime():
    receipt = read(AUDIO_RUNTIME)
    refs = list(receipt['files'])+list(receipt.get('wheels', []))
    if receipt.get('wheel') is not None:
        refs.append(receipt['wheel'])
    for ref in refs:
        checked(ref)
    root = Path(receipt['import_root']).resolve()
    if not root.is_dir():
        raise ValueError('Frozen audio runtime import root missing')
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    librosa = importlib.import_module('librosa')
    if librosa.__version__ != '0.11.0':
        raise ValueError('Use a fresh process with the sealed librosa0.11.0 runtime')
    try:
        Path(librosa.__file__).resolve().relative_to(root)
    except ValueError:
        raise ValueError('librosa was not loaded from the sealed audio import root')
    return dict(receipt_sha256=sha(AUDIO_RUNTIME), librosa_version=librosa.__version__,
        librosa_module_sha256=sha(librosa.__file__), runtime_files_verified=len(refs))


def audio_clock(path):
    """Compute the unchanged R64 beat/accent clock on this full real waveform."""
    runtime = _audio_runtime()
    path = Path(path)
    wave, sr = sf.read(path, always_2d=True, dtype='float64')
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Finite full original waveform required')
    frames = (len(wave)*FPS+sr-1)//sr
    seed = planner_api.timeline(path, frames)
    structure = legacy_api.analyze(path, frames, seed['beats'])
    arrays, metadata = features_api.analyze(path)
    track = dict(path=str(path), audio_sha256=sha(path), structure=structure,
                 structure_sha256=_digest(structure))
    clock = reward_api.clock_from_features(0, track, arrays, metadata)
    # R36 uses this exact node count to select its detector-vs-grid branch.
    raw_nodes = sorted(set([0]+[int(b) for b in seed['beats'] if 0 < b < frames]+[frames]))
    fallback = len(raw_nodes) < 12
    mono = wave.mean(1)
    complete = len(wave)*FPS//sr
    energy = np.asarray([np.sqrt(np.mean(mono[round(t*sr/FPS):round((t+1)*sr/FPS)]**2))
                         for t in range(complete)], float)
    if not np.isfinite(energy).all():
        raise ValueError('Every raw frame RMS must have genuine waveform support')
    beats = np.asarray(clock['beat_frames'], float)
    tempo = float(60*FPS/np.median(np.diff(beats))) if len(beats) >= 2 else None
    clock.update(energy_curve=energy.tolist(), energy_valid_interval=[0, int(complete)],
        energy_curve_sha256=array_hash(energy),
        energy_scope='RAW_MONO_WAVEFORM_COMPLETE30HZ_CELLS_ROUND_ENDPOINTS_RMS_NO_RANK_NO_PADDING',
        rms=arrays['rms'].tolist(),
        rms_scope='FROZEN_R56_STFT_RMS_SEPARATE_FROM_RAW_FRAME_ENERGY',
        tempo_bpm=tempo, tempo_scope='MEDIAN_RETAINED_BEAT_INTERVAL_DETECTOR_OR_R36_GRID_PROXY',
        R36_grid_fallback_used=fallback,
        detected_beat_frames_before_R36_branch=list(seed['beats']),
        legacy_interior_detector_or_grid_fallback_status='R36_15FRAME_GRID_FALLBACK' if fallback else 'LIBROSA_DETECTED_BEATS_NO_R36_GRID_FALLBACK',
        exposure='PAIRED_R59_HISTORICALLY_EXPOSED_AIST_OR_FINEDANCE_DIAGNOSTIC',
        R66_audio_receipt=dict(audio_runtime=runtime,
            planner_timeline_sha256=sha(planner_api.__file__),
            R36_music_sha256=sha(legacy_api.__file__), R56_music_sha256=sha(features_api.__file__),
            R56_config_sha256=sha(Path(features_api.__file__).with_name('protocol.json')),
            R64_music_sha256=sha(reward_api.__file__),
            detector_defaults_changed=False, new_detector=False, history_track_cache_reads=0,
            raw_detected_beat_count=len(seed['beats']), legacy_node_count=len(raw_nodes),
            legacy_structure_canonical_sha256=track['structure_sha256'],
            source_sample_rate=sr, source_samples=len(wave)))
    clock['clock_sha256'] = reward_api._digest(clock)
    reward_api.validate_clock(clock)
    return clock


def score_offsets(events, clock, audio_offset_frames, source_interval, offsets=OFFSETS):
    """One source peak population for all five physical-clock hypotheses."""
    reward_api.validate_clock(clock)
    shifts = tuple(offsets)
    if shifts != OFFSETS:
        raise ValueError('Only the frozen R59 five-offset population is allowed')
    if not np.isfinite(audio_offset_frames):
        raise ValueError('Finite actual source-audio offset required')
    a, b = source_interval
    if not (np.isfinite([a, b]).all() and 0 <= a < b):
        raise ValueError('Actual nonempty source interval required')
    kept, counts = [], dict(non_speed_peak=0, source_evidence_outside=0, audio_common_support_missing=0)
    for p in events:
        if not p['type'].endswith('speed_peak'):
            counts['non_speed_peak'] += 1
            continue
        lo, hi = p['evidence_span']
        if not a <= p['frame'] < b or lo < a or hi > b:
            counts['source_evidence_outside'] += 1
            continue
        t = float(p['frame'])+float(audio_offset_frames)
        if t+min(shifts) < 0 or t+max(shifts) > clock['frames']-1:
            counts['audio_common_support_missing'] += 1
            continue
        kept.append(dict(p))
    peaks = np.asarray([p['frame']+audio_offset_frames for p in kept], float)
    accents = np.asarray(clock['accent_frames'], float)
    strengths = np.interp(accents, np.arange(clock['frames']), clock['accent_strength'])
    sigma = reward_api.SCORE_PROTOCOL['phase_tolerance_frames']
    if sigma != 3 or reward_api.SCORE_PROTOCOL['primary_weights'] != dict(phase_beat_score=.5, phase_accent_score=.5):
        raise ValueError('Frozen R64 reward protocol changed')
    scores, detail = dict(Beat=[], Accent=[], Primary=[]), []
    for shift in shifts:
        times = peaks+shift
        beat = reward_api._reward(times, clock['beat_frames'], sigma)
        accent = reward_api._reward(times, accents, sigma, strengths)
        primary = .5*beat['score']+.5*accent['score'] if beat['score'] is not None and accent['score'] is not None else None
        scores['Beat'].append(beat['score']); scores['Accent'].append(accent['score']); scores['Primary'].append(primary)
        detail.append(dict(offset_frames=shift, Beat=beat, Accent=accent))
    return dict(events=kept, scores=scores, evidence=dict(offsets=list(shifts),
        source_interval=list(source_interval), audio_offset_frames=float(audio_offset_frames),
        input_events=len(events), retained_events=len(kept), rejected=counts,
        same_event_set_for_all_offsets=True, whole_record_peaks_filtered_not_redetected=True,
        clock_sha256=clock['clock_sha256'], audio_sha256=clock['audio_sha256'],
        reward_code_sha256=sha(reward_api.__file__), score_protocol=reward_api.SCORE_PROTOCOL,
        per_shift=detail, section_proxy_score=None,
        section_scope='NOT_EVALUATED_NO_TRUE_CHOREOGRAPHY_JUNCTION_IN_FIXED_SOURCE_CLOCK_QUERY',
        downbeat_truth='UNKNOWN', phase_semantic_truth='UNKNOWN',
        evaluation_scope='SOURCE_CLOCK_OFFSET_DIAGNOSTIC_NOT_ARTISTIC_OR_EVENT_SEMANTIC_TRUTH',
        human_reviewed=False, training_updates=0))


def synthetic_control():
    """Nonperiodic exact clock fixture; no real waveform or runtime is read."""
    n, positions = 150, [24, 61, 105]
    accent = np.zeros(n)
    accent[positions] = 1.
    arrays = dict(accent_curve=accent, held_boundary=np.zeros(n), proposal=np.zeros(n),
                  proxy_accents=np.asarray(positions, np.int32),
                  proxy_boundaries=np.asarray([], np.int32))
    structure = dict(frames=n, beats=[0]+positions+[n])
    metadata = dict(frames=n, audio_samples=15000, sample_rate=3000,
        audio_sha256='0'*64, feature_version='TEST_FIXTURE_R66_EXACT_CLOCK_NOT_REAL_AUDIO',
        arrays={k: array_hash(x) for k, x in arrays.items()})
    track = dict(path='TEST_FIXTURE_NO_REAL_AUDIO_FILE', audio_sha256='0'*64,
                 structure=structure, structure_sha256=_digest(structure))
    clock = reward_api.clock_from_features(0, track, arrays, metadata)
    events = [dict(frame=t, type='speed_peak', body_region='upper',
                   evidence_span=[t-1, t+2], semantic_truth='SYNTHETIC_CODE_FIXTURE_ONLY')
              for t in positions]
    result = score_offsets(events, clock, 0., [15, 135])
    if result['scores']['Primary'][2] != 1. or np.argmax(result['scores']['Primary']) != 2:
        raise ValueError('Exact nonperiodic synthetic clock failed')
    return dict(status='PASS', scores=result['scores'],
        offset_sign='POSITIVE_DELTA_READS_LATER_RAW_AUDIO; AUDIO_FRAME=MOTION_FRAME+SOURCE_OFFSET+DELTA',
        scope='SYNTHETIC_CODE_SIGN_POSITIVE_CONTROL_NOT_REAL_EVENT_TRUTH',
        fixture='TEST_FIXTURE_R66_EXACT_CLOCK_NOT_REAL_AUDIO',
        clock_validated=True, peak_frames=positions, sigma_frames=3,
        real_audio_reads=0, training_steps=0)
