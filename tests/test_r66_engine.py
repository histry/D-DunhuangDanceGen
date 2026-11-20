"""TEST_FIXTURE engine contracts; no research data, network, fitting or labels."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from scipy.spatial.transform import Rotation

from dunhuang.contracts import sixd
from dunhuang.research.timing_validity_r66 import engine as e


def ref(path):
    return dict(path=str(path), sha256=e.sha(path))


@pytest.fixture
def motion(tmp_path):
    n = 300
    t = np.arange(n)
    v = np.zeros((n, 22, 3))
    v[:, 12:22, 0] = (.8*np.tanh((t-60)/5)+.7*np.tanh((t-140)/8))[:, None]
    v[:, [1, 2, 4, 5, 7, 8, 10, 11], 1] = (.5*np.tanh((t-190)/5))[:, None]
    R = Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3).astype(np.float32)
    raw = tmp_path/'TEST_FIXTURE_motion.npy'
    x = np.zeros((n, 315), np.float32)
    x[:, 3:135] = sixd(R, row=True).reshape(n, 132)
    np.save(raw, x)
    full = e.matrices(x[:, 3:135].reshape(-1, 22, 6), row=True)
    prefix = tmp_path/'TEST_FIXTURE_prefix.npz'
    np.savez(prefix, R=full[:220])
    audio = tmp_path/'TEST_FIXTURE_audio_bytes.wav'
    audio.write_bytes(b'TEST_FIXTURE_HASHED_SOURCE_ONLY_NOT_AUDIO_EXTRACTION')
    source = dict(source_uid='TEST_FIXTURE_FINE', dataset='FineDance',
        raw_motion=ref(raw), raw_motion_sha256=e.sha(raw), audio=ref(audio),
        audio_sha256=e.sha(audio), audio_offset_frames=0.,
        clock_scope='TEST_FIXTURE_ARCHIVE_COMMON_START', arrays=str(prefix),
        arrays_sha256=e.sha(prefix), shared_start=0, frames=220, role='genre_train_diagnostic')
    return source, full, R


def make_clock():
    n, positions = 150, [24, 61, 105]
    accent = np.zeros(n); accent[positions] = 1.
    arrays = dict(accent_curve=accent, held_boundary=np.zeros(n), proposal=np.zeros(n),
                  proxy_accents=np.asarray(positions, np.int32), proxy_boundaries=np.asarray([], np.int32))
    meta = dict(frames=n, audio_samples=15000, sample_rate=3000, audio_sha256='0'*64,
        feature_version='TEST_FIXTURE_R66_CLOCK', arrays={k: e.array_hash(v) for k, v in arrays.items()})
    structure = dict(frames=n, beats=[0]+positions+[n])
    track = dict(path='TEST_FIXTURE_NO_AUDIO_FILE', audio_sha256='0'*64,
                 structure=structure, structure_sha256=e._digest(structure))
    return e.reward_api.clock_from_features(0, track, arrays, meta)


def test_finedance_full315_row_decode_precedes_shared_prefix(motion):
    source, expected, _ = motion
    full, evidence = e.load_motion(source, skeleton='TEST_FIXTURE_IGNORED')
    assert np.array_equal(full, expected) and len(full) == 300
    assert evidence['shared_source_interval'] == [0, 220]
    assert evidence['full_record_not_shared_prefix_used_for_peak_extraction']
    assert evidence['R59_prefix_exact'] and not evidence['source_clock_verified']
    assert not evidence['root_used'] and not evidence['skeleton_used']


def test_full_source_rejects_prefix_or_raw_identity_corruption(motion):
    source, expected, _ = motion
    prefix = expected[:220].copy(); prefix[20, 2, 0, 0] += .1
    np.savez(source['arrays'], R=prefix)
    source['arrays_sha256'] = e.sha(source['arrays'])
    with pytest.raises(ValueError, match='shared prefix'):
        e.load_motion(source)
    source['raw_motion_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        e.load_motion(source)


def test_aist_full_native_clock_seal_without_optional_audio(motion, tmp_path, monkeypatch):
    source, _, R = motion
    raw = tmp_path/'TEST_FIXTURE_AIST_raw.bin'; raw.write_bytes(b'TEST_FIXTURE_NOT_REAL_AIST')
    native = np.zeros((len(R), 151), np.float32)
    native[:, 7:139] = sixd(R).reshape(len(R), 132)
    native[:, 139:] = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), 2)
    path = tmp_path/'TEST_FIXTURE_native.npy'; np.save(path, native)
    expected = e.matrices(native[:, 7:139].reshape(-1, 22, 6))
    np.savez(source['arrays'], R=expected[:220])
    clock = dict(calibration=dict(verified=True, origin_anchor_used=True,
                 maximum_residual_seconds=.005, audio_offset_seconds=.0125),
                 raw_motion=dict(sha256=e.sha(raw)), raw_audio=dict(sha256=source['audio_sha256']))
    clock_path = tmp_path/'TEST_FIXTURE_clock.json'; clock_path.write_text(json.dumps(clock))
    seal = dict(raw_motion=ref(raw), native=ref(path), source_clock=ref(clock_path), role='train')
    seals = tmp_path/'TEST_FIXTURE_seals.json'; seals.write_text(json.dumps([seal]))
    monkeypatch.setattr(e, 'AIST_SEALS', seals)
    source.update(dataset='AIST++', role='train', raw_motion=ref(raw), raw_motion_sha256=e.sha(raw),
                  arrays_sha256=e.sha(source['arrays']), audio_offset_frames=.375, clock_proof=ref(clock_path))
    full, proof = e.load_motion(source)
    assert np.array_equal(full, expected) and proof['source_clock_verified']
    clock['calibration']['verified'] = False; clock_path.write_text(json.dumps(clock))
    seal['source_clock'] = ref(clock_path); seals.write_text(json.dumps([seal])); source['clock_proof'] = ref(clock_path)
    with pytest.raises(ValueError, match='calibration'):
        e.load_motion(source)


def test_motion_peaks_are_exact_frozen_whole_record_R47(motion, monkeypatch):
    _, full, _ = motion
    path = Path(e.event_api.__file__).parents[1]/'manifold_bridge_r47'/'config.json'
    monkeypatch.setattr(e, 'R47_CONFIG', path)
    events, proof = e.motion_events(full)
    expected = [p for p in e.event_api.phases(e.event_api.profiles(full), e.read(path)) if p['type'] == 'speed_peak']
    assert events == expected and len(events) >= 2
    assert proof['frames'] == 300 and proof['whole_recording']
    assert not proof['additional_smoothing'] and not proof['window_local_derivatives']
    assert proof['peak_min_distance'] == 12


def test_synthetic_positive_control_uses_a_valid_frozen_R64_clock():
    result = e.synthetic_control()
    assert result['status'] == 'PASS' and result['scores']['Primary'][2] == 1.
    assert np.argmax(result['scores']['Primary']) == 2
    assert result['real_audio_reads'] == result['training_steps'] == 0


def test_five_offsets_equal_R64_score_and_keep_fractional_audio_origin():
    clock = make_clock()
    events = [dict(frame=t, type='speed_peak', body_region='upper', evidence_span=[t-1, t+2])
              for t in (24, 61, 105)]
    result = e.score_offsets(events, clock, .375, [17, 130])
    for i, shift in enumerate(e.OFFSETS):
        record = dict(music_interval=[0, 150], left=dict(
            time_warp=[[17, 17+.375+shift], [129, 129+.375+shift]], phase_inputs=events), right={})
        expected = e.reward_api.score(record, clock)
        assert result['scores']['Primary'][i] == pytest.approx(expected['primary_music_score'], abs=1e-12)
        assert result['scores']['Beat'][i] == pytest.approx(expected['phase_beat_score'], abs=1e-12)
        assert result['scores']['Accent'][i] == pytest.approx(expected['phase_accent_score'], abs=1e-12)
    assert result['evidence']['audio_offset_frames'] == .375
    assert result['evidence']['section_proxy_score'] is None


def test_phase_set_is_common_support_and_missing_is_not_zero():
    clock = make_clock()
    events = [dict(frame=t, type='speed_peak', body_region='upper', evidence_span=[t-1, t+2])
              for t in (2, 24, 61, 145)]
    result = e.score_offsets(events, clock, 0, [0, 150])
    assert [p['frame'] for p in result['events']] == [24, 61]
    assert result['evidence']['rejected']['audio_common_support_missing'] == 2
    events[1]['evidence_span'] = [0, 40]
    trimmed = e.score_offsets(events, clock, 0, [15, 135])
    assert [p['frame'] for p in trimmed['events']] == [61]
    empty = e.score_offsets([], clock, 0, [15, 135])
    assert empty['scores'] == dict(Beat=[None]*5, Accent=[None]*5, Primary=[None]*5)
    with pytest.raises(ValueError, match='five-offset'):
        e.score_offsets(events, clock, 0, [15, 135], [0])


def test_audio_clock_composes_frozen_APIs_and_keeps_floor_raw_RMS(tmp_path, monkeypatch):
    path = tmp_path/'TEST_FIXTURE_audio.wav'
    sr, count = 3000, 15007
    wave = .1*np.sin(np.arange(count)*.1)
    sf.write(path, wave, sr, subtype='FLOAT')
    n = (count*30+sr-1)//sr
    calls = []
    monkeypatch.setattr(e, '_audio_runtime', lambda: dict(librosa_version='0.11.0', fixture='TEST_FIXTURE'))
    def timeline(p, frames):
        assert p == path and frames == n
        calls.append('timeline'); return dict(beats=[24, 61, 105])
    def structure(p, frames, beats):
        calls.append('R36'); return dict(frames=frames, beats=[0]+beats+[frames])
    def analyze(p):
        calls.append('R56')
        accent = np.zeros(n); accent[[24, 61, 105]] = 1
        arrays = dict(accent_curve=accent, held_boundary=np.zeros(n), proposal=np.zeros(n),
                      proxy_accents=np.asarray([24, 61, 105], np.int32), proxy_boundaries=np.asarray([], np.int32), rms=np.ones(n))
        return arrays, dict(frames=n, audio_samples=count, sample_rate=sr,
            audio_sha256=e.sha(path), feature_version='TEST_FIXTURE_NOT_REAL_FEATURE_EXTRACTION',
            arrays={k: e.array_hash(v) for k, v in arrays.items()})
    monkeypatch.setattr(e.planner_api, 'timeline', timeline)
    monkeypatch.setattr(e.legacy_api, 'analyze', structure)
    monkeypatch.setattr(e.features_api, 'analyze', analyze)
    clock = e.audio_clock(path)
    assert calls == ['timeline', 'R36', 'R56']
    assert clock['frames'] == 151 and len(clock['energy_curve']) == 150
    actual, _ = sf.read(path, dtype='float64')
    expected = [float(np.sqrt(np.mean(actual[round(t*sr/30):round((t+1)*sr/30)]**2))) for t in range(150)]
    assert clock['energy_curve'] == expected and clock['energy_valid_interval'] == [0, 150]
    assert clock['tempo_bpm'] == 1800/40.5
    assert clock['R66_audio_receipt']['history_track_cache_reads'] == 0
    assert clock['downbeat_frames'] is None and clock['independent_phrase_labels'] is None
    assert e.reward_api.validate_clock(clock)


def test_audio_runtime_requires_verified_root_and_exact_version(tmp_path, monkeypatch):
    root = tmp_path/'TEST_FIXTURE_vendor'; root.mkdir()
    module = root/'TEST_FIXTURE_init.py'; module.write_text('TEST_FIXTURE = True\n')
    receipt = tmp_path/'TEST_FIXTURE_runtime.json'
    receipt.write_text(json.dumps(dict(files=[ref(module)], import_root=str(root))))
    monkeypatch.setattr(e, 'AUDIO_RUNTIME', receipt)
    monkeypatch.setattr(e.sys, 'path', list(e.sys.path))
    fake = SimpleNamespace(__version__='0.11.0', __file__=str(module))
    monkeypatch.setattr(e.importlib, 'import_module', lambda name: fake)
    assert e._audio_runtime()['librosa_version'] == '0.11.0'
    assert e.sys.path[0] == str(root.resolve())
    fake.__version__ = '0.10.0'
    with pytest.raises(ValueError, match='0.11.0'):
        e._audio_runtime()
