import copy
import json

import numpy as np
import pytest

from dunhuang.research.timing_r64.music import (
    FPS, SCORE_PROTOCOL, build, clock_from_features, import_human_labels, score,
    validate_clock)
from dunhuang.research.warp_coupling_r53.common import array_hash


def prepared(frames=600, beats=None):
    beats = list(range(0, frames, 30))+[frames] if beats is None else beats
    curve = np.zeros(frames)
    curve[[30, 300]] = 1.
    proposal = np.zeros(frames)
    proposal[264] = 1.
    arrays = dict(accent_curve=curve, held_boundary=proposal.copy(), proposal=proposal,
                  proxy_accents=np.array([30, 300]), proxy_boundaries=np.array([264]))
    track = dict(path='real.wav', audio_sha256='a'*64, structure_sha256='b'*64,
                 structure=dict(frames=frames, beats=beats))
    metadata = dict(frames=frames, audio_samples=frames*1470, sample_rate=44100,
                    audio_sha256='a'*64, feature_version='FROZEN_R56_SYNTHETIC_TEST_FIXTURE',
                    arrays={k: array_hash(v) for k, v in arrays.items()})
    return track, arrays, metadata


def clock(beats=None):
    return clock_from_features(0, *prepared(beats=beats))


def record():
    return dict(music_interval=[0, 528], audio_sha256='a'*64,
        left=dict(target_interval=[0, 240], time_warp=[[0, 0], [239, 239]],
                  phase_inputs=[dict(type='upper_speed_peak', frame=30, body_region='upper')]),
        right=dict(target_interval=[288, 528], time_warp=[[0, 288], [239, 527]],
                   phase_inputs=[dict(type='trunk_speed_peak', frame=12, body_region='trunk')]))


def test_clock_uses_actual_samples_seconds_and_audio_only_evidence():
    c = clock()
    assert validate_clock(c) is True
    assert c['sample_rate'] == 44100 and c['duration_seconds'] == 20
    assert c['frame_times_seconds'][1] == 1/FPS
    assert np.array_equal(np.asarray(c['beat_frames'])/FPS, c['beat_seconds'])
    assert c['beat_frames'][0] == 30 and c['beat_frames'][-1] == 570
    assert c['raw_legacy_beat_frames'][0] == 0 and c['raw_legacy_beat_frames'][-1] == 600
    assert c['excluded_legacy_boundary_sentinels'] == [0, 600]
    assert set(c['tempo_curve_bpm']) == {60.}
    assert c['downbeat_frames'] is None and c['independent_phrase_labels'] is None
    assert c['motion_used_to_extract_music'] is False and c['human_reviewed'] is False
    assert 'NOT_ANNOTATED_PHRASES' in c['phrase_proxy_scope']
    json.dumps(c, allow_nan=False)


def test_partial_final_audio_frame_preserves_sample_precision():
    track, arrays, meta = prepared()
    meta['audio_samples'] -= 1
    c = clock_from_features(0, track, arrays, meta)
    assert c['frames'] == 600
    assert c['duration_seconds'] == meta['audio_samples']/44100
    assert c['frame_times_seconds'][-1] < c['duration_seconds']


def test_beat_order_and_extraction_receipt_are_checked():
    track, arrays, metadata = prepared()
    track['structure']['beats'] = [30, 0]
    with pytest.raises(ValueError, match='sorted'):
        clock_from_features(0, track, arrays, metadata)
    track, arrays, metadata = prepared()
    arrays['accent_curve'][0] = .1
    with pytest.raises(ValueError, match='receipt'):
        clock_from_features(0, track, arrays, metadata)


def test_missing_beats_do_not_create_a_tempo_or_perfect_music_score():
    c = clock(beats=[])
    assert c['tempo_curve_bpm'] == [None]*600
    s = score(record(), c)
    assert s['phase_beat_score'] is None and s['primary_music_score'] is None
    assert s['objective_terms']['phase_beat']['evidence_count'] == 0


def test_score_uses_retimed_phase_peaks_and_transition_center():
    c, r = clock(), record()
    s = score(r, c)
    assert [p['target_frame'] for p in s['phase_peak_observations']] == [30., 300.]
    assert s['junction_frames'] == [264.]
    assert s['phase_beat_score'] == pytest.approx(1.)
    assert s['phase_accent_score'] == pytest.approx(1.)
    assert s['section_proxy_score'] == pytest.approx(1.)
    assert s['primary_music_score'] == pytest.approx(1.)
    assert s['score_protocol']['boundary_rewards_in_primary'] is False
    assert 'PHRASE_DOWNBEAT_ZERO' in s['score_protocol']['boundary_reward_scope']
    assert s['independent_quality_metric'] is False


def test_shuffle_changes_reward_only_and_does_not_mutate_actual_maps_or_clock():
    c, r = clock(), record()
    c0, r0 = copy.deepcopy(c), copy.deepcopy(r)
    correct, shuffled = score(r, c), score(r, c, shuffle=True)
    assert r == r0 and c == c0
    assert shuffled['music_shift_frames'] == 300
    assert shuffled['motion_and_warp_changed'] is False
    assert shuffled['phase_accent_score'] < correct['phase_accent_score']
    assert shuffled['phase_peak_observations'] == correct['phase_peak_observations']
    assert shuffled['clock_sha256'] == correct['clock_sha256']


def test_unknown_or_illegal_phase_maps_not_promoted_to_true_alignment():
    c, r = clock(), record()
    r['left']['phase_inputs'] = []
    r['right']['phase_inputs'] = []
    s = score(r, c)
    assert s['primary_music_score'] is None and s['phase_peak_observations'] == []
    assert s['semantic_alignment_truth'] == 'UNKNOWN'
    r['left']['time_warp'][1][1] = -1
    with pytest.raises(ValueError, match='monotone'):
        score(r, c)


def test_score_rejects_wrong_audio_or_target_clock():
    c, r = clock(), record()
    r['audio_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='identities'):
        score(r, c)
    r = record()
    r['left']['time_warp'][0][1] = 1
    with pytest.raises(ValueError, match='frame centers'):
        score(r, c)


def test_imported_human_records_remain_unverified_and_unsure_is_retained(tmp_path):
    c = clock()
    document = dict(audio_sha256=c['audio_sha256'], human_reviewed=True,
        records=[dict(annotation_id='one', label_type='UNSURE', reviewer_id='r1',
            protocol_version='v1', time_seconds=2., reviewed_at='2026-11-19T08:00:00+08:00',
            reason='BOUNDARY_AMBIGUOUS')])
    path = tmp_path/'human.json'
    path.write_text(json.dumps(document), encoding='utf8')
    attached = import_human_labels(c, path)
    assert attached['human_annotations']['status'] == 'IMPORTED_UNVERIFIED'
    assert attached['human_annotations']['records'][0]['label_type'] == 'UNSURE'
    assert attached['human_reviewed'] is False and attached['operational_qualified'] is False
    assert attached['downbeat_frames'] is None and attached['independent_phrase_labels'] is None
    assert attached['clock_sha256'] != c['clock_sha256']
    assert c['human_annotations']['status'] == 'UNAVAILABLE'


def test_import_requires_audio_identity_and_timezone(tmp_path):
    c = clock()
    path = tmp_path/'human.json'
    document = dict(audio_sha256='c'*64, records=[])
    path.write_text(json.dumps(document), encoding='utf8')
    with pytest.raises(ValueError, match='audio identity'):
        import_human_labels(c, path)
    document = dict(audio_sha256=c['audio_sha256'], records=[dict(annotation_id='one',
        label_type='BEAT', reviewer_id='r1', protocol_version='v1', time_seconds=1.,
        reviewed_at='2026-11-19T08:00:00')])
    path.write_text(json.dumps(document), encoding='utf8')
    with pytest.raises(ValueError, match='time zone'):
        import_human_labels(c, path)


def test_tampered_clock_or_invented_qualification_rejected():
    c = clock()
    c['accent_strength'][0] = .1
    with pytest.raises(ValueError, match='hash'):
        validate_clock(c)
    c = clock()
    c['human_reviewed'] = True
    with pytest.raises(ValueError, match='qualification'):
        validate_clock(c)


def test_build_reads_real_sample_rate_and_source_hash(tmp_path):
    import soundfile as sf
    from dunhuang.research.complete_r36.core import sha
    path = tmp_path/'real.wav'
    sample_rate = 44100
    t = np.arange(sample_rate*3+1)/sample_rate
    sf.write(path, .05*np.sin(2*np.pi*440*t), sample_rate)
    track = dict(path=str(path), audio_sha256=sha(path), structure_sha256='b'*64,
                 structure=dict(frames=91, beats=[0, 30, 60, 90, 91]))
    clocks = build([track])
    assert clocks[0]['audio_samples'] == sample_rate*3+1
    assert clocks[0]['sample_rate'] == sample_rate and clocks[0]['frames'] == 91
    assert clocks[0]['feature_version'] == 'SCIPY_STFT_LOG_FREQUENCY48_CHROMA12_V1'


@pytest.mark.parametrize('frames', [3601, 3598])
def test_real_legacy_end_fencepost_is_preserved_but_not_a_beat(frames):
    # The failure had sorted real R36 clocks ending exactly at frames. Only the
    # start/end injected fences may be dropped; detector candidates are kept.
    track, arrays, metadata = prepared()
    track['structure']['frames'] = frames
    track['structure']['beats'] = [0, 30, 75, frames-2, frames]
    arrays = {k: np.resize(v, frames) if k not in ('proxy_accents', 'proxy_boundaries') else v
              for k, v in arrays.items()}
    metadata.update(frames=frames, audio_samples=frames*1470,
                    arrays={k: array_hash(v) for k, v in arrays.items()})
    original = copy.deepcopy(track)
    c = clock_from_features(0, track, arrays, metadata)
    assert track == original
    assert c['beat_frames'] == [30., 75., frames-2.]
    assert c['raw_legacy_beat_frames'] == original['structure']['beats']
    assert c['legacy_boundary_sentinels'] == [0, frames]
    assert c['legacy_raw_beat_count'] == 5 and c['interior_legacy_beat_count'] == 3
    assert c['tempo_observation_bpm'][0] == pytest.approx(60*30/45)


@pytest.mark.parametrize('bad', [-1, 600.5, 601])
def test_arbitrary_out_of_range_legacy_beat_is_not_silently_removed(bad):
    track, arrays, metadata = prepared()
    track['structure']['beats'] = sorted([0, 30, 300, 600, bad])
    with pytest.raises(ValueError, match='actual audio frame centers'):
        clock_from_features(0, track, arrays, metadata)
