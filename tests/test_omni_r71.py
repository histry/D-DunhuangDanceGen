"""Admission and schema controls, no fabricated model or human judgments."""
import json
import pytest
from dunhuang.research.omni_judge_r71.prepare import METHODS, register
from dunhuang.research.omni_judge_r71.judge import DIMS, parse, save, video_clocks
from dunhuang.research.omni_judge_r71.summarize import normalize
from dunhuang.research.omni_judge_r71.controls import wrong_music


def test_wrong_music_uses_recording_identity_not_nearby_same_song_clip():
    files={'near':dict(path='near.wav',raw_audio_sha256='same'),
           'other':dict(path='other.wav',raw_audio_sha256='different')}
    assert wrong_music('same',files)['raw_audio_sha256']=='different'
    with pytest.raises(ValueError):
        wrong_music('same',{'near':files['near']})


def test_order_swap_is_decoded_before_automatic_consistency_counts():
    assert normalize('A', 'AB') == normalize('B', 'BA') == 'baseline'
    assert normalize('B', 'AB') == normalize('A', 'BA') == 'candidate'
    assert normalize('TIE', 'AB') == normalize('TIE', 'BA') == 'TIE'
    assert normalize('UNSURE', 'AB') == 'UNSURE'


def test_registered_population_keeps_missing_and_failed_G1():
    cases = [dict(key=str(i)) for i in range(12)]
    rows = [dict(case_key=c['key'], method=m, status='LEGAL', connection=dict(passes_original_screen=True))
            for c in cases for m in ['E1W'] + METHODS]
    for r in rows:
        if r['method'] == 'BeatDP' and r['case_key'] in ['0', '1', '2']:
            r.update(status='NO_EVENT', connection=None)
        if r['method'] == 'RhythmDTW' and r['case_key'] == '3':
            r['connection']['passes_original_screen'] = False
    result = register(cases, rows)
    assert len(result) == 60 and sum(r['eligible'] for r in result) == 56
    assert len({r['pair_id'] for r in result}) == 60
    assert sum(r['pilot'] for r in result) == 3
    assert all(r['exclusion'] for r in result if not r['eligible'])


def valid_response():
    return {d: dict(preference='UNSURE', timestamps_seconds=[], reason='Missing evidence') for d in DIMS}


def test_uncertainty_and_ties_survive_without_winner_imputation():
    obj = valid_response()
    obj[DIMS[1]]['preference'] = 'TIE'
    assert parse(json.dumps(obj)) == obj


@pytest.mark.parametrize('bad', [True, -0.1, 17.6, float('nan'), '1'])
def test_invalid_timestamps_rejected(bad):
    obj = valid_response()
    obj[DIMS[0]]['timestamps_seconds'] = [bad]
    assert parse(json.dumps(obj)) is None


def test_incomplete_response_never_filled_with_fabricated_scores():
    obj = valid_response()
    del obj[DIMS[0]]
    assert parse(json.dumps(obj)) is None
    assert parse('no JSON') is None


def test_unicode_response_persistence_is_atomic_utf8(tmp_path):
    p=tmp_path/'result.json';obj=dict(raw_response='相当，无法判断。',human_reviewed=False)
    save(p,obj)
    assert json.loads(p.read_bytes().decode('utf-8'))==obj
    assert not p.with_name('result.json.tmp').exists()


def test_real_sample_rate_covers_full_video_not_requested_rate():
    video=[None]*16
    meta=dict(frames_indices=[round(i*527/15) for i in range(16)],total_num_frames=528,fps=30)
    videos,metadata,rates=video_clocks([(video,meta)])
    assert rates==[16/17.6]
    assert abs(8*(2/rates[0])-17.6)<1e-9
    assert metadata[0]['frame_times_seconds'][-1]==527/30


def test_model_reported_intervals_are_validated_without_inventing_point_times():
    obj=valid_response();obj[DIMS[0]]['timestamps_seconds']=['0.00-1.00','17.0-17.6']
    assert parse(json.dumps(obj))==obj
    obj[DIMS[0]]['timestamps_seconds']=['17.0-18.0']
    assert parse(json.dumps(obj)) is None
