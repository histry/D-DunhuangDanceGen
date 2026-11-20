"""Synthetic metadata only; no actual waveforms, motions or human records."""
from copy import deepcopy
import hashlib
import json

import pytest

from dunhuang.research.timing_validity_r66.population import OFFSETS, digest, select


def source(uid, audio, role='train', dataset='AIST++', offset=12.):
    return dict(source_uid=uid, dataset=dataset, role=role, group=dataset+':'+audio,
        recording_uid=uid, raw_motion_sha256=hashlib.sha256(uid.encode()).hexdigest(),
        audio_sha256=hashlib.sha256(audio.encode()).hexdigest(),
        audio=dict(path='TEST_FIXTURE/'+audio+'.wav', sha256=hashlib.sha256(audio.encode()).hexdigest()),
        audio_offset_frames=offset, frames=500, shared_start=0,
        shared_support=dict(frames=500), clock_scope='TEST_FIXTURE_REGISTERED_CLOCK')


def row(s, uid, interval):
    a, b = interval
    return dict(uid=uid, source_uid=s['source_uid'], dataset=s['dataset'], role=s['role'],
        group=s['group'], recording_uid=s['recording_uid'], raw_motion_sha256=s['raw_motion_sha256'],
        audio_sha256=s['audio_sha256'], source_interval=interval,
        audio_interval=[a+s['audio_offset_frames'], b+s['audio_offset_frames']], frames=150,
        music_activity=.4, body_activity=.6)


@pytest.fixture
def fixture():
    base = source('base', 'audio0')
    sources = [base, source('same_audio_alias', 'audio0')]
    sources += [source('donor'+str(i), 'audio'+str(i)) for i in range(1, 5)]
    sources += [source('val', 'val_audio', role='val', offset=0.),
                source('fine', 'fine_audio', role='genre_train_diagnostic', dataset='FineDance', offset=0.)]
    rows = [row(base, 'Z_registered', [0,150]), row(base, '000_not_registered', [30,180]),
            row(base, 'A_registered', [180,330])]
    return dict(sources=sources, rows=rows, queries=[0,2], source_counts={}), dict(
        seed=66066, offsets=list(OFFSETS), trim_each_side=15, fps=30,
        registered_query_count=2, wrong_song_donors=3,
        wrong_song_rule='FROZEN_METADATA_TOP3_THEN_LOCAL_AUDIO_ONLY_RMS_TEMPO')


def test_source_offset_is_applied_once_and_every_shift_has_common120_support(fixture):
    meta, protocol = fixture
    out = select(meta, protocol)
    case = out['cases'][0]
    assert case['source_interval'] == [0,150]
    assert case['audio_interval'] == [12.,162.]
    assert case['common_motion_interval'] == [15,135]
    assert case['common_audio_interval'] == [27.,147.]
    assert case['common_frames'] == 120
    for delta in OFFSETS:
        a, b = case['shifted_audio_intervals'][str(delta)]
        assert [a,b] == [27.+delta,147.+delta] and b-a == 120
        assert 12 <= a < b <= 162
    assert case['source_audio_offset_frames'] == 12.
    assert case['circular_wrapping'] is False and case['padding_frames'] == 0
    assert case['raw_audio_sample_bounds'].startswith('PENDING_ENGINE_ACTUAL_WAV')


def test_wrong_song_pool_has_role_and_sha_isolation_and_fixed_identity_order(fixture):
    meta, protocol = fixture
    out = select(meta, protocol)
    for case in out['cases']:
        donors = case['wrong_song_candidates']
        assert len(donors) == 3 and len({d['audio_sha256'] for d in donors}) == 3
        assert all(d['audio_sha256'] != case['audio_sha256'] for d in donors)
        assert all((d['dataset'],d['role']) == (case['dataset'],case['role']) for d in donors)
        assert all(d['source_uid'] not in ('val','fine','same_audio_alias') for d in donors)
        assert [d['rank_digest'] for d in donors] == sorted(d['rank_digest'] for d in donors)
    reordered = deepcopy(meta)
    reordered['sources'].reverse()
    assert select(reordered,protocol)['cases'] == out['cases']


def test_only_original_query_indices_remain_and_support_failure_has_no_replacement(fixture):
    meta, protocol = fixture
    meta['sources'][0].update(frames=200, shared_support=dict(frames=200))
    out = select(meta, protocol)
    assert out['frozen_query_indices'] == [0,2]
    assert out['registered_query_count'] == 2 and out['selected_query_count'] == 1
    assert [c['query_uid'] for c in out['cases']] == ['Z_registered']
    assert [c['query_uid'] for c in out['excluded']] == ['A_registered']
    assert out['excluded'][0]['replacement_used'] is False
    assert all(c['query_uid'] != '000_not_registered' for c in out['cases'])
    assert out['display_cases'][0]['query_uid'] == 'A_registered'
    assert out['display_cases'][0]['case_key'] is None and out['display_case_keys'] == []


def test_display_is_lexical_per_stratum_before_scores_and_replay_detaches_inputs(fixture):
    meta, protocol = fixture
    for uid in ('val','fine'):
        s = next(s for s in meta['sources'] if s['source_uid'] == uid)
        meta['queries'].append(len(meta['rows']))
        meta['rows'].append(row(s, uid+'_registered', [0,150]))
    protocol['registered_query_count'] = len(meta['queries'])
    before = deepcopy((meta,protocol))
    out = select(meta,protocol)
    assert len(out['display_cases']) == 3
    train = next(d for d in out['display_cases'] if d['role'] == 'train')
    assert train['query_uid'] == 'A_registered'
    changed = deepcopy(meta)
    for r in changed['rows']:
        r.update(music_activity=999., body_activity=-999.)
    other = select(changed,protocol)
    assert other['display_cases'] == out['display_cases'] and other['cases'] == out['cases']
    assert out == select(meta,protocol) and (meta,protocol) == before
    assert out['population_sha256'] == digest({k:v for k,v in out.items() if k!='population_sha256'})
    json.dumps(out,allow_nan=False)


def test_test_roles_and_changed_offset_population_rejected(fixture):
    meta, protocol = fixture
    bad = deepcopy(protocol)
    bad['offsets'] = [0,8,15]
    with pytest.raises(ValueError,match='offsets'):
        select(meta,bad)
    meta['rows'][0]['role'] = 'test'
    meta['sources'][0]['role'] = 'test'
    with pytest.raises(ValueError,match='Test or unregistered role'):
        select(meta,protocol)
