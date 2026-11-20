"""Synthetic metadata contracts; no waveform or feature arrays are loaded."""
import copy
import hashlib
import json

import pytest

from dunhuang.research.sdmr_warp_rank import population as p


@pytest.fixture
def registered(tmp_path, monkeypatch):
    sources, rows = [], []
    for group in range(6):
        for recording in range(2):
            uid = f's{group}_{recording}'
            source = dict(source_uid=uid,dataset='AIST++',role='val',group=f'music{group}',
                recording_uid=uid,raw_motion_sha256=f'{group:02}{recording:02}'+'a'*60,
                audio_sha256=f'{group:02}'+'b'*62,audio_offset_frames=2.5,
                shared_start=10,frames=310,shared_support=dict(frames=310),
                clock_scope='TEST_FIXTURE_PHYSICAL_CLOCK')
            sources.append(source)
            for start in (10,85,160):
                rows.append(dict(source, uid=f'{uid}_{start}',source_interval=[start,start+150]))
    meta = dict(sources=sources,rows=rows,queries=list(range(len(rows))),
                shared_choreography_across_AIST_roles=['already_exposed'])
    path = tmp_path/'manifest.json'
    def save():
        path.write_text(json.dumps(meta,sort_keys=True),encoding='utf8')
        monkeypatch.setattr(p,'PARENT_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    save()
    return meta,path,save


def test_only_registered_recordings_center240_offsets_once_and_no_mutation(registered):
    meta,path,_ = registered
    before = copy.deepcopy(meta)
    result = p.paired(path)
    assert len(result['cases']) == result['eligible_cases'] == 12
    for case in result['cases']:
        assert case['source_interval'] == [40,280]
        assert case['target_interval'] == case['audio_interval'] == [42.5,282.5]
        assert case['sourceMnormTrue'] and len(case['original_query_indices']) == 3
        assert all(case['source_interval'] != interval for interval in case['original_query_intervals'])
        assert not case['replacement_used'] and not case['selection_uses_scores']
    assert meta == before and result['training_updates'] == result['test_tensor_reads'] == 0


def test_wrong_music_is_fixed_next_group_role_safe_and_distinct_sha(registered):
    _,path,_ = registered
    result = p.paired(path)
    for case in result['cases']:
        donor = case['donor']
        expected = (int(case['group'][-1])+1)%6
        assert donor['group'] == f'music{expected}' and donor['source_uid'] == f's{expected}_0'
        assert donor['audio_sha256'] != case['audio_sha256']
        assert (donor['dataset'],donor['role']) == ('AIST++','val')
        assert donor['source_interval'] == [40,280] and donor['sourceMnormTrue']


def test_insufficient_fixed_source_is_retained_without_substitution(registered):
    meta,path,save = registered
    meta['sources'][0]['frames'] = meta['sources'][0]['shared_support']['frames'] = 200
    save()
    result = p.paired(path)
    case = next(c for c in result['cases'] if c['source_uid'] == 's0_0')
    assert len(result['cases']) == 12 and not case['eligible'] and case['source_interval'] is None
    assert not case['replacement_used']
    assert any(r['source_uid']=='s0_0' for r in result['excluded'])


def test_manifest_change_fails_before_metadata_selection(registered):
    _,path,_ = registered
    path.write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='manifest bytes changed'):
        p.paired(path)
