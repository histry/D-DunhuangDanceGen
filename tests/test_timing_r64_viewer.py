"""R64 packet clock slicing and unqualified music-label constraints."""
import json

import numpy as np
import pytest

from dunhuang.research.timing_r64 import viewer as v


def test_clock_slices_actual30hz_acoustics_and_retains_unknown_truth():
    clock = dict(beat_frames=[90, 100, 627, 628], accent_strength=list(range(700)),
                 onset_strength=list(range(700)), section_proxy_frames=[90, 110, 640],
                 phrase_proxy_frames=[115], downbeat_frames=[100],
                 independent_phrase_labels=['unverified-label'])
    out = v.music_clock(clock, dict(music_interval=[100, 628], audio_sha256='audio'))
    assert out['beat_frames'] == [0., 527.]
    assert len(out['accent_strength']) == 528
    assert out['accent_strength'][0] == 100 and out['accent_strength'][-1] == 627
    assert out['section_proxy_frames'] == [10.]
    assert out['phrase_proxy_frames'] == [15.]
    assert out['downbeat_frames'] is None and out['independent_phrase_labels'] is None
    assert out['downbeat_status'] == out['independent_phrase_status'] == 'UNKNOWN'
    assert 'NOT_INDEPENDENT_PHRASE_TRUTH' in out['section_scope']
    with pytest.raises(ValueError, match='cover'):
        v.music_clock(dict(accent_strength=[0]*500), dict(music_interval=[100, 628]))


def test_template_retains_actual_audio_master_and_source_readout_only():
    html = v.template()
    assert '__R64_PAYLOAD__' in html and '__R63_PAYLOAD__' not in html
    assert 'Math.floor(audio.currentTime*D.fps' in html
    assert 'id="sourceClock"' in html and 'id="musicTimeline"' in html
    assert "window.R64_EVIDENCE.downbeat_status='UNKNOWN'" in html
    assert "window.R64_EVIDENCE.independent_phrase_status='UNKNOWN'" in html
    assert '来源帧映射（当前重定时预览）' in html
    assert '原速播放' not in html
    assert v.MAIN_POLICIES == ('Fixed48', 'LocalBeat')
    assert 'BalancedEqual' not in v.CONTROL_POLICIES


def test_per_case_array_files_and_single_embedded_safe_packet(tmp_path, monkeypatch):
    (tmp_path/'main').mkdir()
    cases, array_files = [], {}
    for i in range(2):
        key = 'case%d' % i
        path = tmp_path/'graph'/key/'arrays.npz'
        path.parent.mkdir(parents=True)
        np.savez(path, J=np.zeros((22, 3)), parents=np.arange(-1, 21))
        array_files[key] = str(path.relative_to(tmp_path))
        cases.append(dict(key=key, role='DEV' if i == 0 else 'EVAL', songidx=0,
            music_interval=[100, 628], beats=[120, 180],
            left=dict(source_uid='left', event_uid='L', source_interval=[0, 240]),
            right=dict(source_uid='right', event_uid='R', source_interval=[0, 240])))
    result = dict(cases=cases, records=[], selections=[], array_files=array_files,
                  music_clocks={'0': dict(beat_frames=[120, 180], accent_strength=[0.]*700)}, binding={})
    v.write(tmp_path/'main'/'result.json', result)
    def fake_audio(case, out, seen):
        (out/'audio').mkdir(exist_ok=True)
        (out/'audio'/'same.m4a').write_bytes(b'fake')
        return dict(path='audio/same.m4a')
    monkeypatch.setattr(v.frozen, '_audio', fake_audio)
    manifest = v.viewer(tmp_path)
    out = tmp_path/'viewer'
    assert {p['path'] for p in manifest['files']} == {'index.html', 'protocol.json', 'audio/same.m4a'}
    assert not (out/'data.json').exists()
    assert (tmp_path/'viewer_manifest.json').exists()
    html = (out/'index.html').read_text(encoding='utf8')
    start = html.index('<script id="r64-data" type="application/json">')
    start = html.index('>', start)+1
    data = json.loads(html[start:html.index('</script>', start)])
    assert data['case_count'] == 2 and data['fps'] == 30 and data['frames'] == 528
    assert len(data['cases'][0]['panels']) == 2
    assert all(p['data'] is None for c in data['cases'] for p in c['panels'])
    assert data['cases'][0]['music_clock']['beat_frames'] == [20., 80.]
    protocol = v.read(out/'protocol.json')
    assert protocol['source_roles'] == {'DEV': 1, 'EVAL': 1}
    assert protocol['quality_probability_available'] is False
    assert protocol['downbeat_status'] == protocol['independent_phrase_status'] == 'UNKNOWN'
    assert 'READOUT_ONLY' in protocol['source_clock_selector']


def test_graph_array_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match='inside'):
        v._array_path(tmp_path, '../other/arrays.npz')
