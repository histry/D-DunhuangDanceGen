"""Display provenance and fail-closed packet checks; browser playback is separate."""
import base64
import json
from pathlib import Path

import numpy as np
import pytest

from dunhuang.research.budget_r63 import viewer as v


def test_panel_decodes_actual_fk_and_rejected_has_no_substitute():
    R = np.tile(np.eye(3), (528, 22, 1, 1))
    J = np.zeros((22, 3)); J[:, 1] = np.arange(22)*.1
    parents = np.arange(-1, 21)
    method = dict(constructible=True, passes_original_screen=True,
                  arrays_key='candidate', full_R_sha256=v.array_hash(R), core_exact=True)
    side = dict(status='LEGAL_CORE', source_uid='uid', source_interval=[0, 240],
                target_interval=[100, 340], source_positions=list(range(240)),
                time_warp=[[0, 100], [239, 339]])
    record = dict(allocation=dict(left_frames=240, right_frames=240), gap_frames=48,
                  left=side, right=side, methods={'G1': method})
    selection = dict(record_key='record', accepted=True, method='G1')
    panel = v._panel('Fixed48', selection, {'record': record}, {'candidate': R}, J, parents)
    values = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(528, 22, 3)
    assert np.array_equal(values, v.relative_fk(R, J, parents).astype('<f4'))
    assert panel['regions'][1] == dict(kind='TRANSITION', interval=[240, 288])
    rejected = v._panel('Fixed48', dict(selection, accepted=False), {'record': record},
                        {'candidate': R}, J, parents)
    assert rejected['data'] is None and rejected['status'] == 'REJECTED_BLANK'
    method['full_R_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash mismatch'):
        v._panel('Fixed48', selection, {'record': record}, {'candidate': R}, J, parents)


def test_viewer_safe_packet_and_all_cases_including_failure(tmp_path, monkeypatch):
    study = tmp_path/'main'; study.mkdir()
    cases = [dict(key='case%d' % i, role='DEV' if i == 0 else 'EVAL',
                  music_interval=[100, 628], beats=[120, 180], songidx=0,
                  left=dict(source_uid='left', event_uid='left_event', source_interval=[0, 240]),
                  right=dict(source_uid='right', event_uid='right_event', source_interval=[20, 260]))
             for i in range(2)]
    result = dict(cases=cases, records=[], selections=[], binding={'commit': 'test'})
    v.write(study/'result.json', result)
    np.savez(study/'arrays.npz', J=np.zeros((22, 3)), parents=np.arange(-1, 21))
    def fake_audio(case, out, seen):
        folder = out/'audio'; folder.mkdir(exist_ok=True)
        (folder/'clip.m4a').write_bytes(b'unit-test-audio-placeholder')
        return dict(path='audio/clip.m4a', source_frame_interval=case['music_interval'])
    monkeypatch.setattr(v, '_audio', fake_audio)
    manifest = v.viewer(tmp_path)
    packet = tmp_path/'viewer'
    data = v.read(packet/'data.json')
    assert len(data['cases']) == 2
    assert data['frames'] == 528 and data['fps'] == 30
    assert all(p['data'] is None for c in data['cases'] for p in c['panels'])
    html = (packet/'index.html').read_text(encoding='utf8')
    start = html.index('<script id="r63-data" type="application/json">')
    start = html.index('>', start)+1
    embedded = json.loads(html[start:html.index('</script>', start)])
    assert embedded == data
    for field in ('#play', '#prev', '#next', '#seek', '#rate', '#masterAudio', '#evidence'):
        assert field in v.read(packet/'protocol.json')['acceptance_dom']
    paths = {row['path'] for row in manifest['files']}
    assert paths == {'index.html', 'data.json', 'protocol.json', 'audio/clip.m4a'}
    assert all(not p.endswith('.npz') for p in paths)
    assert data['main_result_sha256'] == v.sha(study/'result.json')
    with pytest.raises(ValueError, match='Fresh'):
        v.viewer(tmp_path)
