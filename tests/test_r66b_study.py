"""TEST_FIXTURE study integration; never imports real research arrays/audio."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from dunhuang.research.rhythm_r66 import study,viewer

@pytest.fixture
def fake(monkeypatch):
    R=np.broadcast_to(np.eye(3),(300,22,3,3)).copy()
    case=dict(key='TEST_FIXTURE',source_uid='fake',source_interval=[20,260],target_interval=[500,740],
        phase_inputs=[],phases=[],beats=[510,550,600,650,700],time_warp=[[20,500.],[259,739.]],
        songidx=0,audio_sha256='TEST_FIXTURE',support_component=[0,300])
    pop=dict(cases=[case],source_interval_count=1,source_count=1,music_count=1)
    J=np.zeros((22,3));J[:,1]=np.arange(22)*.05;parents=np.arange(-1,21)
    monkeypatch.setattr(study.population_b,'build',lambda _:deepcopy(pop))
    monkeypatch.setattr(study.population_b,'load_assets',lambda _:dict(lib=SimpleNamespace(rotations={'fake':R}),J=J,parents=parents))
    monkeypatch.setattr(study,'clocks',lambda:{'0':dict(frames=1000,beat_frames=[510,550,600,650,700],accent_frames=[550,650],accent_strength=[1.]*1000,audio_sha256='TEST_FIXTURE')})
    monkeypatch.setattr(study,'multi_warp',lambda *a,**k:dict(time_warp=case['time_warp'],cost=0.))
    monkeypatch.setattr(study,'binding',lambda:dict(commit='TEST_FIXTURE',code={},dependencies={}))
    captures=[]
    monkeypatch.setattr(viewer,'build',lambda target,display,*args:captures.extend(display))
    return captures

def test_registered_missing_candidate_stays_visible_and_no_bridge(tmp_path,fake):
    out=tmp_path/'effect'
    study.compute('main',out)
    result=study.read(out/'result.json')
    assert len(result['records'])==3
    assert [r['status'] for r in result['records']]==['LEGAL','LEGAL','NOT_EVALUABLE']
    assert len(fake[0][1])==3 and fake[0][1][-1]['R'] is None
    assert all(not r['adjacent_bridge_revalidated'] for r in result['records'])
    assert all(r['target_interval']==[500,740] for r in result['records'])

def test_no_control_rerun_without_exact_main_packet(tmp_path,fake,monkeypatch):
    root=tmp_path/'root';monkeypatch.setattr(study,'ROOT',root)
    study.compute('main',root/'main')
    (root/'main/result.json').write_text('{}')
    with pytest.raises(ValueError,match='Main candidate changed'):
        study.compute('controls',root/'controls')
