import numpy as np
import pytest
from dunhuang.paper1.warp_bank import build
from dunhuang.paper1.contracts import source_reads
from dunhuang.research.complete_r36.core import multi_warp

@pytest.mark.parametrize('length',[180,233,240,300])
def test_variable_actual_centres(length):
    u=dict(source_uid='test',unit_uid='unit',source_interval=[100,100+length],phases=[dict(frame=100+length//2,type='speed_peak',body_region='upper')])
    A=900;beats=list(range(A+15,A+length,20));rows,proof=build(u,[A,A+length],beats)
    original=multi_warp(u['source_interval'],[A,A+length],[dict(p,type='upper_speed_peak') for p in u['phases']],beats)
    assert rows[0]['time_warp']==original['time_warp'] and len(rows)<=2
    for r in rows:
        assert len(r['q'])==length and r['q'][0]==100 and r['q'][-1]==99+length
        assert r['certificate']['status']=='REACHABLE_BY_RATE'
    if len(rows)==2:assert not np.allclose(rows[0]['q'],rows[1]['q'],atol=1e-9,rtol=0)

def test_no_fill_when_only_uniform():
    rows,p=build(dict(source_uid='x',unit_uid='u',source_interval=[0,240],phases=[]),[900,1140],[])
    assert len(rows)==1 and p['artificial_fill']==0

def test_infeasible_kept_absent():
    rows,p=build(dict(source_uid='x',unit_uid='u',source_interval=[0,80],phases=[]),[0,240],[])
    assert rows==[] and p['reason']=='ORIGINAL_E1W_INFEASIBLE'

def test_source_and_target_limits():
    with pytest.raises(ValueError):source_reads(np.linspace(-1,239,240),[0,240],[0,240])
    with pytest.raises(ValueError):source_reads(np.linspace(0,239,239),[0,240],[0,240])

def test_whole_frame_holes_fail():
    from dunhuang.research.whole_song_geometry.study import coverage
    with pytest.raises(ValueError):coverage([dict(interval=[0,3]),dict(interval=[4,8])],8)

def test_shadow_variable_support(monkeypatch):
    from dunhuang.paper1 import sdmr_adapter as a
    class Fake:
        binding={'test_fixture':True}
        def encode_music_raw(self,v,m):return v[6:-6,:3],m[6:-6].any(-1),{}
        def encode_rotations(self,R):return np.ones((len(R)-12,3)),np.ones(len(R)-12,bool),{}
        def check(self):return 'fixture'
    N=233;music={k:np.ones((N,50)) for k in ('actual','wrong','zero')};mask={k:np.ones((N,50),bool) for k in music};mask['wrong'][-2:]=False
    r=a.score_executed_bank_shadow(Fake(),music,mask,[np.zeros((N,22,3,3))]*2)
    assert not r['can_route'] and len(r['common_support'])==N-12
    hashes=[d['readout']['support']['common_support_sha256'] for d in r['details'].values()]
    assert len(set(hashes))==1 and all(len(v['Single64'])==2 for v in r['scores'].values())

@pytest.mark.parametrize('frames',[180,233,300])
def test_dynamic_readout_does_not_reuse_fixed240(frames):
    from dunhuang.paper1.evaluation import measure
    source=np.broadcast_to(np.eye(3),(frames,22,3,3)).copy();J=np.zeros((22,3));parents=np.r_[-1,np.zeros(21,dtype=int)]
    r=measure(source,source,J,parents,np.arange(frames),[0,frames])
    assert r['frames']==frames and r['sampler_exact'] and r['actual']['frames']==frames
