import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import components,multi_warp,bounded_rotations,smooth_budget,rotation_sample

def test_source_gaps_cannot_be_merged():
    assert components([[0,80],[80,180],[190,250]])==[[0,180],[190,250]]
    with pytest.raises(ValueError):components([[0,80],[79,90]])

def test_multi_phase_clock_is_monotone_and_in_bounds():
    phases=[dict(frame=t,type='upper_speed_peak') for t in (40,100,160)]
    w=multi_warp([0,200],[50,270],phases,list(range(50,270,15)))
    a=np.array(w['time_warp']);d=np.diff(a,axis=0)
    assert (a[0]==[0,50]).all() and (a[-1]==[199,269]).all()
    assert len(a)>3 and (d>0).all() and (d[:,1]/d[:,0]>=.75).all() and (d[:,1]/d[:,0]<=1.33).all()
    assert multi_warp([0,100],[0,300],phases,[]) is None

def test_old_and_new_half_open_endpoint_contracts_stay_distinct():
    from dunhuang.planning.time_map import source_positions
    old=dict(source_interval=[0,30],music_interval=[0,36],time_warp=[[0,0],[30,36]])
    positions=source_positions(old,np.arange(36));assert positions.max()==29
    new=multi_warp([0,30],[0,36],[],[])['time_warp']
    assert new[-1]==[29,35.] and np.interp(35,np.array(new)[:,1],np.array(new)[:,0])==29

def test_residual_rotation_contract_and_zero_budget():
    r=np.tile(np.eye(3),(12,22,1,1));v=np.zeros((12,22,3));v[...,1]=1.
    p=Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(r.shape)
    out=bounded_rotations(r,p,np.full((12,1),.2))
    assert np.max(abs(Rotation.from_matrix(out.reshape(-1,3,3)).magnitude()-.2))<1e-8
    assert np.array_equal(bounded_rotations(r,p,np.zeros((12,1))),r)
    assert np.max(abs(out.swapaxes(-1,-2)@out-np.eye(3)))<1e-12
    with pytest.raises(ValueError):bounded_rotations(r,p,-np.ones((12,1)))

def test_source_read_cannot_extrapolate():
    r=np.tile(np.eye(3),(5,22,1,1))
    assert rotation_sample(r,[0,1.5,4]).shape==(3,22,3,3)
    with pytest.raises(ValueError):rotation_sample(r,[4.1])
    budget=smooth_budget(256)
    assert budget[0]==budget[-1]==.6 and np.isclose(budget[128],.2)

def test_training_factors_are_not_the_old_locked_pairings():
    pytest.importorskip('torch')
    from dunhuang.research.complete_r36.train import schedule
    rows=schedule(list(range(16)),36041)
    assert len(rows)==160 and sum(r['task']=='change' for r in rows)==40
    assert {r['provenance'] for r in rows}=={0,1,2}
    paired=[r for r in rows if r['task']=='paired']
    for t in (999,519,119,19):
        assert {r['state'] for r in paired if r['t']==t}=={'actual','q_sample'}
    assert all(len(set(r['cases']))==4 for r in rows)

def test_viewer_packages_multiple_generation_cases_and_both_seeds(tmp_path,monkeypatch):
    import json,base64,gzip
    from dunhuang.research.complete_r36 import view
    root=tmp_path/'run';(root/'planning').mkdir(parents=True);(root/'planning/result.json').write_text('{"songs": []}');(root/'planning/catalog.json').write_text('{"units": []}')
    parent=tmp_path/'baseline/generation';parent.mkdir(parents=True);np.savez(parent/'arrays.npz',skeleton_J22=np.zeros((22,3)),parents=np.array([-1]+[0]*21))
    generated=tmp_path/'gen';generated.mkdir();records=[];arrays={}
    for i in range(3):
        key=f'case{i}';arrays[key+'_reference22']=np.zeros((256,22,3))
        for j,arm in enumerate(('Frozen','Context','Edit')):arrays[key+'_'+arm+'_22']=np.full((2,256,22,3),i+j*.1)
        (generated/(key+'.wav')).write_bytes(b'fixture-audio')
        binding=dict(window_start=0,music_interval=[64,160],source_uid='fixture',source_interval=[0,96],source_memory_frames=96,context_intervals=[[56,64],[160,168]])
        records.append(dict(key=key,name=key,binding=binding,responsible_interval=[52,172],diagnostics={},all_seed_metrics={'Edit':[{},{}]}))
    np.savez(generated/'arrays.npz',**arrays);(generated/'result.json').write_text(json.dumps({'records':records}))
    class EmptyLibrary:rows={}
    monkeypatch.setattr(view,'Library',lambda _:EmptyLibrary())
    out=tmp_path/'viewer';view.run(dict(catalog='fixture',r35=str(parent.parent)),root,out,generated)
    text=(out/'data.js').read_text();payload=json.loads(gzip.decompress(base64.b64decode(text.split('"')[1])))
    assert len(payload['cases'])==3
    assert all(len(c['panels'])==4 and len(c['panels'][3]['variants'])==2 for c in payload['cases'])
