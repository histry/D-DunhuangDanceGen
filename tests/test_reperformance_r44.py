import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.reperformance_r44.phase import smooth_map,maps
from dunhuang.research.reperformance_r44.motion import retime,decode

CFG=json.loads((Path(__file__).parents[1]/'src/dunhuang/research/reperformance_r44/config.json').read_text())

def raw_motion():
    x=np.zeros((256,139));R=Rotation.from_rotvec(np.c_[np.sin(np.arange(256)/11)*.3,np.zeros((256,2))]).as_matrix();x[:,7:]=sixd(np.repeat(R[:,None],22,axis=1),row=True).reshape(256,132);x[:,4]=np.arange(256)*.001;return x

def test_analytical_rate_and_c2_endpoint_join():
    p,r=smooth_map([8,35,70,98],[8,41,64,98],CFG);t=np.array(r['target']);assert r['certificate']['legal'];np.testing.assert_allclose(p(t),r['source']);np.testing.assert_allclose(p(t,1),1,atol=1e-10);np.testing.assert_allclose(p(t,2),0,atol=1e-10)

def test_observed_path_integer_anchors_and_unknown_contact():
    raw=raw_motion();R,_=decode(raw);linear,q,records=maps(R,[20,43,66,90,175,201,230],CFG);out=retime(raw,q);protected=np.ones(256,bool)
    for a,b in CFG['core_intervals']:protected[a:b]=False
    assert np.array_equal(raw[protected],out[protected]);assert np.array_equal(raw[:,:4],out[:,:4]);assert np.array_equal(retime(raw,np.arange(256)),raw)
    for r in records:
        for s,t in zip(r['source'],r['target']):assert np.array_equal(out[int(t),4:],raw[int(s),4:])

def test_quiet_sources_are_not_given_false_peaks():
    raw=raw_motion();raw[:,7:]=raw[0,7:];R,_=decode(raw);_,q,records=maps(R,[30,60,90,180,210],CFG);assert np.array_equal(q,np.arange(256));assert all(r['static_preserved'] for r in records)

@pytest.mark.parametrize('source,target', [([0,0],[0,1]),([0,10],[0,float('nan')]),([0,10],[0])])
def test_invalid_maps_fail_closed(source,target):
    with pytest.raises(ValueError):smooth_map(source,target,CFG)
