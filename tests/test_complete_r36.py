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
