import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('r40_crops',Path(__file__).parents[1]/'src/dunhuang/research/contract_r40/crops.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def test_original_clock_not_diagonal_labels():
    i,j=c.clock_labels(90,7,23)
    assert np.array_equal(i+7,j+23)
    assert not np.array_equal(i,j)
    x=c.crop_origins(40041,0,100)
    assert (x[:,0]!=x[:,1]).sum()>80

def test_equal_support_recovers_both_offset_signs_without_clock_input():
    rng=np.random.default_rng(1);x=rng.normal(size=(150,64));x/=np.linalg.norm(x,axis=-1,keepdims=True)
    for shift in (-15,-8,0,8,15):
        m=x[30+shift:120+shift];d=x[30:120];scores=c.offset_scores(m,d,[-15,-8,0,8,15]);assert [-15,-8,0,8,15][scores.argmax()]==shift

def test_uniform_ties_not_argmax_success():
    s=np.ones((5,5));assert np.all(c.tie_accuracy(s,np.arange(5))==.2)

def test_invalid_crop_support_fails():
    import pytest
    with pytest.raises(ValueError):c.clock_labels(60,0,0)
    with pytest.raises(ValueError):c.clock_labels(90,31,0)
# Runtime checks for the continuous-source preflight.
def test_relative_root_gauge_does_not_read_hidden_frames():
    import importlib.util
    import sys
    root=Path(__file__).parents[1]
    sys.path.insert(0,str(root/'src'))
    from dunhuang.research.contract_r40.change_data import rooted_clip
    x=np.arange(300*3,dtype=float).reshape(300,3)/1000
    a,ma=rooted_clip(x,0,256,np.array([-10.,0.,-10.]),np.array([10.,2.,10.]))
    x[110:140]+=100
    b,mb=rooted_clip(x,0,256,np.array([-10.,0.,-10.]),np.array([10.,2.,10.]))
    assert np.array_equal(a[0],b[0])
    assert not mb['model_box_supported'] and not mb['clipped']
    assert np.max(b)>90
