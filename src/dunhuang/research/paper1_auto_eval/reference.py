"""Numerical parity on synthetic 24-joint inputs, not a perceptual calibration."""
import ast
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from .common import ROOT,CFG,sha
from .music import motion_beats,gaussian_match

def check():
    p=ROOT/'reference/mace_metric.py';assert sha(p)==CFG['reference_sha256']
    tree=ast.parse(p.read_text(encoding='utf8'));nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ['BA','get_db']];assert len(nodes)==2
    scope=dict(np=np,G=gaussian_filter,argrelextrema=argrelextrema);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    differences=[]
    for seed in [76001,76002]:
        rng=np.random.default_rng(seed);t=np.arange(300)[:,None,None];P=np.sin(t*rng.uniform(.03,.13,(1,24,3))+rng.uniform(0,6,(1,24,3)));mb=np.arange(15,285,30);ref=scope['get_db'](P);actual=motion_beats(P)
        assert np.allclose(ref[0],actual*30,atol=1e-10,rtol=0)
        error=abs(float(scope['BA'](mb,ref))-gaussian_match(mb/30,actual));assert error<1e-12;differences.append(error)
    return dict(status='PASS',official_commit=CFG['reference_commit'],reference_sha256=sha(p),nonempty_24_joint_synthetic_sequences=2,maximum_score_difference=max(differences),adaptations=['22_ROOT_RELATIVE_JOINTS_IN_ACTUAL_AUDIT','EMPTY_EVENTS_RETURN_NULL_NOT_ZERO','EXPLICIT_ORIGINAL_SECONDS_AND_READONLY_BEAT_CATALOG'],human_calibration=False)
