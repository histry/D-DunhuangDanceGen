import numpy as np
from dunhuang.research.compensation_r73.warp import proposals,solve,field,match

def test_same_anchor_edge_contract_and_signed_dose():
    case=dict(time_warp=[[0,0],[239,239]],target_interval=[0,240])
    ps=proposals(120,np.linspace(0,1,240));p=next(p for p in ps if p['kind']=='High' and p['width']==60)
    for delta in [-6,6,-3,3]:
        r=solve(case,p,delta);assert r['status']=='LEGAL'
        assert np.array_equal(r['q'][:4],np.arange(4)) and np.array_equal(r['q'][-4:],np.arange(236,240))
        assert abs(field(120,p['segments'])-1)<1e-10
        assert np.all(np.diff(r['q'])>0)
        inverse=r['q']
        assert np.array_equal(r['compensation_mask'],abs(field(inverse,p['segments'],1))>1e-12)
        assert np.max(abs(inverse+delta*field(inverse,p['segments'])-np.arange(240)))<1e-8

def test_distributed_not_falsely_claimed_same_curvature_budget():
    case=dict(time_warp=[[0,0],[239,239]],target_interval=[0,240]);ps=proposals(120,np.ones(240))
    h=solve(case,next(p for p in ps if p['kind']=='High' and p['width']==60),3)
    d=solve(case,next(p for p in ps if p['kind']=='Distributed' and p['width']==60),3)
    assert abs(h['dose']['postmap_derivative_energy']['1']-d['dose']['postmap_derivative_energy']['1'])<1e-8
    assert d['dose']['postmap_derivative_energy']['2']>3.9*h['dose']['postmap_derivative_energy']['2']

def test_illegal_dose_never_repaired_or_clipped():
    case=dict(time_warp=[[0,0],[239,239]],target_interval=[0,240]);p=proposals(120,np.ones(240))[0]
    r=solve(case,p,30);assert r['status']=='NOT_FEASIBLE' and r['q'] is None


def test_numpy_match_decisions_serialize_without_changing_truth():
    import json
    from dunhuang.research.local_timing_r68.study import plain
    record={'primary_match': {'eligible': np.bool_(False)}, 'dose': np.float64(3.)}
    recovered=json.loads(json.dumps(plain(record),allow_nan=False))
    assert recovered=={'primary_match': {'eligible': False}, 'dose': 3.}
