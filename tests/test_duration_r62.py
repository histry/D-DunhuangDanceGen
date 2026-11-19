import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.duration_r62.methods import solve,angles,measure
from dunhuang.research.duration_r62.continuous import interval,source_balanced


def trajectory(gap):
    t=np.arange(gap+8)/(gap+7)
    return Rotation.from_rotvec((t[:,None,None]*np.tile([.05,.1,.03],(22,1))[None]).reshape(-1,3)).as_matrix().reshape(gap+8,22,3,3)


@pytest.mark.parametrize('gap',[24,36,48,60,72])
def test_hidden_poison_and_exact_visible(gap):
    raw=trajectory(gap);a,proof=solve(raw,gap,controls=True)
    bad=raw.copy();bad[4:4+gap]=np.nan;b,other=solve(bad,gap,controls=True)
    bad[4:4+gap]=98765;c,_=solve(bad,gap,controls=True)
    assert proof==other and proof['solver_hidden_read'] is False
    assert proof['jet_consumed_local_frames']==[1,2,3,gap+4,gap+5,gap+6]
    for name in ('G1','SLERP','SingleChartHermite'):
        assert a[name] is not None
        assert np.array_equal(a[name],b[name]) and np.array_equal(a[name],c[name])
        assert np.array_equal(a[name][:4],raw[:4]) and np.array_equal(a[name][-4:],raw[-4:])
    assert angles(a['SLERP'][4:4+gap],raw[4:4+gap]).max()<1e-12


@pytest.mark.parametrize('gap',[24,36,48,60,72])
def test_main_and_controls_G1_identical(gap):
    a,ea=solve(trajectory(gap),gap);b,eb=solve(trajectory(gap),gap,controls=True)
    assert list(a)==['G1'] and np.array_equal(a['G1'],b['G1'])
    assert ea['guards']['G1']==eb['guards']['G1']


def test_frozen_G1_and_single_chart_share_identical_observed_jets(monkeypatch):
    calls=[]
    def fake(R,gap,counts,radius):
        calls.append((R.copy(),gap,counts,radius));return None,dict(status='REJECT')
    monkeypatch.setitem(solve.__globals__,'segmented',fake)
    curves,p=solve(trajectory(48),48,controls=True)
    assert curves['G1'] is None and curves['SingleChartHermite'] is None and curves['SLERP'] is not None
    assert np.array_equal(calls[0][0],calls[1][0],equal_nan=True)
    assert calls[0][1:]==((4,52),(4,8,16),np.pi-.05)
    assert calls[1][1:]==((4,52),(1,),np.pi-.05)
    assert p['rejection_fallback'] is False


@pytest.mark.parametrize('gap',[True,0,-2,1.5])
def test_invalid_gap_contract(gap):
    with pytest.raises(ValueError):solve(trajectory(48),gap)


def test_missing_boundary_rejected():
    raw=trajectory(48);raw[0]=np.nan
    with pytest.raises(ValueError):solve(raw,48)


def test_metadata_eligibility_preserves_fixed_center():
    assert interval(100,24,[50,150])==dict(source_interval=[84,116],edit_source_interval=[88,112],eligible=True)
    q=interval(100,72,[70,140])
    assert q['source_interval']==[60,140] and q['eligible'] is False
    with pytest.raises(ValueError):interval(100,25,[0,200])


def test_source_balancing_ignores_unconstructible_difference_but_not_population():
    rows=[dict(source_uid='a',d=1.)]*12+[dict(source_uid='b',d=3.),dict(source_uid='c',d=None)]
    q=source_balanced(rows,'d')
    assert q['mean']==2 and q['sources']==2 and q['tasks']==13
    assert source_balanced(rows,'missing')['mean'] is None


def test_48_baseline_curve_and_measurements_match_R61_exactly():
    from dunhuang.research.edit_support_r61.fixed_gap import solve as old_solve,measure as old_measure
    raw=trajectory(48)
    raw[4:52]=Rotation.from_rotvec(np.tile([.3,.1,-.1],(48*22,1))).as_matrix().reshape(48,22,3,3)
    new,proof=solve(raw,48,controls=True);old,old_proof=old_solve(raw)
    assert np.array_equal(new['G1'],old['G1']) and np.array_equal(new['SLERP'],old['SLERP'])
    assert proof['guards']['G1']==old_proof['G1']
    J=np.zeros((22,3));J[:,1]=np.arange(22)*.1;parents=np.r_[-1,np.arange(21)]
    ref=dict(D2=1000.,D3=1000.,shoulder_wrist_lower_m=0.)
    a=measure(new['G1'],raw,J,parents,ref,48);b=old_measure(old['G1'],raw,J,parents,ref)
    a.pop('target_center_scope');b.pop('target_center_scope')
    assert a==b
