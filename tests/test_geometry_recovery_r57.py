import numpy as np,pytest
from types import SimpleNamespace
from dunhuang.research.geometry_recovery_r57 import study as mod
def mocked_inputs(monkeypatch):
 monkeypatch.setattr(mod,'certificate',lambda *a:dict(chart_eligible=True));monkeypatch.setattr(mod,'metrics',lambda *a:dict(gap_D2=1.,gap_D3=2.,shoulder_wrist_min_m=.5));left=np.tile(np.eye(3),(8,22,1,1));return left,dict(D2=2.,D3=3.,shoulder_wrist_lower_m=.1)
def test_g0_chart_failure_does_not_discard_independent_g1(monkeypatch):
 left,ref=mocked_inputs(monkeypatch);called=[]
 def fail(*a):raise ValueError('Ambiguous SO3 chart')
 def g1(observed,gap,*a):
  called.append(gap);q=observed.copy();q[gap[0]:gap[1]]=np.eye(3);return q,dict(status='GEOMETRICALLY_ADMITTED_NOT_PHYSICS')
 monkeypatch.setattr(mod,'observed_bridge',fail);monkeypatch.setattr(mod,'segmented',g1);q,e=mod.bridge(left,left,48,SimpleNamespace(joints=None,parents=None),ref);assert q.shape==(48,22,3,3) and e['selected']=='G1' and len(called)==1;assert not e['physical_qualified']
def test_unrelated_g0_exception_is_not_swallowed(monkeypatch):
 left,ref=mocked_inputs(monkeypatch)
 def fail(*a):raise ValueError('missing observation')
 monkeypatch.setattr(mod,'observed_bridge',fail)
 with pytest.raises(ValueError,match='missing observation'):mod.bridge(left,left,48,SimpleNamespace(),ref)
