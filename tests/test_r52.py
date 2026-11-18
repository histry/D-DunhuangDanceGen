import numpy as np
import pytest
from dunhuang.research.validation_r51.common import regions


def test_absolute_regions_equal_explicit_local_clock_and_fail_without_origin():
 n=40;R=np.broadcast_to(np.eye(3),(n,22,3,3)).copy();J=np.zeros((22,3));parents=np.r_[-1,np.arange(21)]
 spans=[dict(kind='core',interval=[1500,1516]),dict(kind='transition',interval=[1516,1524]),dict(kind='core',interval=[1524,1540])]
 with pytest.raises(ValueError,match='clock'):regions(R,J,parents,spans)
 absolute=regions(R,J,parents,spans,origin=1500)
 local=regions(R,J,parents,[dict(r,interval=[t-1500 for t in r['interval']]) for r in spans])
 for a,b in zip(absolute['regions'],local['regions']):
  assert a['interval']==b['interval'] and a['metrics']==b['metrics']
  assert a['metrics']['D3_Q95']==0 and a['metrics']['frames']>0
 assert len(local['regions'])==5
 assert absolute['whole']==local['whole']
