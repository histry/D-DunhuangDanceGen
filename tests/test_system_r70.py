import importlib.util
from pathlib import Path
import numpy as np
import pytest
from dunhuang.contracts import MotionAdapter,sixd
spec=importlib.util.spec_from_file_location('system_r70',Path(__file__).resolve().parents[1]/'scripts/system_r70.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_external_coordinate_conversion_uses_native_yup_and_shared_skeleton():
    x=np.zeros((10,151),np.float32);x[:,7:]=sixd(np.broadcast_to(np.eye(3),(10,24,3,3))).reshape(10,144)
    J=np.c_[np.zeros(22),np.arange(22)*.05,np.zeros(22)];parents=np.r_[-1,np.arange(21)]
    edge=MotionAdapter.to_edge(x)
    actual=m.native_points(edge,J,parents,10)
    np.testing.assert_allclose(actual,np.broadcast_to(J,(10,22,3)),atol=1e-6)
    with pytest.raises(ValueError,match='cover'):m.native_points(edge,J,parents,11)