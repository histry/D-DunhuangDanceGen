import numpy as np
import pytest
from dunhuang.research.music_reperformance_r43.replay import exact_arrays

def test_archive_metadata_difference_keeps_exact_nan_arrays(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz';np.savez(a,x=np.array([1.,np.nan]));np.savez_compressed(b,x=np.array([1.,np.nan]));exact_arrays(a,b);exact_arrays(a,a)

def test_numerical_and_dtype_changes_still_fail(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz';np.savez(a,x=np.array([1,2],dtype=np.int32));np.savez_compressed(b,x=np.array([1,3],dtype=np.int32))
    with pytest.raises(ValueError):exact_arrays(a,b)
    np.savez(b,x=np.array([1,2],dtype=np.float32))
    with pytest.raises(ValueError):exact_arrays(a,b)
