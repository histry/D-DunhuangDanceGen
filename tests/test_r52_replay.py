import json
import numpy as np
import pytest
from dunhuang.research.strong_baseline_r52.publication import compare


def test_execution_metadata_exception_cannot_hide_scientific_or_array_changes(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
 left=dict(weights=dict(device='cuda',sha256='pinned'),records=[dict(score=1)])
 right=dict(weights=dict(device='cpu',sha256='pinned'),records=[dict(score=1)])
 def save():
  (a/'result.json').write_text(json.dumps(left));(b/'result.json').write_text(json.dumps(right))
 save();np.savez(a/'arrays.npz',x=np.arange(5));np.savez(b/'arrays.npz',x=np.arange(5))
 with pytest.raises(AssertionError):compare(a,b)
 assert compare(a,b,True)['exact_arrays']==1
 right['records'][0]['score']=2;save()
 with pytest.raises(AssertionError):compare(a,b,True)
 right['records'][0]['score']=1;right['weights']['sha256']='changed';save()
 with pytest.raises(AssertionError):compare(a,b,True)
 right['weights']['sha256']='pinned';save();np.savez(b/'arrays.npz',x=np.arange(5)+.00001)
 with pytest.raises(AssertionError):compare(a,b,True)
