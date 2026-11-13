import json,re
import numpy as np
import pytest
from dunhuang.research.source_expression.viewer_v2 import modes,old_points,page
def test_fitted_root_motion_is_preserved_without_changing_pose():
    relative=np.zeros((4,22,3));relative[:,10,1]=-1
    root=np.array([[2,1,3],[3,1,3],[4,1.1,3],[5,1.1,3]])
    value=modes(relative,root)
    assert np.array_equal(value["pose"],relative)
    assert np.allclose(np.diff(value["world"][:,10],axis=0),np.diff(root,axis=0))
    assert np.array_equal(relative[:,0],np.zeros((4,3)))
def test_reject_incomplete_or_nonfinite_root():
    with pytest.raises(ValueError):modes(np.zeros((4,22,3)),np.zeros((3,3)))
    with pytest.raises(ValueError):modes(np.zeros((4,22,3)),np.full((4,3),np.nan))
def test_original_frame_samples_survive_page_serialization():
    relative=np.zeros((4,22,3));relative[2,12]=np.array([.12345671,.23456712,.34567891],np.float32);root=np.arange(12,dtype=float).reshape(4,3)/1000
    html=page(relative,root,"EVENT")
    value=json.loads(re.search(r"const motion=(.*?),parents=",html).group(1))
    assert np.array_equal(value["pose"],relative)
    assert "requestAnimationFrame(tick)" in html and "performance.now()" in html
    assert "setInterval" not in html and "%xyz.length" not in html
    assert 'if(frame>=last)playing=false' in html and 'translate="no"' in html
def test_old_parser_requires_fixed_root_and_valid_frame_shape():
    text="const xyz="+json.dumps(np.zeros((4,22,3)).tolist())+",parents=[]"
    assert old_points(text).shape==(4,22,3)
    with pytest.raises(ValueError):old_points(text.replace("const xyz=","other="))
