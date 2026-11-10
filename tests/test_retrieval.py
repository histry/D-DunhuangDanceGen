import numpy as np
import pytest
from dunhuang.retrieval import retrieve
from dunhuang.io import sha

def test_catalog_hash_and_source_bounds(tmp_path):
    x=np.zeros((2,151),np.float32);p=tmp_path/'motion.npy';np.save(p,x)
    row=dict(id='a',source_frame=1,motion=dict(path=str(p),sha256=sha(p)),descriptor=[0.,1.])
    assert retrieve(dict(entries=[row]),query=[0.,1.])[0]['source_frame']==1
    row['source_frame']=2
    with pytest.raises(ValueError):retrieve(dict(entries=[row]))
    row['source_frame']=1;np.save(p,x+1)
    with pytest.raises(ValueError):retrieve(dict(entries=[row]))
