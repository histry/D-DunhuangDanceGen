import sys
from pathlib import Path
import numpy as np
from dunhuang.retrieval import primitive_index
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from b3 import apply
from test_contracts import poses

def test_merge_only_overlapping_event_intervals():
    rows=[dict(source_uid='s',event='a',raw_center=t) for t in (20,21,80)]
    rows.append(dict(source_uid='s',event='b',raw_center=20))
    groups=primitive_index(rows)
    assert len(groups)==2
    assert [len(g['clusters']) for g in groups]==[2,1]
    assert groups[0]['clusters'][0]['centers']==[20,21]

def test_so3_edits_preserve_unowned_bytes_and_boundary():
    x=poses(40)
    delta=np.zeros((20,24,3));delta[:,17,0]=.04
    y=apply(x,10,30,delta,np.zeros((20,3)),.5)
    allowed=np.zeros_like(x,dtype=bool);allowed[11:29,7+17*6:13+17*6]=True
    assert x[~allowed].tobytes()==y[~allowed].tobytes()
    assert not np.array_equal(x[allowed],y[allowed])
