import numpy as np
import pytest
from dunhuang.planning.event_plan import validate
from dunhuang.planning.planner import plan
from dunhuang.io import sha

def test_complete_plan_and_source_identity(tmp_path):
    file=tmp_path/'motion.npy';np.save(file,np.ones((500,151),np.float32))
    entries=[]
    for i in range(3):
        entries.append(dict(event_uid=str(i),source_uid=str(i),source_interval=[0,90],center=45,
            duration_frames=90,reference=dict(path=str(file),sha256=sha(file)),activity_percentile=i/3,
            entry=np.tile(np.eye(3),(8,1,1)).tolist(),exit=np.tile(np.eye(3),(8,1,1)).tolist(),
            feasibility=dict(temporal8='PASS',native='UNKNOWN')))
    timing=dict(frames=270,slots=[dict(start=a,end=a+90,density=.5) for a in (0,90,180)],beats=[])
    result=plan(dict(entries=entries),timing,'audio')
    assert result['events'][0]['music_interval']==[0,90]
    assert result['events'][-1]['music_interval']==[180,270]
    assert len(set(e['source_uid'] for e in result['events']))>=2
    result['events'][1]['music_interval'][0]+=1
    with pytest.raises(ValueError):validate(result,'audio',270)
