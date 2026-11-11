import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.planning.time_map import source_positions,motion_features74
from dunhuang.generation.soft_reference import resample_native

def test_piecewise_mapping_drives_both_pose_and_rhythm():
    source=np.zeros((151,151),np.float32)
    source[:,7:]=np.tile([1,0,0,0,1,0],24)
    source[:,79:85]=sixd(Rotation.from_euler('z',np.arange(151)*.003).as_matrix())
    event=dict(source_interval=[0,150],music_interval=[0,150],time_warp=[[0,0],[65,75],[150,150]])
    times=source_positions(event,np.arange(150))
    assert times[75]==65
    values,mask=motion_features74(resample_native(source,times))
    assert values[120,12*3+2]>values[30,12*3+2]
    assert not mask[:,[68,69,71]].any() and not values[:,[68,69,71]].any()

def test_candidate_identity_binds_actual_execution_map_and_model():
    from dunhuang.planning.event_plan import candidate_identity
    event=dict(event_uid='event',source_interval=[0,150],music_interval=[0,150],
        time_warp=[[0,0],[150,150]],reference=dict(path='source.npy',sha256='source'))
    first=candidate_identity(event,dict(checkpoint='a'))
    event['time_warp']=[[0,0],[65,75],[150,150]]
    assert candidate_identity(event,dict(checkpoint='a'))!=first
    assert candidate_identity(event,dict(checkpoint='a'))!=candidate_identity(event,dict(checkpoint='b'))
