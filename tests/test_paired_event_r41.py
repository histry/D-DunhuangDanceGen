import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.paired_event_r41.events import profiles,segment,clock_relations,validate_roles

CFG=dict(boundary_smoothing_sigma_frames=1.5,minimum_core_frames=90,maximum_core_frames=360,preferred_core_frames=180,boundary_grid_frames=6)

def test_long_hold_is_not_split_for_a_length_cap():
    p={k:np.zeros(700) for k in ('global','upper','lower','trunk')}
    spans,evidence=segment(p,[dict(type='pose_hold',evidence_span=[0,700])],CFG)
    assert spans==[[0,700]]
    assert evidence['protected_spans']==[[0,700]]

def test_partition_covers_all_real_frames_and_protects_turns():
    v=np.ones(720);v[150:175]=0;v[340:365]=0;v[530:555]=0;p={k:v for k in ('global','upper','lower','trunk')}
    result,_=segment(p,[dict(type='root_turn_span',evidence_span=[160,330])],CFG)
    assert result[0][0]==0 and result[-1][1]==720
    assert all(a[1]==b[0] for a,b in zip(result,result[1:]))
    assert all(not 160<a<330 for a,b in result)

def test_whole_record_derivative_is_not_restarted_at_event_boundaries():
    rv=np.zeros((200,22,3));rv[:,:,2]=np.arange(200)[:,None]**2*.0001
    R=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(200,22,3,3)
    full=profiles(R)['omega'];local=profiles(R[90:150])['omega']
    assert not np.allclose(full[90],local[0])
    np.testing.assert_allclose(full[91:150],local[1:],atol=1e-9)

def row(start,end,role='train',raw='r',audio='a',group='g'):
    return dict(audio_frame_interval=[start,end],role=role,raw_motion_sha256=raw,audio_sha256=audio,source_group=group)

def test_clock_overlap_unknown_and_hard_negative_are_distinct():
    r=[row(0,120),row(10,130,raw='s'),row(80,200,raw='t'),row(140,260,raw='u')]
    p,n=clock_relations(r)
    assert p[0,1] and not n[0,1]
    assert not p[0,2] and not n[0,2]
    assert n[0,3] and not p[0,3]

def test_source_audio_and_music_group_cannot_cross_roles():
    for key in ('raw_motion_sha256','audio_sha256','source_group'):
        a=row(0,120);b=row(150,270,role='val',raw='new',audio='other',group='different');b[key]=a[key]
        with pytest.raises(ValueError,match='crosses roles'):validate_roles([a,b])
    with pytest.raises(ValueError,match='Test tensors'):validate_roles([row(0,120,role='test')])
