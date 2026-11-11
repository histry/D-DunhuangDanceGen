import copy
import numpy as np
import pytest
from dunhuang.generation.source_pairs import source_pairs, validate_pair
from dunhuang.research.event_metrics import identity_probe, representation_stats, event_margin
from dunhuang.contracts import sixd
from scipy.spatial.transform import Rotation


def row(uid, recording, role='train'):
    return dict(window_uid=uid, role=role, recording_uid='aist_capture:gXX_sBM_'+recording,
                choreography_uid='routine1', music_uid='song', start_frame=0, fps=30,frames=150,
                audio_offset_seconds=0, audio={'sha256':'music'},raw_motion={'sha256':recording},dataset='aistpp')


def test_independent_source_task_never_crosses_roles_or_fabricates_exact_pose():
    a,b,c=row('a','r1'),row('b','r2'),row('c','r3','val')
    pairs=source_pairs({'rows':[a,b,c]})
    assert len(pairs)==2
    assert all(not p['pose_exact_correspondence'] for p in pairs)
    assert pairs[0]['target_uid']!=pairs[0]['source_uid']
    by_id={r['window_uid']:r for r in [a,b,c]}
    bad=dict(pairs[0],source_uid='c')
    with pytest.raises(ValueError,match='role'):
        validate_pair(bad,by_id)
    with pytest.raises(ValueError,match='Train/Val'):
        source_pairs({'rows':[a,b,c]},roles=('test',))
    duplicate=copy.deepcopy(b);duplicate['raw_motion']=a['raw_motion']
    assert source_pairs({'rows':[a,duplicate]})==[]


def test_correspondence_requires_exact_music_and_time():
    a,b=row('a','r1'),row('b','r2')
    p=source_pairs({'rows':[a,b]})[0]
    b['start_frame']=75
    with pytest.raises(ValueError,match='start_frame'):
        validate_pair(p,{'a':a,'b':b})
    assert not source_pairs({'rows':[a,b]})


def test_closed_set_probe_detects_identity_and_constant_feature_chance():
    labels=np.repeat(np.arange(4),3);views=np.tile(np.arange(3),4)
    features=np.eye(4)[labels]
    assert identity_probe(features,labels,views)['accuracy']==1
    assert identity_probe(np.zeros_like(features),labels,views)['accuracy']==.25
    assert representation_stats(np.ones((12,5)))['effective_rank']==0


def test_pose_preference_is_not_merely_large_output_change():
    a=np.zeros((150,151),np.float32)
    a[:,7:]=np.tile(sixd(np.eye(3)).reshape(1,6),(150,24))
    b=a.copy()
    r=Rotation.from_euler('x',.7).as_matrix()
    b[:,79:127]=np.tile(sixd(r),(150,8))
    assert event_margin(a,b,a,b)['preference_margin_radians']>.69
    assert abs(event_margin(a,a,a,b)['preference_margin_radians'])<1e-7
    assert event_margin(b,a,a,b)['preference_margin_radians']<-.69
