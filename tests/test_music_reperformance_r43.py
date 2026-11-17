import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.music_reperformance_r43.contract import permissions,UPPER


def fixture():
    from dunhuang.research import music_reperformance_r43
    cfg=json.loads((Path(music_reperformance_r43.__file__).parent/'config.json').read_text())
    rng=np.random.default_rng(43);R=Rotation.from_rotvec(rng.normal(scale=.1,size=(256*22,3))).as_matrix().reshape(256,22,3,3)
    return cfg,R


def test_permissions_never_use_hidden_motion_or_its_derivatives():
    cfg,R=fixture();a=permissions(R,cfg);R[104:152]=np.nan;b=permissions(R,cfg)
    assert np.array_equal(a.body_budget,b.body_budget)
    assert np.array_equal(a.phase_proxy,b.phase_proxy) and a.anchors==b.anchors
    R[103]=np.nan
    with pytest.raises(ValueError):permissions(R,cfg)


def test_real_core_permissions_protect_peaks_lower_body_and_outer_context():
    cfg,R=fixture();c=permissions(R,cfg);core=np.r_[0:104,152:256];lower=[i for i in range(22) if i not in UPPER]
    assert not c.body_budget[np.ix_(core,lower)].any()
    assert (c.body_budget[32:80,UPPER]>0).any() and c.body_budget[core].max()<=.1+1e-7
    assert c.body_budget[104:152].max()<=.06+1e-7
    for p in c.anchors:assert not c.body_budget[p['frame']-2:p['frame']+3].any()
    assert c.protected_frames[:8].all() and c.protected_frames[-8:].all()
    assert not c.root_envelope[core].any()


def test_regional_zero_residual_exact_and_large_proposal_bounded():
    torch=pytest.importorskip('torch');pytest.importorskip('pytorch3d')
    from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle
    from dunhuang.research.music_reperformance_r43.projector import RegionalProjector
    from types import SimpleNamespace
    cfg,R=fixture();c=permissions(R,cfg);a=torch.zeros(2,256,139);a[:,:,7:]=torch.tensor(R[...,:2,:].reshape(256,132),dtype=a.dtype)
    rt=SimpleNamespace(n=SimpleNamespace(legacy=SimpleNamespace(scaler=SimpleNamespace(scale_=torch.ones(139)))))
    projector=RegionalProjector(c,.03);assert torch.equal(projector(rt,a,a,None),a)
    p=torch.randn_like(a);out=projector(rt,p,a,None);A=rotation_6d_to_matrix(a[:,:,7:].reshape(2,256,22,6));B=rotation_6d_to_matrix(out[:,:,7:].reshape(2,256,22,6));angle=matrix_to_axis_angle(A.transpose(-1,-2)@B).norm(dim=-1)
    assert (angle<=torch.tensor(c.body_budget)[None]+2e-6).all()
    assert torch.equal(out[:,c.protected_frames],a[:,c.protected_frames])
    assert ((out[:,:,4:7]-a[:,:,4:7]).norm(dim=-1)<=torch.tensor(c.root_envelope)[None]*.03+1e-6).all()
    assert torch.equal(out[:,:,:4],a[:,:,:4])


def test_viewer_uses_exact_case_ids_when_legacy_new_plan_prefix_collides():
    from dunhuang.research.music_reperformance_r43.view import selected_cases
    cases=[dict(id=k) for k in ['old_failed','new_plan','new_new_plan','new_paired_0']]
    assert [r['id'] for r in selected_cases(cases,['new_plan','paired_0'])]==['new_new_plan','new_paired_0']
