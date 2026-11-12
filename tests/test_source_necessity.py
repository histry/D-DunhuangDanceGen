import copy
import numpy as np
import pytest
from dunhuang.research.necessity.factors import ddim_times,timesteps,phase_mask,donors
from dunhuang.research.necessity.support import fit,build

def test_time_support_uses_matched_quantiles_and_exact_sampler_grid():
    u=np.array([0,.5,.999999])
    assert timesteps(u,False).tolist()==[50,500,949]
    assert timesteps(u,True).tolist()==[19,519,999]
    assert sorted(set(ddim_times())-set(range(50,950)))==[19,39,959,979,999]
    for bad in ([-.1],[1],[float("nan")]):
        with pytest.raises(ValueError):timesteps(bad,True)

def test_phase_supervision_retains_keys_and_excludes_hidden_frames():
    known=np.ones((150,8),bool);known[25:40]=False;known[90:115]=False
    keys=np.zeros_like(known);keys[75]=True
    phases=[dict(frame=130,type="upper_speed_peak"),dict(frame=150,type="upper_pose_hold"),dict(frame=160,type="root_turn_proxy")]
    m=phase_mask(phases,100,known,keys,2)
    assert not m[25:40].any() and m[48:53].all() and m[75].all()
    assert not m[58:63].any() and not (m&~known).any()

def test_corruption_preserves_exact_occurrences_and_forbids_known_associations():
    rows=[dict(source_group="g"+str(i//2),choreography_uid="c"+str(i//2),recording_uid="r"+str(i)) for i in range(6)]
    s=np.tile(np.arange(6),4).reshape(12,2);d=donors(s,rows,1)
    assert sorted(s.flat)==sorted(d.flat)
    assert all(rows[a]["source_group"]!=rows[b]["source_group"] and rows[a]["choreography_uid"]!=rows[b]["choreography_uid"] for a,b in zip(s.flat,d.flat))
    with pytest.raises(ValueError):donors(np.zeros((3,2),int),rows,1)

def test_train_support_fit_and_cutoff_ignore_val_outliers():
    rows=[dict(role="train" if i<3 else "val",music_uid="m"+str(i),choreography_uid="c"+str(i),
        window_uid="w"+str(i),source_group="g"+str(i)) for i in range(6)]
    z=np.arange(30,dtype=float).reshape(6,5)/30;mask=np.array([r["role"]=="train" for r in rows]);cfg=dict(donor_music_groups=2,per_music_cap=4,nearest_quantile=.9,radial_quantile=.99)
    m=fit(z,mask,.1);c,s,_=build(rows,np.zeros(6,int),z,m,cfg,"association",np.ones(6,bool))
    altered=z.copy();altered[3:]+=1e9
    n=fit(altered,mask,.1);d,t,_=build(rows,np.zeros(6,int),altered,n,cfg,"association",np.ones(6,bool))
    assert all(np.array_equal(m[k],n[k]) for k in m) and s["cutoff"]==t["cutoff"]
    assert np.array_equal(c["candidates"][:3],d["candidates"][:3])

def test_nuisance_scorer_does_not_read_query_target_motion():
    pytest.importorskip("torch")
    from dunhuang.research.necessity.a2 import cross_nuisance
    s=dict(counts=np.ones((3,2)),quantiles=np.ones((3,2,16)),hist=np.ones((3,2,4)),
        energy=np.ones((3,2)),onset_density=np.ones(3))
    first=cross_nuisance(s,0,1);other=copy.deepcopy(s)
    for name in ("counts","quantiles","hist","energy"):other[name][0,1]+=100
    assert np.array_equal(first,cross_nuisance(other,0,1))

def test_observed_null_centering_is_exact_identity_with_nonzero_output_weights():
    torch=pytest.importorskip("torch")
    from dunhuang.generation.event_music_adapter import EventMusicAdapter
    adapter=EventMusicAdapter().eval()
    with torch.no_grad():adapter.output.weight.normal_();adapter.output.bias.normal_()
    geometry=torch.randn(1,150,153);geometry[...,144:152]=1
    music=torch.randn(1,150,512)
    with torch.no_grad():
        a=adapter(geometry,music,visibility_policy="observed")
        null=adapter(torch.zeros_like(geometry),music,visibility_policy="observed")
    assert torch.count_nonzero(null)==0 and torch.equal(a-null,a)


def test_indexed_candidate_pools_equal_full_scan_for_both_registered_tasks():
    from dunhuang.research.necessity.support import candidate_pool
    rows=[dict(role="train" if i<12 else "val",music_uid="m"+str(i%3),choreography_uid="c"+str(i%4),window_uid="w"+str(i)) for i in range(24)]
    ticks=np.array([i//4*1470 for i in range(24)]);cfg=dict(min_time_difference_frames=1,donor_music_groups=3,per_music_cap=16)
    for task in ("time","association"):
        for i,r in enumerate(rows):
            pool=[j for j,v in enumerate(rows) if v["role"]==r["role"] and (v["music_uid"]==r["music_uid"] and v["choreography_uid"]==r["choreography_uid"] if task=="time" else ticks[j]==ticks[i])]
            assert candidate_pool(rows,ticks,i,task,cfg)==candidate_pool(rows,ticks,i,task,cfg,pool)

def test_hidden_only_source_difference_cannot_qualify_reference_pair():
    pytest.importorskip("torch")
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import sixd
    from dunhuang.research.necessity.factors import visible_source_distance
    a=np.zeros((150,151),np.float32);a[:,7:]=np.tile([1,0,0,0,1,0],(150,24));b=a.copy()
    known=np.ones((150,8),bool);known[25:40]=False;known[90:115]=False
    for j in range(8):
        b[~known[:,j],79+6*j:85+6*j]=sixd(Rotation.from_euler("x",1).as_matrix())
    assert visible_source_distance(a,b,known,known)<1e-8
    assert visible_source_distance(a,b,np.ones_like(known),np.ones_like(known))>.05
