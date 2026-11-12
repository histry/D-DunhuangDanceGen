import copy
import numpy as np
import pytest
from dunhuang.research.source_tasks.math import response,noise_support

def motion():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],(150,24));return x

def test_response_separates_magnitude_from_direction_and_ignores_unknown_storage():
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import sixd
    a=motion();b=a.copy();b[:,79:127]=np.tile(sixd(Rotation.from_euler("x",.3).as_matrix()),(150,8))
    known=np.ones((150,8),bool);known[20:50]=False
    assert response(a,a,a,b,known)["normalized_response"]==0
    assert abs(response(a,b,a,b,known)["normalized_response"]-1)<1e-9
    b[20:50,79:127]=np.nan
    assert np.isfinite(response(a,a,a,b,known)["reference_separation_rad"])
    with pytest.raises(ValueError):response(a,b,a,b,np.zeros_like(known))

def test_noise_support_uses_total_gaussian_kl_and_excludes_pure_noise():
    a=np.zeros((150,151));b=np.ones_like(a)
    alpha=np.linspace(.999,1e-8,1000)
    cfg=dict(minimum_ddim_t=899,max_snr=.02,max_mean_kl=.5)
    times,proof=noise_support(a,b,alpha,cfg)
    assert len(times)==1 and times[0]==999 and proof["max_selected_kl"]<=.5
    times,_=noise_support(a,b*1e6,alpha,cfg);assert not len(times)
    alpha[-1]=0
    with pytest.raises(ValueError):noise_support(a,b,alpha,cfg)

def test_dunhuang_legacy_selection_is_rejected_before_silent_anchor_override(monkeypatch):
    from dunhuang.pipeline import run_once
    with pytest.raises(ValueError,match="Planner"):
        run_once(dict(profile="dunhuang",learned_retrieval=dict(checkpoint="nonexistent")))
    import sys,types
    monkeypatch.setitem(sys.modules,"soundfile",types.SimpleNamespace())
    with pytest.raises(KeyError,match="audio"):run_once(dict(profile="dunhuang"))

def test_cf_gradient_and_unknown_reference_storage():
    torch=pytest.importorskip("torch")
    from dunhuang.research.source_tasks.math import cf_loss
    from dunhuang.contracts import MotionAdapter
    ref=np.stack([motion(),motion()]);ref[1,:,79:127]=np.tile([1,0,0,0,0,1],(150,8))
    target=torch.from_numpy(np.stack([MotionAdapter.to_edge(r,np.zeros(3)) for r in ref]))
    predicted=(target.mean(0,keepdim=True).repeat(2,1,1)+.01*torch.randn_like(target)).requires_grad_()
    known=torch.ones(150,8,dtype=torch.bool);known[20:50]=False
    source=torch.from_numpy(ref);source[:,20:50,79:127]=float("nan")
    value,margin=cf_loss(predicted,source,known,torch.ones(151),torch.zeros(151),dict(cf_margin_fraction=.25,cf_margin_max_squared_rad=.01))
    value.backward()
    assert margin>0 and torch.isfinite(value) and torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum()>0

def test_rate_models_have_identical_capacity_and_real_seconds():
    torch=pytest.importorskip("torch")
    from dunhuang.research.source_tasks.content_model import ContentModel
    states=[];counts=[]
    torch.set_num_threads(1)
    for rate in ("15","30","dual"):
        torch.manual_seed(17);m=ContentModel(rate).eval();counts.append(sum(p.numel() for p in m.parameters()));states.append(m.state_dict())
        x=torch.randn(2,120,50);mask=torch.ones_like(x,dtype=torch.bool)
        with torch.no_grad():global_x,tokens,valid=m.music(x,mask)
        assert global_x.shape==(2,256) and tokens.shape==(2,120,128) and valid.all()
    assert len(set(counts))==1
    assert all(torch.equal(states[0][k],states[1][k]) and torch.equal(states[0][k],states[2][k]) for k in states[0])

def test_paired_content_delta_and_pessimistic_ties():
    pytest.importorskip("torch")
    from dunhuang.research.source_tasks.content import metrics
    rows=[dict(role="val",recording_uid=str(i),source_group=str(i%3)) for i in range(30)]
    c=dict(eligible=np.ones(30,bool),valid=np.ones((30,2),bool),positive=np.tile([True,False],(30,1)))
    scores=np.tile([2.,1.],(30,1));base=np.tile([0.,0.],(30,1))
    r=metrics(scores,base,c,rows,dict(bootstrap_seed=17,bootstrap_replicates=100))
    assert r["delta_R1"]==1 and r["delta_MRR"]==.5 and r["gain_lower_bounds_positive"]
