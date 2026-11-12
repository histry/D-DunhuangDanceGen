import copy
import numpy as np
import pytest
from dunhuang.research.transfer_audit.contracts import validate_song_task,Moments,training_gate

def rows():
    return [dict(role=role,music_uid="m"+str(i),recording_uid="r"+str(i),source_group="g"+str(i),
        raw_motion={"sha256":"h"+str(i)},audio={"sha256":"a"+str(i)},choreography_uid="shared_choreography") for i,role in enumerate(("train","val"))]

def test_unseen_song_closure_allows_shared_choreography_without_alias_leakage():
    r=rows();validate_song_task(r)
    for key in ("music_uid","recording_uid","source_group"):
        bad=copy.deepcopy(r);bad[1][key]=bad[0][key]
        with pytest.raises(ValueError):validate_song_task(bad)
    for key in ("raw_motion","audio"):
        bad=copy.deepcopy(r);bad[1][key]=bad[0][key]
        with pytest.raises(ValueError):validate_song_task(bad)

def test_test_tensor_rows_fail_before_feature_read():
    r=rows();r[1]["role"]="test"
    with pytest.raises(ValueError):validate_song_task(r)

def test_train_statistics_ignore_val_outlier_and_unobserved_nonfinite_values():
    n=Moments(2);n.add(np.array([[1.,float("nan")],[3,float("inf")]]),np.array([[True,False],[True,False]]),"train")
    n.add(np.full((2,2),1e8),np.ones((2,2),bool),"val");r=n.result()
    assert r["mean"]==[2,0] and r["std"]==[1,1] and r["observed_count"]==[2,0]
    with pytest.raises(ValueError):n.add(np.array([[float("nan"),0]]),np.ones((1,2),bool),"train")

def test_population_does_not_certify_aliases_or_new_weights():
    p={r:dict(population_sufficient=True) for r in ("train","val")}
    g=training_gate(p,dict(ready_for_new_training=False),"future_binding")
    assert not g["ready"] and not g["checks"]["work_aliases_certified"]
    task=dict(ready_for_new_training=True,composition_aliases_outside_publisher_ids="VERIFIED",pretrained_backbone_exposure="VERIFIED")
    assert training_gate(p,task,"new_checkpoint")["ready"]
    assert not training_gate(p,task,None)["ready"]

def test_historical_default_candidate_validation_stays_strict():
    pytest.importorskip("torch")
    from dunhuang.research.source_identifiability.a2 import candidates
    with pytest.raises(ValueError,match="crosses roles"):candidates(rows(),{}, {}, {})
