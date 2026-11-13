import copy
from types import SimpleNamespace
import numpy as np
import pytest
from test_primitive_reperformance import fixture,CFG

def test_r9_relative_root_leaves_initial_orientation_and_root_translation_unmodified(tmp_path):
    torch=pytest.importorskip("torch");pytest.importorskip("pytorch3d")
    import json
    from dunhuang.io import sha
    from dunhuang.research.feasible_reperformance.guide import prepare
    p=fixture(tmp_path);file=tmp_path/"plan.json";file.write_text(json.dumps(p));folder=tmp_path/"out";folder.mkdir()
    config=copy.deepcopy(CFG);config["groups"]["T2R"]=[0]+list(range(12,22))
    q=dict(event_plan=dict(path=str(file),sha256=sha(file)),frames=104,windows=1,audio={"sha256":"audio"},
        condition_group="T2R",r9_relative_root=True,reperformance_config=config)
    normalizer=SimpleNamespace(scaler=SimpleNamespace(scale_=torch.ones(151),min_=torch.zeros(151)))
    guide=prepare(q,normalizer,folder,"cpu")
    x=torch.zeros(1,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24);x[:,:,4:7]=.4
    y=guide(x,.9)
    assert torch.equal(y[0,0,7:13],x[0,0,7:13]) and torch.equal(y[:,:,4:7],x[:,:,4:7])
    assert not torch.equal(y[0,20:80,7:13],x[0,20:80,7:13])
