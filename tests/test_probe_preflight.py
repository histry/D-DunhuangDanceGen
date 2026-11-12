import json
import numpy as np
import pytest
from dunhuang.io import sha
from dunhuang.research.legacy.paired_training import train

def test_real_preflight_groups_by_recording_without_touching_cuda(tmp_path,monkeypatch):
    import torch
    def forbidden():raise AssertionError('CPU preflight touched CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    entries=[]
    for i in (1,2):
        motion=np.zeros((150,151),np.float32)
        motion[:,7:]=np.tile(np.array([1,0,0,0,1,0],np.float32),24)
        motion[:,4]=np.linspace(0,i*.1,150)
        path=tmp_path/f'source{i}.npy';np.save(path,motion)
        entries.append(dict(source_uid=f'alias{i}',recording_uid=f'recording{i}',
            reference=dict(path=str(path),sha256=sha(path))))
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(entries=entries)))
    scaler=tmp_path/'scaler.npz';np.savez(scaler,scale=np.ones(151,np.float32),bias=np.zeros(151,np.float32))
    assets=dict(catalog=dict(path=str(catalog),sha256=sha(catalog)),
        scaler=dict(path=str(scaler),sha256=sha(scaler)),checkpoint=dict(path='not-loaded-on-preflight',sha256='unknown'))
    out=tmp_path/'preflight'
    train(assets,out,300,'upperbody_probe',preflight_only=True)
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['train_groups']==['recording1'] and protocol['validation_groups']==['recording2']
    assert json.loads((out/'status.json').read_text())['training_steps']==0
    assert not (out/'adapter.pt').exists()
    # Relabeling identical source bytes as another recording must fail closed.
    entries[1]['reference']=entries[0]['reference']
    catalog.write_text(json.dumps(dict(entries=entries)))
    assets['catalog']['sha256']=sha(catalog)
    with pytest.raises(ValueError,match='Duplicate source'):
        train(assets,tmp_path/'leak',300,'upperbody_probe',preflight_only=True)
