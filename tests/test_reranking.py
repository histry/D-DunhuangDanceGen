import copy,json
import torch
import pytest
from dunhuang.alignment.model import RetrievalModel
from dunhuang.alignment.rank import synchronized_scores,authorize
from dunhuang.io import sha

def test_old_test_qualification_is_not_silently_promoted(tmp_path):
    file=tmp_path/'weights';file.write_bytes(b'checkpoint')
    report=tmp_path/'report.json'
    report.write_text(json.dumps(dict(qualified=True,retrieval_checkpoint_sha256=sha(file),preference_status='UNKNOWN')))
    assert not authorize(file,report)['enabled']

def test_sync_encoding_identity_is_required():
    first=RetrievalModel().eval();second=copy.deepcopy(first)
    with torch.no_grad():second.motion.input.weight[0,0]+=1
    m=torch.zeros(1,30,50);d=torch.zeros(1,30,74)
    with pytest.raises(ValueError,match='identities'):
        synchronized_scores(first,second,m,torch.ones_like(m,dtype=torch.bool),d,torch.ones_like(d,dtype=torch.bool))

def test_qualification_binds_both_loaded_checkpoints(tmp_path):
    global_file=tmp_path/'global';global_file.write_bytes(b'global')
    sync=tmp_path/'sync';sync.write_bytes(b'sync')
    other=tmp_path/'other';other.write_bytes(b'other')
    report=tmp_path/'qualification.json'
    report.write_text(json.dumps(dict(qualified=True,qualification_basis='validation_only',
        retrieval_checkpoint_sha256=sha(global_file),sync_checkpoint_sha256=sha(sync),preference_status='UNKNOWN')))
    assert not authorize(global_file,report)['enabled']
    assert authorize(global_file,report,sync)['enabled']
    with pytest.raises(ValueError,match='Synchronization checkpoint'):
        authorize(global_file,report,other)
