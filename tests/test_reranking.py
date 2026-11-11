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
