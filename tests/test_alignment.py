import torch
import pytest
from dunhuang.alignment.model import Encoder,contrastive
from dunhuang.alignment.semantics import annotate

def test_filename_labels_keep_recording_scope():
    a=annotate('female_FeiTian.npz')
    assert a['category']=='飞天'
    assert a['event_specific_semantics']=='UNKNOWN'
    assert not a['music_semantic_pair_label']

def test_missing_values_cannot_affect_encoder():
    torch.manual_seed(4)
    model=Encoder(50).eval()
    x=torch.randn(2,60,50);mask=torch.ones_like(x,dtype=torch.bool)
    mask[:,30:]=False
    altered=x.clone();altered[:,30:]=100
    with torch.no_grad():
        first=model(x,mask)[0];second=model(altered,mask)[0]
    torch.testing.assert_close(first,second)
    with pytest.raises(ValueError):model(x,torch.zeros_like(mask))

def test_related_groups_are_not_false_negatives():
    m=torch.eye(3)
    assert torch.isfinite(contrastive(m,m,['a','a','b']))
