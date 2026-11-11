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


def test_song_identity_does_not_hide_local_time_errors():
    import numpy as np
    from dunhuang.alignment.evaluate import retrieval
    class Fixed:
        def music(self,x,mask):return torch.tensor([[1.,0.],[1.,0.]]),None,None
        def motion(self,x,mask):return torch.tensor([[0.,1.],[1.,0.]]),None,None
    rows=[dict(domain='finedance',audio_sha256='same',group='recording',start_frame=t) for t in (0,75)]
    arrays={k:np.ones((2,150,2),dtype=bool if k in ('mm','dm') else np.float32) for k in ('music','motion','mm','dm')}
    assert retrieval(Fixed(),rows,arrays,[0,1],'cpu')['finedance']['recall1']==1
    local=retrieval(Fixed(),rows,arrays,[0,1],'cpu',task='local')['finedance']
    assert local['recall1']==.5 and local['chance']==.5
