"""Mock weights prove API/normalization contracts, not actual learned quality."""
import numpy as np
import pytest
import torch

from dunhuang.research.sdmr_warp_rank import encoder as e
from dunhuang.research.event_alignment_r49.common import Frozen
from dunhuang.research.paired_event_r41.model import EventContent


class FixtureFrozen(Frozen):
    def __init__(self):
        torch.manual_seed(67067)
        self.device='cpu'
        self.net=EventContent(96).cpu().eval().requires_grad_(False)
        self.joints=np.zeros((22,3));self.parents=np.arange(-1,21)
        self.normalizer={'music':dict(mean=np.full(50,2.),std=np.full(50,2.))}
        self.normalize_calls=0;self.motion_calls=0
    def binding(self):
        return dict(checkpoint_sha256=e.CHECKPOINT_SHA,normalizer_sha256=e.NORMALIZER_SHA,device='cpu')
    def normalize(self,values,mask,name):
        self.normalize_calls+=1
        return super().normalize(values,mask,name)
    def motion(self,rotations):
        self.motion_calls+=1
        values=np.ones((len(rotations),274),np.float32)
        values[:,-2:]=1000
        mask=np.ones_like(values,bool);mask[:,[266,267,269]]=False;mask[0,198:272]=False
        return np.where(mask,values,0),mask
    def motion_tokens(self,values,mask):
        self.seen_values=values.copy();self.seen_mask=mask.copy()
        return super().motion_tokens(values,mask)


@pytest.fixture
def fixture():
    f=FixtureFrozen()
    return f,e.FrozenEncoder(_fixture=f)


def test_prenormalized_r59_path_never_normalizes_again_and_matches_raw_once(fixture):
    f,enc=fixture
    normalized=np.linspace(-1,1,240*50,dtype=np.float32).reshape(240,50)
    valid=np.ones(240,bool);valid[100]=False
    a,av,ap=enc.encode_music_normalized(normalized,valid)
    assert f.normalize_calls==0 and ap['additional_normalizations']==0
    raw=normalized*2+2;mask=np.broadcast_to(valid[:,None],raw.shape).copy()
    # Invalid normalized values must carry the same zeroed content as raw mask.
    normalized[~valid]=0
    a,av,ap=enc.encode_music_normalized(normalized,valid)
    b,bv,bp=enc.encode_music_raw(raw,mask)
    assert f.normalize_calls==1 and bp['additional_normalizations']==1
    assert np.allclose(a,b,atol=3e-7,rtol=0) and np.array_equal(av,bv)
    assert ap['token_frame_centers']==list(range(6,234)) and a.shape==(228,96)
    assert ap['model_digest_before']==ap['model_digest_after'] and ap['test_fixture']


def test_rotations_rederive_and_mask_clock_metadata_and_unknown_first_dynamic(fixture):
    f,enc=fixture
    rotations=np.broadcast_to(np.eye(3),(240,22,3,3)).copy()
    before=rotations.copy()
    tokens,valid,proof=enc.encode_rotations(rotations)
    assert f.motion_calls==1 and np.array_equal(rotations,before)
    assert not f.seen_mask[:,-2:].any() and not f.seen_values[:,-2:].any()
    assert not f.seen_mask[0,198:272].any() and not f.seen_values[0,198:272].any()
    assert not proof['clock_duration_channels_observed'] and not proof['first_frame_dynamics_observed']
    assert tokens.shape==(228,96) and valid.shape==(228,)
    assert proof['training_updates']==proof['optimizer_calls']==0


def test_weight_mutation_is_rejected_on_next_encode(fixture):
    f,enc=fixture
    with torch.no_grad():next(f.net.parameters()).view(-1)[0].add_(1)
    with pytest.raises(ValueError,match='weights changed'):
        enc.encode_music_normalized(np.zeros((240,50),np.float32),np.ones(240,bool))


def test_invalid_rotations_do_not_reach_observation_reconstruction(fixture):
    f,enc=fixture
    rotations=np.broadcast_to(np.eye(3),(240,22,3,3)).copy();rotations[0,0,0,0]=2
    with pytest.raises(ValueError,match='Proper SO'):
        enc.encode_rotations(rotations)
    assert f.motion_calls==0
