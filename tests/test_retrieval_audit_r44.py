import numpy as np
import pytest
pytest.importorskip("numba", reason="R44 optional CPU audit extra")
from dunhuang.research.sdmr_event_rank.scorer import batch_partial
from dunhuang.research.retrieval_audit_r44.path import partial

@pytest.mark.parametrize('seed',[0,7,41])
def test_fast_path_matches_frozen_recurrence_including_missing_tokens(seed):
    rng=np.random.default_rng(seed);n=37
    def token():
        a=rng.normal(size=(n,12)).astype('float32');a/=np.linalg.norm(a,axis=-1,keepdims=True);return dict(mt=a,dt=a.copy(),mv=rng.random(n)>.2,dv=rng.random(n)>.1,positions=np.arange(n))
    q=token();cs=[token() for _ in range(3)];cfg=dict(token_stride=3,path_band_frames=15,path_drop_cost=.3,path_null_penalty=.3,path_warp_penalty=.05);expected=batch_partial(q,cs,cfg)
    for i,c in enumerate(cs):np.testing.assert_allclose(partial(q,c,cfg),[v[i] for v in expected],atol=1e-12,rtol=0)

def test_all_null_and_exact_match_threshold():
    q=dict(mt=np.eye(9,dtype='float32'),dt=np.eye(9,dtype='float32'),mv=np.ones(9,bool),dv=np.ones(9,bool),positions=np.arange(9));c=dict(q,dt=-q['mt']);cfg=dict(token_stride=1,path_band_frames=1,path_drop_cost=.3,path_null_penalty=.3,path_warp_penalty=.05);assert partial(q,c,cfg)[1]==0;assert partial(q,q,cfg)[1]==1
