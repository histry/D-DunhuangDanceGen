"""R67 selection wiring regression; synthetic scores do not qualify real motion."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.sdmr_warp_rank import study as s


def test_real_readout_vector_preserves_original_bank_indices_and_all_ties():
    # encode_bank returns one float/None per candidate, not (index, score) pairs.
    scores=[0.4,None,0.8,0.8]
    connections=[{'passes_original_screen':True},{'passes_original_screen':True},
                 {'passes_original_screen':False},{'passes_original_screen':True}]
    assert s.rank_indices(scores)==[2,3]
    assert s.screened_rank_indices(scores,connections)==[3]
    assert s.screened_rank_indices([.4,.8,.8],[{'passes_original_screen':True}]*3)==[1,2]


def test_missing_scores_or_no_verified_connection_abstain():
    assert s.rank_indices([None,None])==[]
    assert s.screened_rank_indices([.4,.8],[{}, {'passes_original_screen':False}])==[]
    assert s.screened_rank_indices([.4],[{'passes_original_screen':1}])==[]
    assert s.screened_rank_indices([],[])==[]


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),True,'0.8'])
def test_invalid_score_is_rejected_even_on_screened_out_candidate(bad):
    with pytest.raises(ValueError,match='finite numeric'):
        s.rank_indices([.4,bad])
    with pytest.raises(ValueError,match='finite numeric'):
        s.screened_rank_indices([.4,bad],[{},{}])


def test_candidate_bank_length_mismatch_is_rejected():
    with pytest.raises(ValueError,match='same candidate bank'):
        s.screened_rank_indices([.1,.2],[{'passes_original_screen':True}])


def test_run_main_writes_raw_and_g1_screened_selections_from_actual_score_vectors(monkeypatch,tmp_path):
    baseline=np.broadcast_to(np.eye(3),(240,22,3,3)).copy()
    other=baseline.copy()
    other[120,0]=Rotation.from_rotvec([.02,0,0]).as_matrix()
    J=np.zeros((22,3));parents=np.arange(-1,21)
    q0=np.arange(240,dtype=float);q1=q0+.01
    candidates=[dict(uid='candidate0',q=q0.tolist(),q_sha256='q0',time_warp=[[0,0],[239,239]]),
                dict(uid='candidate1',q=q1.tolist(),q_sha256='q1',time_warp=[[0,0],[239,239]])]
    case=dict(key='case',target_interval=[0,240],songidx=0,audio_sha256='A',source_uid='S',
              source_interval=[0,240],baseline_core_R_sha256=s.array_hash(baseline))
    ctx=dict(change_cases=[case],J=J,parents=parents,lib=SimpleNamespace(rotations={'S':baseline}),counters={})
    monkeypatch.setattr(s.connectivity,'prepare',lambda:ctx)
    monkeypatch.setattr(s.candidates,'bank',lambda *a:dict(candidates=candidates))
    class FakeEncoder:
        joints=J
        before='unchanged'
        binding={}
        def encode_music_raw(self,*a):return 'music'
        def check(self):pass
        def weight_digest(self):return self.before
    FakeEncoder.parents=parents
    monkeypatch.setattr(s.encoder,'FrozenEncoder',FakeEncoder)
    monkeypatch.setattr(s,'sha',lambda p:s.CFG['change_raw50_feature_arrays_sha256'] if str(p).endswith('arrays.npz') else s.CFG['r64_music_clocks_sha256'])
    inventory=[dict(case=0,audio=dict(sha256='A'),clock=dict(frames=240))]
    clocks={'0':dict(audio_sha256='A')}
    monkeypatch.setattr(s,'read',lambda p:inventory if str(p).endswith('inventory.json') else clocks)
    monkeypatch.setattr(s.np,'load',lambda p:{'0__music50':np.zeros((240,50)),'0__music_mask':np.ones((240,50),bool)})
    monkeypatch.setattr(s,'rotation_sample',lambda src,q:baseline.copy() if q[0]==0 else other.copy())
    scores={'actual':{'Single64':[.4,.8],'Multi64':[.8,.4],'RawDiagonal':[.5,.5]}}
    monkeypatch.setattr(s,'encode_bank',lambda *a:([(None,None,{}),(None,None,{})],np.ones(228,bool),{},scores))
    calls=[]
    def check(case,body,ctx):
        calls.append(body.copy())
        return dict(passes_original_screen=len(calls)==1),{}
    monkeypatch.setattr(s.connectivity,'check',check)
    monkeypatch.setattr(s.body_metrics,'measure',lambda *a,**k:{})
    monkeypatch.setattr(s,'binding',lambda:{'test_fixture':True})
    monkeypatch.setattr(s,'save_arrays',lambda *a:None)
    monkeypatch.setattr(s,'seal',lambda *a:None)
    from dunhuang.research.sdmr_warp_rank import viewer
    shown=[]
    monkeypatch.setattr(viewer,'build',lambda target,display,*a:shown.extend(display))
    out=tmp_path/'fresh'
    s.run_main(out)
    selection=json.loads((out/'selections/case.json').read_text(encoding='utf8'))
    assert selection['raw_score_ties']=={'Single64':[1],'Multi64':[0],'RawDiagonal':[0,1]}
    assert selection['G1_screened_score_ties']=={'Single64':[0],'Multi64':[0],'RawDiagonal':[0]}
    assert selection['abstain_when_empty'] is True
    assert len(calls)==2
    # Raw display remains transparent about selection; G1 does not silently reroute it.
    assert shown[0][1][2]['row']['candidate_uid']=='candidate1'
    assert shown[0][1][2]['row']['connectivity']['passes_original_screen'] is False
