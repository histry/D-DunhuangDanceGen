import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.phase_r45.clock import matched_maps,feasible_bank,choose,stages

CFG=json.loads((Path(__file__).parents[1]/'src/dunhuang/research/phase_r45/config.json').read_text())

def example():
    raw=np.zeros((256,139));R=Rotation.from_rotvec(np.c_[np.sin(np.arange(256)/11)*.15,np.zeros((256,2))]).as_matrix();raw[:,7:]=sixd(np.repeat(R[:,None],22,axis=1),row=True).reshape(256,132);parents=np.r_[-1,np.arange(21)];J=np.c_[np.zeros(22),np.arange(22)*.03,np.zeros(22)];return raw,J,parents

def test_factor_maps_share_every_anchor_and_support():
    rows=[dict(interval=[8,99],source=[8,40,70,98],target=[8,43,67,98])];linear,smooth=matched_maps(rows)
    for s,t in zip(rows[0]['source'],rows[0]['target']):assert linear[t]==smooth[t]==s
    assert np.array_equal(smooth[99:],np.arange(99,256));assert np.all(np.diff(linear)>0) and np.all(np.diff(smooth)>0);assert not np.array_equal(linear,smooth)

def test_source_only_bank_and_no_music_identity():
    raw,J,parents=example();bank,phases,_=feasible_bank(raw,J,parents,8,99,CFG);assert len(bank)<=25
    chosen=choose(bank,phases,None,CFG);assert chosen['identity'];assert all(r['certificate']['legal'] for r in bank)
    for r in bank:
        if r['eligible']:assert .85<=r['ratios']['D1']<=1.15 and r['ratios']['D2']<=1.05 and r['ratios']['D3']<=1.05

def test_phase_extraction_does_not_read_hidden_bridge():
    raw,J,parents=example();a=stages(raw,J,parents,8,99,CFG);raw[104:152,7:]=raw[90,7:];assert stages(raw,J,parents,8,99,CFG)==a

def test_quiet_source_has_no_invented_phase():
    raw,J,parents=example();raw[:,7:]=raw[0,7:];phases,evidence=stages(raw,J,parents,8,99,CFG);assert phases==[] and evidence['status']=='LOW_ACTIVITY_NO_FORCED_PHASE'
