import numpy as np
import pytest
from dunhuang.research.music_events.catalog import event,agreement,reward

def test_seconds_remain_exact_when_nearest_frame_is_outside_support():
    r=event(.999,'beat','fixture',1000,1000,0)
    assert r['seconds']==.999 and r['seconds_supported'] and not r['nearest_frame_supported'] and r['nearest_frame']==30
    a=event(.100,'beat','x',1000,1000,0);b=event(.101,'beat','x',1000,1000,1)
    assert a['nearest_frame']==b['nearest_frame'] and a['seconds']!=b['seconds'] and a['uid']!=b['uid']
    assert event(1.,'section_boundary','x',1000,1000,2)['seconds_supported'] is False

def test_model_agreement_is_one_to_one_not_nearest_many_to_one():
    x=agreement([.1,.12],[.11],.07)
    assert x['one_to_one_matches']==1 and x['matched_fraction_a']==.5 and x['interpretation']=='MODEL_AGREEMENT_NOT_ACCURACY'
    assert agreement([],[])['symmetric_matched_fraction'] is None
    with pytest.raises(ValueError):agreement([.2,.1],[.1])

def test_nonperiodic_clock_control_and_shift_direction():
    times=[.8,2.1,3.5]
    assert reward(times,times)==1
    assert all(reward(times,times,d)<1 for d in [-.5,-.25,.25,.5])
    assert reward([],times) is None

def test_legacy_attention_alias_preserves_bias_and_kernel_arguments(monkeypatch):
    import types,sys
    from dunhuang.research.music_events.compat import install
    calls=[];f=types.ModuleType('natten.functional');pkg=types.ModuleType('natten');pkg.functional=f
    def qk(*args,**kwargs):calls.append((args,kwargs));return 'fixture'
    f.na1d_qk=f.na2d_qk=qk;f.na1d_av=f.na2d_av=qk
    monkeypatch.setitem(sys.modules,'natten',pkg);monkeypatch.setitem(sys.modules,'natten.functional',f)
    install();assert f.natten1dqkrpb('q','k','bias',7,2)=='fixture'
    assert calls[-1]==(('q','k'),{'kernel_size':7,'dilation':2,'rpb':'bias'})
