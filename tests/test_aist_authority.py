import numpy as np
import pytest
from dunhuang.data.aist_authority import align_samples,ncc_valid,identify,NAME

def test_waveform_delay_with_repeated_pulses_and_nonzero_source_offset():
    rng=np.random.default_rng(2);x=rng.normal(size=12*8000);y=x[2*8000:9*8000]
    r=align_samples(x,y)
    assert r['verified'] and r['audio_offset_seconds']==2 and r['minimum_ncc']>.99999

def test_wrong_audio_and_clock_stretch_do_not_calibrate():
    rng=np.random.default_rng(3);x=rng.normal(size=12*8000);y=rng.normal(size=7*8000)
    assert not align_samples(x,y)['verified']
    # Anchor residuals reveal a wrong rate even when a short crop looks close.
    source=rng.normal(size=12*8000);y=np.interp(np.arange(7*8000)*1.02,np.arange(len(source)),source)
    assert not align_samples(source,y)['verified']

def test_repeated_music_prefix_needs_official_origin_and_reports_ambiguity():
    rng=np.random.default_rng(5);loop=rng.normal(size=4*8000)
    x=np.tile(loop,4);y=x[:8*8000]
    r=align_samples(x,y)
    assert r['verified'] and r['origin_anchor_used'] and r['periodic_alternatives']

def test_official_basic_identity_preserves_capture_and_shared_choreography():
    n='gBR_sBM_cAll_d04_mBR0_ch01'
    url='https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'+n.replace('cAll','c01')+'.mp4'
    auth=dict(all={n},mapping={n:'setting1'},table={('gBR','ch01'):'indian step'},urls={url})
    row=dict(sequence_id=n,group_uid='legacy',purpose_role='train')
    r=identify(row,auth);assert r['recording_uid'].endswith(n) and r['choreography_uid']=='aist_basic:gBR:ch01'
    changed=dict(row,sequence_id=n.replace('sBM','sFM'));auth['all'].add(changed['sequence_id']);auth['mapping'][changed['sequence_id']]='setting1'
    assert identify(changed,auth)['status']=='EXCLUDED_UNRESOLVED_ADVANCED_CHOREOGRAPHY'
    auth['urls']=set()
    with pytest.raises(ValueError,match='official refined'):identify(row,auth)
