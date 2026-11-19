import copy
import pytest
from dunhuang.research.edit_support_r61.intervention import choose

def source(kind='speed_peak',span=(100,103)):
    return dict(phase_evidence=[dict(type=kind,frame=(sum(span)-1)//2,body_region='upper',evidence_span=list(span))],accepted_components=[[0,500]])

def test_peak_deletion_not_band_crossing_and_no_added_neighbor():
    s=source();r=choose(s,'speed_peak');assert r['status']=='MATCHED'
    assert r['A5'][0]==r['A4'][0] and r['A5'][1]<r['A4'][1]
    assert r['added_frames']==0 and r['removed_raw_frames']==3 and r['existing_center_removed']
    assert r['A5'][1]<=s['phase_evidence'][0]['evidence_span'][0]

@pytest.mark.parametrize('kind,span',[('pose_hold',(100,180)),('root_turn_span',(100,120))])
def test_continuous_raw_support_actually_truncated(kind,span):
    r=choose(source(kind,span),kind);assert r['removed_raw_frames']==(span[1]-span[0])//2
    assert r['raw_previously_fully_in_A4'] and r['source_length_A5']<r['source_length_A4']

def test_quality_gap_blocks_view_and_real_context():
    s=source();s['accepted_components']=[[0,80],[81,500]]
    assert choose(s,'speed_peak')['status']=='NOT_MATCHABLE'

def test_exposed_six_intervals_excluded_before_scores():
    assert choose(source(),'speed_peak',[[60,110]])['status']=='NOT_MATCHABLE'

def test_turn_too_short_kept_out_without_retuning_detector():
    assert choose(source('root_turn_span',(100,106)),'root_turn_span')['status']=='NOT_MATCHABLE'

def test_selection_is_time_metadata_not_invented_scores():
    s=source();later=copy.deepcopy(s['phase_evidence'][0]);later.update(frame=201,evidence_span=[200,203]);s['phase_evidence'].insert(0,later)
    assert choose(s,'speed_peak')['evidence']['frame']==101
