import pytest
from dunhuang.research.gtw_kinematic_baseline.reviews import summarize,validate_phases,validate

def test_empty_records_remain_not_evaluable_not_ties():
    x=summarize([],dict(pack_id='test',items=[]))
    assert x['states']=={} and x['quality']=='NOT_EVALUABLE' and x['source_groups']==[]

def test_actual_phase_support_and_source_hash_are_required():
    src={'s':dict(native_sha256='a'*64,frames=300)}
    d=dict(schema='SOURCE_PHASE_V1',source_uid='s',source_sha256='a'*64,annotator_id='fixture',intervals=[dict(start=0,end=300,role='UNKNOWN')])
    assert validate_phases(d,src)['intervals'][0]['role']=='UNKNOWN'
    d['intervals'][0]['end']=301
    with pytest.raises(ValueError):validate_phases(d,src)
