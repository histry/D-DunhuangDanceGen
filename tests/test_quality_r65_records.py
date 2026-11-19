"""TEST_FIXTURE only: synthetic in-memory identities/media, no real reviews."""
from copy import deepcopy
import math

import pytest

from dunhuang.research.quality_r65.records import (
    METHODS, DIMENSIONS, TRACKS, validate_document, qualify)

TEST_FIXTURE = 'SYNTHETIC_IN_MEMORY_ONLY_NOT_HUMAN_EVIDENCE'


@pytest.fixture
def manifest():
    variants = [dict(blind_variant_id='V'+str(i), method=name,
        trajectory_sha256=('%064x' % (i+1)),
        proxies=dict(beat=i/4, accent=i/5, section=i/6, primary=i/7))
        for i, name in enumerate(METHODS)]
    return dict(pack_id='TEST_FIXTURE', protocol_sha256='a'*64,
        music=[dict(blind_item_id='M01', raw_audio_sha256='b'*64,
                    media_sha256='c'*64, duration_seconds=10.)],
        cases=[dict(blind_item_id='C01', case_key='TEST_FIXTURE_CASE', role='EVAL',
                    music_id='M01', music_interval_seconds=[0., 10.], variants=variants)],
        forms=[dict(form_id='F01', case_order=['C01'], panel_orders={'C01':['V2','V0','V3','V1']})])


def base(schema):
    return dict(schema=schema, pack_id='TEST_FIXTURE', protocol_sha256='a'*64,
        review_id='TEST_FIXTURE_R1', reviewer_id='TEST_FIXTURE_PERSON1',
        reviewer_roster_ref='TEST_FIXTURE_PRIVATE', form_id='F01',
        reviewed_at='2026-11-20T11:00:00+08:00',
        independence=dict(own_judgment=True, other_reviews_seen=False,
                          method_identity_seen=False, prior_r64_exposure=False))


@pytest.fixture
def music_doc():
    doc = base('R65_MUSIC_REVIEW_V1')
    doc['items'] = [dict(blind_item_id='M01', media_sha256='c'*64,
        coverage_seconds=[0.,10.], unsure_intervals=[], tracks={
            k:dict(status='COMPLETE', events=[dict(seconds=1.,confidence='HIGH')]) for k in TRACKS})]
    return doc


@pytest.fixture
def quality_doc(manifest):
    doc = base('R65_QUALITY_REVIEW_V1')
    doc['items'] = [dict(blind_item_id='C01',variant_order=['V2','V0','V3','V1'],
        ratings=[dict(blind_variant_id='V'+str(i),scores={
            k:dict(value=i+1,reason=None) for k in DIMENSIONS}) for i in range(4)])]
    return doc


@pytest.fixture
def checked_roster():
    # A caller-supplied assertion is tested here; this is not a real identity.
    return dict(entries=[dict(reviewer_id='TEST_FIXTURE_PERSON1',reviewer_roster_ref='TEST_FIXTURE_PRIVATE',
        person_uid='TEST_FIXTURE_P1',assigned_form_id='F01',verified=True,
        authentication_evidence_sha256='d'*64,independence_verified=True)])


def test_validation_detaches_and_never_grants_human_truth(manifest,music_doc,quality_doc):
    for doc in (music_doc,quality_doc):
        before=deepcopy(doc); normalized=validate_document(doc,manifest)
        assert normalized==before and normalized is not doc and doc==before
        q=qualify(doc,manifest)
        assert q['status']=='IMPORTED_UNVERIFIED'
        assert not q['contract_qualified'] and not q['operational_qualified']
        assert q['semantic_truth']=='UNKNOWN'


def test_private_roster_assertion_and_assignment(manifest,quality_doc,checked_roster):
    q=qualify(quality_doc,manifest,checked_roster)
    assert q['status']=='CONTRACT_QUALIFIED' and q['person_uid']=='TEST_FIXTURE_P1'
    assert not q['operational_qualified']
    for key,value in [('verified',False),('independence_verified',False),
                      ('assigned_form_id','F02'),('authentication_evidence_sha256','invalid'),
                      ('reviewer_roster_ref','other')]:
        bad=deepcopy(checked_roster);bad['entries'][0][key]=value
        assert qualify(quality_doc,manifest,bad)['status']=='IMPORTED_UNVERIFIED'
    duplicate=deepcopy(checked_roster);duplicate['entries']*=2
    assert qualify(quality_doc,manifest,duplicate)['status']=='IMPORTED_UNVERIFIED'


@pytest.mark.parametrize('key,value', [('pack_id','other'),('protocol_sha256','e'*64),
    ('form_id','F02'),('review_id',''),('reviewed_at','2026-11-20T11:00:00'),
    ('reviewed_at','not-a-time')])
def test_tampered_top_identity_time_rejected(manifest,music_doc,key,value):
    music_doc[key]=value
    with pytest.raises(ValueError):validate_document(music_doc,manifest)


@pytest.mark.parametrize('value', [True,False,float('nan'),float('inf'),-1.,10.])
def test_invalid_event_timestamp(manifest,music_doc,value):
    music_doc['items'][0]['tracks']['beat']['events'][0]['seconds']=value
    with pytest.raises(ValueError):validate_document(music_doc,manifest)


def test_event_order_duplicates_and_media_identity(manifest,music_doc):
    for modification in ('media','duplicate','order','confidence'):
        d=deepcopy(music_doc);item=d['items'][0];events=item['tracks']['beat']['events']
        if modification=='media':item['media_sha256']='d'*64
        elif modification=='duplicate':events.append(deepcopy(events[0]))
        elif modification=='order':events.insert(0,dict(seconds=2.,confidence='HIGH'))
        else:events[0]['confidence']='CERTAIN'
        with pytest.raises(ValueError):validate_document(d,manifest)


def test_partial_and_unsure_are_retained_not_qualified(manifest,music_doc,quality_doc,checked_roster):
    music_doc['items'][0]['tracks']['phrase']['status']='UNSURE'
    q=qualify(music_doc,manifest,checked_roster)
    assert q['status']=='PARTIAL_REVIEW' and q['contract_qualified'] and not q['coverage_complete']
    music_doc['items']=[]
    assert qualify(music_doc,manifest,checked_roster)['status']=='PARTIAL_REVIEW'
    quality_doc['items'][0]['ratings'][0]['scores']['continuity']=dict(value=None,reason='UNSURE')
    d=validate_document(quality_doc,manifest)
    assert d['items'][0]['ratings'][0]['scores']['continuity']['value'] is None
    q=qualify(d,manifest,checked_roster)
    assert q['status']=='PARTIAL_REVIEW' and q['contract_qualified'] and not q['coverage_complete']


def test_explicit_reviewed_empty_and_unknown_coverage(manifest,music_doc,checked_roster):
    track=music_doc['items'][0]['tracks']['phrase'];track['events']=[]
    with pytest.raises(ValueError):validate_document(music_doc,manifest)
    track['status']='NO_CLEAR_EVENTS'
    assert qualify(music_doc,manifest,checked_roster)['contract_qualified']
    music_doc['items'][0]['unsure_intervals']=[dict(start=2.,end=3.,reason='UNSURE')]
    assert qualify(music_doc,manifest,checked_roster)['status']=='PARTIAL_REVIEW'


@pytest.mark.parametrize('dimension', list(DIMENSIONS))
@pytest.mark.parametrize('value,reason', [(True,None),(1.,None),(0,None),(6,None),(2,'UNSURE'),(None,None)])
def test_bad_ordinal_or_contradictory_abstention(manifest,quality_doc,dimension,value,reason):
    quality_doc['items'][0]['ratings'][0]['scores'][dimension]=dict(value=value,reason=reason)
    with pytest.raises(ValueError):validate_document(quality_doc,manifest)


def test_missing_duplicate_or_reordered_blind_variants(manifest,quality_doc):
    for change in ('order','missing','duplicate','dimension'):
        d=deepcopy(quality_doc);item=d['items'][0]
        if change=='order':item['variant_order'].reverse()
        elif change=='missing':item['ratings'].pop()
        elif change=='duplicate':item['ratings'][1]['blind_variant_id']='V0'
        else:item['ratings'][0]['scores'].pop('richness')
        with pytest.raises(ValueError):validate_document(d,manifest)


def test_exposed_review_kept_but_excluded(manifest,quality_doc,checked_roster):
    for field,value in [('prior_r64_exposure',True),('method_identity_seen',True),
                        ('other_reviews_seen',True),('own_judgment',False)]:
        d=deepcopy(quality_doc);d['independence'][field]=value
        assert qualify(d,manifest,checked_roster)['status']=='EXCLUDED_EXPOSED'
    quality_doc['independence']['own_judgment']=1
    with pytest.raises(ValueError):validate_document(quality_doc,manifest)


def test_missing_or_duplicate_manifest_id_is_error(manifest,quality_doc):
    manifest['cases'][0]['music_id']='OTHER'
    with pytest.raises(ValueError):validate_document(quality_doc,manifest)
