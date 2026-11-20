"""Synthetic review contracts; fixtures are not human evaluation records."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).with_name('reviews.py')
if not path.is_file():path=Path(__file__).resolve().parents[1]/'src/dunhuang/research/retiming_baselines_r69/reviews.py'
spec=importlib.util.spec_from_file_location('r69_reviews_tested',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def admin():
    return dict(schema='R69_ADMIN_V1',pack_id='SYNTHETIC_PACK_NOT_REAL_REVIEW',protocol_sha256='a'*64,form_id='F01',cases=[
        dict(blind_item_id='blind'+str(i),case_key='case'+str(i),source_uid='source',source_interval=[100,340],variants=[
            dict(blind_variant_id='v'+str(i)+'_'+str(j),method=m,available=True,trajectory_sha256='b'*64)
            for j,m in enumerate(r.METHODS)]) for i in range(2)])


def document(reviewer='r1',review_id='review1',scores=None):
    if scores is None:scores={'Uniform':2,'E1W':4}
    variants={v['method']:v['blind_variant_id'] for v in admin()['cases'][0]['variants']}
    return dict(schema='R69_QUALITY_REVIEW_V1',pack_id=admin()['pack_id'],protocol_sha256='a'*64,
        review_id=review_id,reviewer_id=reviewer,reviewer_roster_ref='private/'+reviewer,
        reviewed_at='2026-11-20T12:00:00+08:00',form_id='F01',status='PARTIAL',
        independence=dict(own_judgment=True),exposure={k:'NO' for k in r.EXPOSURE},items=[
            dict(blind_item_id='blind0',ratings=[dict(blind_variant_id=variants[m],dimensions=dict(
                music_coordination=dict(status='RATED',value=v))) for m,v in scores.items()])])


def roster(aliases=False):
    return dict(schema='R69_REVIEWER_ROSTER_V1',pack_id=admin()['pack_id'],reviewers=[
        dict(reviewer_id=reviewer,reviewer_roster_ref='private/'+reviewer,person_uid='person1' if aliases else 'person'+str(i),
            identity_manually_verified=True,exposure_manually_checked=True,verified_by='synthetic_admin_fixture',
            verified_at='2026-11-20T11:00:00+08:00',exposure={k:False for k in r.EXPOSURE})
        for i,reviewer in enumerate(('r1','r2'))])


def test_no_records_and_self_report_without_roster_remain_not_evaluable():
    empty=r.summarize_reviews(admin(),[])
    assert empty['status']=='NOT_EVALUABLE' and empty['human_reviewed'] is False
    assert empty['records']==[] and empty['admin_attested_distinct_people_with_paired_scores']==0
    one=r.summarize_reviews(admin(),[document()])
    assert one['status']=='NOT_EVALUABLE' and one['records'][0]['status']=='UNVERIFIED'
    contrast=one['comparisons']['E1W_minus_Uniform']['music_coordination']
    assert contrast['paired_records']==1 and contrast['raw_paired_records'][0]['difference']==2
    assert contrast['source_interval_balanced_mean'] is None and contrast['evaluable_cases']==0


def test_partial_unsure_and_unrated_do_not_become_scores_or_complete_records(tmp_path):
    doc=document();doc['items'][0]['ratings'][1]['dimensions']['music_coordination']=dict(status='UNSURE',reason='Difficult to compare')
    doc['items'][0]['ratings'][1]['dimensions']['naturalness']=dict(status='UNRATED')
    saved=tmp_path/'synthetic_review.json';saved.write_text(json.dumps(doc),encoding='utf8')
    normalized=r.load_documents([saved],admin())[0]
    assert normalized['status']=='PARTIAL' and normalized['coverage']['status']=='PARTIAL'
    assert normalized['coverage']['rated_cells']==1 and normalized['coverage']['unsure_cells']==1
    assert normalized['coverage']['explicit_unrated_cells']==1 and normalized['coverage']['missing_cells']==57
    assert 'value' not in normalized['items'][0]['ratings'][1]['dimensions']['music_coordination']
    analysis=r.summarize_reviews(admin(),[doc])['comparisons']['E1W_minus_Uniform']['music_coordination']
    assert analysis['paired_records']==0 and analysis['source_interval_balanced_mean'] is None
    assert analysis['missing_status_counts']=={'UNSURE/RATED':1,'MISSING/MISSING':1}


def test_two_reviewer_aliases_cannot_substitute_for_two_distinct_people():
    docs=[document(),document('r2','review2',{'Uniform':1,'E1W':4})]
    alias=r.summarize_reviews(admin(),docs,roster(aliases=True))
    assert alias['admin_attested_distinct_people_with_paired_scores']==1
    assert alias['independent_comparison_condition_met'] is False
    eligible=r.summarize_reviews(admin(),docs,roster())
    contrast=eligible['comparisons']['E1W_minus_Uniform']['music_coordination']
    assert eligible['records'][0]['status']=='ADMIN_ATTESTED_ELIGIBLE'
    assert eligible['admin_attested_distinct_people_with_paired_scores']==2
    assert contrast['evaluable_cases']==1 and contrast['source_interval_balanced_mean']==2.5
    assert contrast['case_results'][0]['admin_attested_distinct_persons']==2
    assert contrast['case_results'][1]['independent_comparison_condition_met'] is False
    assert contrast['ci95'] is None and contrast['p_value'] is None
    assert eligible['comparisons']['E1W_minus_Uniform']['naturalness']['source_interval_balanced_mean'] is None
    assert eligible['quality_winner'] is None and eligible['can_route'] is False


def test_manual_identity_and_exposure_attestation_are_both_required():
    doc=r.validate_document(document(),admin())
    reg=roster();reg['reviewers'][0]['identity_manually_verified']=False
    assert r.qualify(doc,admin(),reg)['eligible'] is False
    reg=roster();reg['reviewers'][0]['exposure']['prior_named_results_seen']=True
    assert r.qualify(doc,admin(),reg)['eligible'] is False
    exposed=copy.deepcopy(doc);exposed['exposure']['other_reviews_seen']='UNSURE'
    assert r.qualify(exposed,admin(),roster())['eligible'] is False
    assert r.qualify(doc,admin(),roster())['status']=='ADMIN_ATTESTED_ELIGIBLE'


@pytest.mark.parametrize('problem',['default_score','unavailable','wrong_pack','bad_score','naive_time','duplicate_item','missing_exposure'])
def test_inconsistent_imports_fail_closed(problem):
    packet=admin();doc=document()
    if problem=='default_score':doc['items'][0]['ratings'][0]['dimensions']['music_coordination']=dict(status='UNRATED',value=3)
    elif problem=='unavailable':packet['cases'][0]['variants'][0]['available']=False;packet['cases'][0]['variants'][0]['trajectory_sha256']=None
    elif problem=='wrong_pack':doc['pack_id']='other'
    elif problem=='bad_score':doc['items'][0]['ratings'][0]['dimensions']['music_coordination']['value']=True
    elif problem=='naive_time':doc['reviewed_at']='2026-11-20T12:00:00'
    elif problem=='duplicate_item':doc['items'].append(copy.deepcopy(doc['items'][0]))
    else:doc['exposure'].pop('prior_named_results_seen')
    with pytest.raises(ValueError):r.validate_document(doc,packet)


def test_repeated_versions_of_one_person_do_not_inflate_case_denominator():
    doc=document();again=document('r1','review1_revision',{'Uniform':1,'E1W':4})
    analysis=r.summarize_reviews(admin(),[doc,again],roster())
    contrast=analysis['comparisons']['E1W_minus_Uniform']['music_coordination']
    assert contrast['paired_records']==2 and contrast['case_results'][0]['admin_attested_distinct_persons']==1
    assert contrast['source_interval_balanced_mean'] is None
    with pytest.raises(ValueError,match='Duplicate review_id'):
        r.summarize_reviews(admin(),[doc,copy.deepcopy(doc)],roster())
