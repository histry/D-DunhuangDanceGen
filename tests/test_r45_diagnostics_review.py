import datetime
import numpy as np
import pytest
from scipy.interpolate import PPoly
from dunhuang.research.phase_r45.attribution import integrated_terms,stencil
from dunhuang.research.target_review_r45.reviews import ndcg,resolve,validate
from dunhuang.research.target_review_r45.prepare import sample_bounds

def test_partial_final_audio_bin_preserves_real_supported_pose_center():
    start,finish,r=sample_bounds([3284,3598],16000,1918433);assert finish==1918433;assert r['partial_final_audio_bin'] and r['last_pose_center_observed'];assert r['requested_sample_interval'][1]-finish==500

def test_genuine_missing_pose_center_audio_is_not_padded():
    with pytest.raises(ValueError):sample_bounds([3284,3599],16000,1918433)

def test_integrated_chain_rule_recovers_actual_discrete_third_difference():
    poly=PPoly(np.array([[.01],[1.],[0.]]),[0,100])
    def derivatives(q):return [3*q[:,None,None]**2,6*q[:,None,None],np.full((len(q),1,1),6.)]
    target=np.arange(10);actual=np.diff(poly(target)**3,n=3)*27000
    for order in (3,5):
        terms=integrated_terms(np.arange(7),poly,derivatives,order);np.testing.assert_allclose(sum(terms)[:,0,0],actual,rtol=1e-10,atol=1e-6)

def test_source_derivative_stencil_has_correct_units_and_sign():
    h=.025;x=2+np.arange(-3,4)*h;assert abs(stencil(3,h)@x**5-240)<1e-6

def test_tied_ndcg_and_empty_relevance_are_not_faked_accuracy():
    assert ndcg([0,0],[2,0],1)==.5;assert ndcg([1,0],[0,0],5) is None

def test_two_versions_from_one_person_do_not_create_independence():
    records=[dict(reviewer_id='r1',review_id='x',decision='VALID'),dict(reviewer_id='r1',review_id='y',decision='VALID')];assert resolve(records)['state']=='PARTIAL_REVIEW'

def test_uncertainty_and_disagreement_need_distinct_handling():
    a=dict(reviewer_id='r1',review_id='x',decision='UNSURE');b=dict(reviewer_id='r2',review_id='y',decision='UNSURE');assert resolve([a,b])['state']=='UNSURE';b['decision']='VALID';assert resolve([a,b])['decision'] is None

def test_adjudicator_cannot_be_one_of_original_reviewers():
    a=dict(reviewer_id='r1',review_id='x',decision='VALID');b=dict(reviewer_id='r2',review_id='y',decision='INVALID');adj=dict(reviewer_id='r1',review_id='z',decision='VALID',resolves_review_ids=['x','y']);assert resolve([a,b],adj)['state']=='PARTIAL_REVIEW';adj['reviewer_id']='r3';assert resolve([a,b],adj)['state']=='HUMAN_REVIEW_VERIFIED'

@pytest.mark.parametrize('mutation',[dict(demonstration=True),dict(media_ref='wrong')])
def test_demo_or_wrong_media_never_becomes_human_truth(mutation):
    r=dict(review_id='x',blind_item_id='b',reviewer_id='r',task_type='POSE',decision='UNSURE',confidence=1,reason_code='VISIBILITY_INSUFFICIENT',reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),media_ref='known',review_protocol_version='DUNHUANG_BLIND_REVIEW_V1_R45',demonstration=False,status='UNVERIFIED_CLIENT_RECORDS');r.update(mutation);people=dict(r=dict(identity_verified=True,independent_from_model_development=True,reviewer_roster_ref='fixture_only'))
    with pytest.raises(ValueError):validate(r,dict(b=dict(media_ref=dict(POSE='known'))),{},people)
