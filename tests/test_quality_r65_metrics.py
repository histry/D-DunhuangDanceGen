"""TEST_FIXTURE synthetic timing/ordinal diagnostics, never real human evidence."""
from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest

from dunhuang.research.quality_r65.metrics import event_match,spearman_tied,summarize_reviews
from test_quality_r65_records import manifest,music_doc,quality_doc,checked_roster


def test_event_one_to_one_maximum_cardinality_not_greedy():
    q=event_match([1.,1.11],[.94,1.05],.07,[0,2])
    assert (q['TP'],q['FP'],q['FN'])==(2,0,0)
    assert q['matched_MAE_seconds']==pytest.approx(.06)
    assert q['F1']==1


def test_event_minimum_error_and_duplicate_detection_penalty():
    q=event_match([1.],[.95,1.01,1.01],.1,[0,2])
    assert (q['TP'],q['FP'],q['FN'])==(1,2,0)
    assert q['matched_MAE_seconds']==pytest.approx(.01)
    assert q['precision']==pytest.approx(1/3)


def test_matching_agrees_with_small_exhaustive_optimum():
    R=[.2,.3,.8];E=[.22,.29,.7,.82];tol=.15
    q=event_match(R,E,tol,[0,1])
    possibilities=[]
    for count in range(4):
        for rr in combinations(range(len(R)),count):
            for ee in combinations(range(len(E)),count):
                ds=[abs(R[i]-E[j]) for i,j in zip(rr,ee)]
                if all(d<=tol for d in ds):possibilities.append((count,sum(ds)))
    best=min(possibilities,key=lambda z:(-z[0],z[1]))
    assert q['TP']==best[0]
    assert sum(r['absolute_error_seconds'] for r in q['matches'])==pytest.approx(best[1])


def test_empty_unknown_is_not_perfect_and_no_unknown_bridge():
    q=event_match([],[],.1,[0,2])
    assert q['F1'] is q['matched_MAE_seconds'] is q['precision'] is q['recall'] is None
    q=event_match([.99,1.2],[1.01,1.2],.1,[0,2],excluded=[[1.,1.1]])
    assert q['TP']==1 and q['FN']==1 and q['estimate_in_unknown']==1
    assert event_match([.99],[1.11],.2,[0,2],[[1.,1.1]])['TP']==0


def test_evaluation_interval_is_absolute_and_half_open():
    q=event_match([2.,3.,4.],[2.01,3.01,4.],.1,[2,4])
    assert q['TP']==2 and q['reference_outside_interval']==1 and q['estimate_outside_interval']==1
    assert q['matches'][0]['reference_seconds']==2.


@pytest.mark.parametrize('args', [([True],[],.1,[0,2]),([float('nan')],[],.1,[0,2]),
    ([],[],True,[0,2]),([],[],0,[0,2]),([],[],.1,[2,2]),([-1],[],.1,[0,2])])
def test_bad_event_inputs(args):
    with pytest.raises(ValueError):event_match(*args)


def test_exclusions_must_have_known_support():
    with pytest.raises(ValueError):event_match([],[],.1,[0,2],[[-1,1]])
    with pytest.raises(ValueError):event_match([],[],.1,[0,2],[[1,3]])


def test_spearman_coefficients_ties_and_constants():
    assert spearman_tied([1,2,3,4],[4,3,2,1])==pytest.approx(-1.)
    assert spearman_tied([1,1,2,3],[5,5,7,9])==pytest.approx(1.)
    assert spearman_tied([1,1,1,1],[1,2,3,4]) is None
    assert spearman_tied([],[]) is None
    assert spearman_tied([1],[1]) is None
    with pytest.raises(ValueError):spearman_tied([1],[1,2])
    with pytest.raises(ValueError):spearman_tied([True,2],[1,2])


def test_no_documents_stays_not_evaluable(manifest):
    result=summarize_reviews(manifest,[])
    assert result['status']=='NOT_EVALUABLE' and result['independent_person_count']==0
    assert result['music_agreement']==result['proxy_correlations']==[]
    assert result['confidence_intervals'] is None and not result['can_route']


def second(doc,roster,same_person=False):
    d=deepcopy(doc);d['review_id']+='_2';d['reviewer_id']='TEST_FIXTURE_PERSON2'
    r=deepcopy(roster);r['entries'].append(dict(r['entries'][0],reviewer_id=d['reviewer_id'],
        person_uid='TEST_FIXTURE_P1' if same_person else 'TEST_FIXTURE_P2'))
    return d,r


def test_two_synthetic_music_records_agreement_not_accuracy(manifest,music_doc,checked_roster):
    other,roster=second(music_doc,checked_roster)
    other['items'][0]['tracks']['beat']['events'][0]['seconds']=1.04
    result=summarize_reviews(manifest,[music_doc,other],roster)
    assert result['independent_person_count']==2
    beat=next(x for x in result['music_agreement'] if x['kind']=='beat')
    assert beat['F1']==1 and beat['matched_MAE_seconds']==pytest.approx(.04)
    assert 'NOT_ACCURACY' in beat['interpretation']


def test_roster_alias_cannot_manufacture_independent_human(manifest,music_doc,checked_roster):
    other,roster=second(music_doc,checked_roster,same_person=True)
    result=summarize_reviews(manifest,[music_doc,other],roster)
    assert result['independent_person_count']==1 and result['music_agreement']==[]
    assert sum(q.get('duplicate_person_document',False) for q in result['qualifications'])==1


def test_synthetic_four_method_ranks_correlations_and_baseline_differences(manifest,quality_doc,checked_roster):
    result=summarize_reviews(manifest,[quality_doc],checked_roster)
    raw=result['quality_reviews'][0]
    assert raw['rank_groups_descending']==[['ShuffledBeat'],['LocalBeat'],['BeatOnly'],['Fixed48']]
    assert all(r['spearman']==pytest.approx(1.) for r in result['proxy_correlations'])
    q=next(r for r in result['fixed48_differences'] if r['method']=='LocalBeat' and r['dimension']=='continuity')
    assert q['difference']==2 and q['qualified']
    assert result['confidence_intervals'] is None


def test_ties_unsure_partial_and_unverified_not_scored_as_success(manifest,quality_doc,checked_roster):
    for r in quality_doc['items'][0]['ratings']:r['scores']['richness']=dict(value=3,reason=None)
    result=summarize_reviews(manifest,[quality_doc],checked_roster)
    assert next(r for r in result['proxy_correlations'] if r['dimension']=='richness')['spearman'] is None
    quality_doc['items'][0]['ratings'][0]['scores']['continuity']=dict(value=None,reason='UNSURE')
    result=summarize_reviews(manifest,[quality_doc],checked_roster)
    assert result['status']=='DESCRIPTIVE_SINGLE_PERSON_PER_TASK'
    raw=next(r for r in result['quality_reviews'] if r['dimension']=='continuity')
    assert raw['scores']['Fixed48'] is None and raw['unsure_reasons']['Fixed48']=='UNSURE'
    # This abstention keeps the other three method scores, and other dimensions.
    assert all(r['spearman']==pytest.approx(1.) for r in result['proxy_correlations'] if r['dimension']=='continuity')
    assert not result['two_person_quality_requirement_met']
    q=next(r for r in result['fixed48_differences'] if r['dimension']=='continuity')
    assert q['difference'] is None and q['missing_reasons']['Fixed48']=='UNSURE'


def test_two_rating_reviewers_keep_paired_case_agreement(manifest,quality_doc,checked_roster):
    other,roster=second(quality_doc,checked_roster)
    result=summarize_reviews(manifest,[quality_doc,other],roster)
    assert len(result['quality_agreement'])==4
    assert all(r['exact_agreement_count']==4 and r['mean_absolute_ordinal_difference']==0 for r in result['quality_agreement'])


def test_partial_track_keeps_other_music_kinds_and_common_reviewed_interval(manifest,music_doc,checked_roster):
    other,roster=second(music_doc,checked_roster)
    other['items'][0]['tracks']['phrase']['status']='UNSURE'
    other['items'][0]['coverage_seconds']=[0.,5.]
    result=summarize_reviews(manifest,[music_doc,other],roster)
    assert result['qualified_documents']==2 and result['two_person_music_requirement_met']
    assert len(result['music_agreement'])==3
    assert all(r['interval_seconds']==[0.,5.] for r in result['music_agreement'])
    assert result['music_not_evaluable'][0]['kind']=='phrase'


def test_partial_rating_keeps_numeric_pair_agreement(manifest,quality_doc,checked_roster):
    other,roster=second(quality_doc,checked_roster)
    other['items'][0]['ratings'][0]['scores']['continuity']=dict(value=None,reason='UNRATED')
    result=summarize_reviews(manifest,[quality_doc,other],roster)
    q=next(r for r in result['quality_agreement'] if r['dimension']=='continuity')
    assert q['rated_method_count']==3 and q['exact_agreement_count']==3
    assert q['mean_absolute_ordinal_difference']==0
    assert result['two_person_quality_requirement_met']


def test_duplicate_documents_or_unregistered_tolerances_rejected(manifest,music_doc):
    with pytest.raises(ValueError):summarize_reviews(manifest,[music_doc,music_doc])
    with pytest.raises(ValueError):summarize_reviews(manifest,[],tolerances={'beat':.1})
    with pytest.raises(ValueError):summarize_reviews(manifest,[],tolerances=dict(beat=True,downbeat=.1,accent=.1,phrase=.5))
